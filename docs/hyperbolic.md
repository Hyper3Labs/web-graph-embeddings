# Hyperbolic decoder (experimental)

`--decoder hyperbolic` trains the same shallow, featureless host table as the published artifact,
but scores a link by **distance in hyperbolic space** instead of a dot product. It is opt-in: the
default `--decoder dot` is the published model, and its code path is unchanged.

Status: implemented, unit-tested on CPU, and compared against `dot` on a sample of the Common Crawl
host graph (results below). **It has not been run on the 4×H100 WholeGraph path**; see
[Caveats](#caveats).

## Why hyperbolic

The web graph is heavy-tailed and loosely hierarchical: a few hubs, many communities, a long tail.
Hyperbolic space has room for that. Its volume grows exponentially with radius, so trees and
power-law graphs embed with low distortion in very few dimensions (Nickel & Kiela 2017, 2018;
Krioukov et al. 2010, who also show that network popularity and similarity map onto a point's
radius and angle). The hope is the same link-prediction quality in fewer dimensions, and a radius
that reads directly as "how central is this host".

## Using it

Add these flags to the stage 5 command in `REPRODUCE.md`. The learning rate is 10× the published
dot rate and has not been tried at full scale.

```bash
--decoder hyperbolic --lr 0.03 \
--hyp-margin 2 --hyp-temp 1 --hyp-max-norm 20 --hyp-init-radius 2
```

Hyperbolic training needs roughly a **10× higher learning rate** than `dot` (on the sample below the
best rates were 0.1 vs 0.01 at batch 4,096; the published dot run used 0.003 at 32,768/rank), and a
**non-zero init radius** (default 2; see below).

Stage 6 reads `decoder` from the trainer's JSON sidecar, so `wgl release shard-export --meta …`
publishes a hyperbolic table **un-normalized** (L2-normalizing it would erase every host's radius and
is refused) and the manifest records `distance: "hyperbolic"` plus a `geometry` block.

Consumers rank neighbours by hyperbolic distance, not cosine:

```python
import numpy as np, pyarrow.parquet as pq
from wgl import hyperbolic

t = pq.read_table("vectors/emb-00000-of-00014.parquet")
vecs = np.stack(t["embedding"].to_numpy(zero_copy_only=False))  # [n, dim] tangent vectors
d = hyperbolic.distance(vecs[0], vecs)                          # geodesic distance to row 0
nearest = np.argsort(d)[:10]
radius = np.linalg.norm(vecs, axis=1)                           # distance from the origin
```

`hyperbolic.to_lorentz` / `to_poincare` convert to the two standard models (e.g. for plotting, or a
max-inner-product index over Lorentz coordinates; see the function docstrings).

## How it works

All of the geometry is in `scripts/cupyg/geometry.py`.

- **The table stores tangent vectors at the origin.** A host's point is `exp_0(v)` on the
  hyperboloid (Lorentz model, curvature −1) and `|v|` is exactly its distance from the origin. The
  optimisation variable is still a plain Euclidean `[N, dim]` table, so the WholeMemory sharding,
  its sparse Adam, checkpoints and the chunked export work unchanged. No Riemannian optimizer is
  needed.
- **Decoder:** Fermi-Dirac, `logit = (margin − d(u, v)) / temperature` (Chami et al. 2019), trained
  with the same BCE-over-negatives objective as `dot`. Eval ranks by the same logit, which is
  monotone in `−d`.
- **Numerically stable distance.** `cosh d = cosh(a − b) + sinh a · sinh b · (1 − cos θ)` with
  `a = |u|`, `b = |v|`. Every term is non-negative, so there is no cancellation, and `cosh d − 1` is
  formed directly. In fp32 the textbook `x₀y₀ − ⟨x, y⟩` gives 30% errors and NaNs for nearby points
  at radius 12; this form stays near 1e-5 relative error. Radii are clipped at `--hyp-max-norm`
  (≤ 40) to stay inside fp32 range.
- **Initialisation.** Both table inits (WholeMemory's Xavier-uniform, ~1e-3 at full scale, and the
  dense `N(0, 0.1)`) put every host near the origin. For a dot product that is neutral. In
  hyperbolic space the origin is close to *everything*, so hosts that are rarely sampled as
  negatives outrank true neighbours, and early MRR comes out *below chance*. `--hyp-init-radius`
  starts every host on a sphere of that radius instead. On the WholeGraph path this rewrites each
  rank's local shard in place, the way `create_embedding(random_init=True)` wrote it.
- **Curvature** is not a separate flag: changing it is equivalent to rescaling the table, margin and
  temperature.

## Results on a Common Crawl sample

**Setup.** One edge shard (1/144 of the edges) of `cc-main-2025-26-nov-dec-jan`. From it: the
induced subgraph on the 400k highest-degree hosts, edges subsampled to 9%, then a 4-core. That leaves
**358,660 hosts and 4.92 M edges** with a heavy-tailed degree distribution (median 12, p99 528). The
split is the trainer's 80/10/10. Each run is 3 epochs with a cosine schedule, batch 4,096, 50
in-batch negatives (the published objective), and the best-val epoch is kept. The table reports
test MRR / hits against 100 uniform negatives on 50k held-out edges. Learning rate was tuned per
decoder at dim 32: dot 0.01, hyperbolic 0.1 with `--hyp-init-radius 2`.

| dim | dot MRR | hyp MRR | dot hits@1 | hyp hits@1 | dot hits@10 | hyp hits@10 |
|--:|--:|--:|--:|--:|--:|--:|
| 8 | **0.946** | 0.797 | **0.918** | 0.656 | 0.979 | **0.992** |
| 16 | **0.956** | 0.901 | **0.932** | 0.825 | 0.985 | **0.998** |
| 32 | **0.957** | 0.949 | **0.933** | 0.911 | 0.986 | **0.998** |
| 64 | 0.939 | **0.955** | 0.900 | **0.921** | 0.987 | **0.998** |

What this sample shows:

- **hits@10 is consistently better with hyperbolic** (≥ 0.992 at every dim, vs 0.979–0.987). It
  almost never loses a true neighbour from the top 10, which matters for candidate generation.
- **Top-1 is worse at low dimension** and the gap closes as dim grows. At dim 64 hyperbolic is
  ahead, but dot-64 scored below dot-32, so it is probably under-trained at the dim-32 learning rate;
  treat that row as inconclusive.
- **Hyperbolic does not win at low dimension here**, contrary to the usual pitch. It also loses on
  *reconstruction* (ranking training edges) at dim 8 (0.816 vs 0.956), so the gap is underfitting,
  not dot overfitting.
  Uniform instead of in-batch negatives did not change it either (dim 8: 0.795).
- **It needs care to train.** Hyperbolic needs a ~10× higher learning rate and a non-zero init
  radius. With the defaults it inherits from `dot`, dim-32 MRR was 0.77, and early epochs score
  *below chance*. It peaks a little earlier: by epoch 3 at dims 16–32 hits@1 falls while hits@10
  keeps rising, so watch `--patience` / keep-best.
- **Radius tracks degree in under-trained runs** (Spearman −0.98 between radius and degree after one
  epoch at a low learning rate: hubs near the origin), but much less in the tuned runs.

Whether the full 52.9 M-host graph at dim 128 behaves like this sample is open; it needs a GPU run.

## Caveats

- **Not run on GPU.** Everything above ran on CPU through the trainer's own modules (`load_split`,
  `LPModel`, `_SamplerFreeEdgeLoader`, `_inbatch_loss`, `eval_mrr_gathered`,
  `export_table_chunked`), with `nn.Embedding(sparse=True)` + `SparseAdam` standing in for the
  WholeMemory table and its sparse Adam. The WholeGraph init call (`get_local_tensor()` + barrier)
  mirrors pylibwholegraph's own `random_init` code but is untested here.
- **Cost.** On CPU a hyperbolic epoch took ~2.5–3× as long as a dot epoch, and the negative
  scoring holds roughly 2–3× the activations (normalized copies and differences of the `[B, K, dim]`
  negative block). GPU throughput is unmeasured.
- **Tooling.** Cosine/dot ANN indexes do not apply directly. Use exact distance, or a
  max-inner-product index over `to_lorentz` coordinates with exact re-ranking.
- **Hyperparameters** were tuned only at dim 32 on one sample, and briefly: learning rate, init
  radius, margin/temperature. Larger margins (4–10) were much worse. Uniform negatives were tried
  and dropped.
