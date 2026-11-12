"""Tests for the hyperbolic decoder (``scripts/cupyg/geometry.py``) and its numpy twin.

Covers the distance itself (against the textbook Poincare/Lorentz formulas, fp32 stability,
gradients), its parity with ``wgl.hyperbolic``, and the trainer plumbing it touches -- the in-batch
loss, the gathered MRR and the chunked export -- all on CPU with the dense ``nn.Embedding`` table.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from cupyg import geometry  # noqa: E402
from cupyg.evaluate import eval_mrr, eval_mrr_gathered, export_table_chunked  # noqa: E402
from cupyg.loader_free import _SamplerFreeEdgeLoader, _SFBatch  # noqa: E402
from cupyg.losses import _inbatch_loss  # noqa: E402
from cupyg.model import LPModel  # noqa: E402

from wgl import hyperbolic  # noqa: E402


def _poincare_distance(x, y):
    num = 2.0 * np.sum((x - y) ** 2, axis=-1)
    den = (1.0 - np.sum(x * x, axis=-1)) * (1.0 - np.sum(y * y, axis=-1))
    return np.arccosh(1.0 + num / den)


def _rand(*shape, scale=1.0, seed=0):
    return np.random.default_rng(seed).standard_normal(shape) * scale


# --- the distance -------------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.01, 0.3, 1.0, 2.0])
def test_distance_matches_poincare_formula(scale):
    u, v = _rand(500, 8, scale=scale, seed=1), _rand(500, 8, scale=scale, seed=2)
    ref = _poincare_distance(hyperbolic.to_poincare(u), hyperbolic.to_poincare(v))
    np.testing.assert_allclose(hyperbolic.distance(u, v), ref, rtol=1e-9, atol=1e-9)


def test_lorentz_points_lie_on_hyperboloid_and_give_cosh_distance():
    u, v = _rand(200, 6, scale=2.0, seed=3), _rand(200, 6, scale=2.0, seed=4)
    x, y = hyperbolic.to_lorentz(u), hyperbolic.to_lorentz(v)
    np.testing.assert_allclose(-(x[:, 0] ** 2) + np.sum(x[:, 1:] ** 2, axis=1), -1.0, atol=1e-8)
    minus_inner = x[:, 0] * y[:, 0] - np.sum(x[:, 1:] * y[:, 1:], axis=1)
    np.testing.assert_allclose(minus_inner, np.cosh(hyperbolic.distance(u, v)), rtol=1e-9)


def test_distance_metric_properties():
    u, v, w = (_rand(300, 5, scale=1.5, seed=s) for s in (5, 6, 7))
    duv, dvw, duw = hyperbolic.distance(u, v), hyperbolic.distance(v, w), hyperbolic.distance(u, w)
    np.testing.assert_allclose(duv, hyperbolic.distance(v, u))  # symmetric
    assert np.all(duw <= duv + dvw + 1e-9)  # triangle inequality
    np.testing.assert_allclose(hyperbolic.distance(u, u), 0.0, atol=1e-5)
    # The tangent norm IS the distance from the origin.
    np.testing.assert_allclose(
        hyperbolic.distance(np.zeros(5), u), np.linalg.norm(u, axis=1), rtol=1e-9
    )


def test_torch_decoder_matches_numpy_twin():
    u, v = _rand(400, 16, scale=3.0, seed=8), _rand(400, 16, scale=3.0, seed=9)
    dec = geometry.HyperbolicDecoder(max_norm=6.0)
    got = dec.distance(torch.from_numpy(u), torch.from_numpy(v)).numpy()
    np.testing.assert_allclose(got, hyperbolic.distance(u, v, max_norm=6.0), rtol=1e-9)


def test_fp32_stable_for_near_points_far_from_origin():
    # The naive x0*y0 - <xs, ys> loses everything here in fp32 (cancellation, NaNs from acosh<1).
    u = _rand(500, 32, seed=10)
    u = u / np.linalg.norm(u, axis=1, keepdims=True) * 12.0
    v = u + _rand(500, 32, scale=1e-3, seed=11)
    ref = hyperbolic.distance(u, v)
    got = geometry.HyperbolicDecoder().distance(
        torch.from_numpy(u).float(), torch.from_numpy(v).float()
    )
    rel = np.abs(got.double().numpy() - ref) / ref
    assert rel.max() < 1e-3


def test_gradients_exact_and_finite_at_degenerate_points():
    dec = geometry.HyperbolicDecoder()
    u = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(dec, (u, v))
    for a, b in [
        (torch.zeros(2, 4), torch.zeros(2, 4)),  # both at the origin
        (torch.ones(2, 4), torch.ones(2, 4)),  # coincident
        (torch.full((2, 4), 50.0), torch.full((2, 4), -50.0)),  # beyond max_norm, antipodal
    ]:
        a.requires_grad_()
        dec(a, b).sum().backward()
        assert torch.isfinite(a.grad).all()


def test_max_norm_clips_radius_and_prepare_matches():
    dec = geometry.HyperbolicDecoder(max_norm=5.0)
    far = torch.tensor([[30.0, 40.0]])  # |v| = 50
    origin = torch.zeros(1, 2)
    assert dec.distance(origin, far).item() == pytest.approx(5.0, rel=1e-5)
    prepared = dec.prepare(torch.tensor([[30.0, 40.0], [0.3, 0.4]]))
    np.testing.assert_allclose(prepared.numpy(), [[3.0, 4.0], [0.3, 0.4]], rtol=1e-6)


def test_candidate_scoring_matches_pairwise():
    h, c = torch.randn(7, 8), torch.randn(7, 5, 8)
    for dec in (geometry.DotDecoder(), geometry.HyperbolicDecoder()):
        pair = torch.stack([dec(h, c[:, k]) for k in range(5)], dim=1)
        torch.testing.assert_close(dec.score_candidates(h, c), pair, rtol=1e-5, atol=1e-5)


def test_init_places_rows_on_sphere():
    table = torch.randn(1000, 8) * 0.01
    geometry.HyperbolicDecoder(init_radius=3.0).init_(table, chunk=300)  # several chunks
    np.testing.assert_allclose(table.norm(dim=1).numpy(), 3.0, rtol=1e-5)
    assert table.mean(dim=0).abs().max() < 0.5  # directions spread, not collapsed
    before = table.clone()
    geometry.HyperbolicDecoder(init_radius=0.0).init_(table)  # 0 keeps the table's own init
    assert torch.equal(table, before)


def test_make_decoder_and_validation():
    assert isinstance(geometry.make_decoder("dot"), geometry.DotDecoder)
    hyp = geometry.make_decoder(
        "hyperbolic", margin=3.0, temperature=0.5, max_norm=10.0, init_radius=1.0
    )
    assert hyp.describe() == {
        "decoder": "hyperbolic",
        "hyp_margin": 3.0,
        "hyp_temp": 0.5,
        "hyp_max_norm": 10.0,
        "hyp_init_radius": 1.0,
    }
    with pytest.raises(ValueError, match="unknown decoder"):
        geometry.make_decoder("euclid")
    with pytest.raises(ValueError, match="temperature"):
        geometry.HyperbolicDecoder(temperature=0.0)
    with pytest.raises(ValueError, match="max_norm"):
        geometry.HyperbolicDecoder(max_norm=geometry.MAX_NORM_LIMIT + 1)
    with pytest.raises(ValueError, match="init_radius"):
        geometry.HyperbolicDecoder(max_norm=5.0, init_radius=6.0)


# --- trainer plumbing ---------------------------------------------------------------------------


def _model(num_nodes, dim, weights=None):
    torch.manual_seed(0)
    model = LPModel(num_nodes, dim, "shallow", 2)
    if weights is not None:
        model.emb.weight.data.copy_(torch.as_tensor(weights, dtype=torch.float32))
    return model


def test_dot_decoder_reproduces_published_loss():
    # The default decoder must leave the published objective bit-for-bit unchanged.
    model = _model(50, 8)
    batch = _SFBatch(np.stack([np.arange(0, 20), np.arange(20, 40)]))
    x = model(batch)
    torch.manual_seed(1)
    got, n = _inbatch_loss(x, batch, 5)
    torch.manual_seed(1)
    src, dst = batch.edge_label_index
    hs = x[src]
    pos = (hs * x[dst]).sum(dim=-1)
    neg_idx = torch.randint(0, x.shape[0], (src.shape[0], 5))
    neg = (hs.unsqueeze(1) * x[neg_idx]).sum(dim=-1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    want = bce(pos, torch.ones_like(pos)) + bce(neg, torch.zeros_like(neg))
    assert torch.equal(got, want)
    assert n == 20 * 6


def test_hyperbolic_inbatch_loss_backprops():
    model = _model(50, 8)
    batch = _SFBatch(np.stack([np.arange(0, 20), np.arange(20, 40)]))
    loss, _ = _inbatch_loss(model(batch), batch, 5, geometry.HyperbolicDecoder())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.emb.weight.grad).all()


def test_gathered_eval_ranks_by_distance():
    # Each held-out pair (2i, 2i+1) sits together on its own ray, far from every other pair, so
    # under the hyperbolic decoder a positive is outranked only when the uniform negatives happen to
    # draw the source itself (distance 0). Replaying the generator predicts every rank exactly.
    n = 40
    rng = np.random.default_rng(0)
    dirs = rng.standard_normal((n // 2, 16))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    w = np.repeat(dirs * 4.0, 2, axis=0)
    w[1::2] *= 1.02
    pos = np.stack([np.arange(0, n, 2), np.arange(1, n, 2)])
    dec = geometry.HyperbolicDecoder()
    model = _model(n, 16, w)
    gen = torch.Generator().manual_seed(0)
    res = eval_mrr_gathered(model, pos, n, 20, (1, 10), gen, dec)
    neg = torch.randint(0, n, (n // 2, 20), generator=torch.Generator().manual_seed(0)).numpy()
    ranks = 1 + (neg == pos[0][:, None]).sum(axis=1)
    assert res["mrr"] == pytest.approx(float(np.mean(1.0 / ranks)), rel=1e-6)
    assert res["hits@1"] == pytest.approx(float(np.mean(ranks == 1)), rel=1e-6)
    # The dense eval agrees with the gathered one (same generator state -> same negatives).
    dense = eval_mrr(
        model.emb.weight.detach(), pos, 20, (1, 10), torch.Generator().manual_seed(0), dec
    )
    assert dense == res


def test_export_holds_clipped_tangent_vectors():
    w = np.zeros((5, 3), dtype=np.float32)
    w[0] = [30.0, 0.0, 40.0]  # |v| = 50 -> clipped to 20
    w[1] = [0.1, 0.2, 0.2]
    model = _model(5, 3, w)
    out = export_table_chunked(
        model, 5, 3, keep=True, chunk=2, decoder=geometry.HyperbolicDecoder()
    )
    np.testing.assert_allclose(out[0], [12.0, 0.0, 16.0], rtol=1e-6)
    np.testing.assert_allclose(out[1:], w[1:], rtol=1e-6)
    dot = export_table_chunked(model, 5, 3, keep=True, chunk=2)
    np.testing.assert_array_equal(dot, w)


def test_hyperbolic_training_learns_a_hierarchy():
    # A complete 4-ary tree of depth 4 (341 nodes) is the textbook case for hyperbolic space.
    # A few hundred steps at dim 4 should rank the tree's edges far above chance (MRR ~0.05).
    branching, depth = 4, 4
    nodes = (branching ** (depth + 1) - 1) // (branching - 1)
    children = np.arange(1, nodes)
    edges = np.stack([(children - 1) // branching, children])  # heap layout: (parent, child)
    torch.manual_seed(0)
    dec = geometry.HyperbolicDecoder()
    model = LPModel(nodes, 4, "shallow", 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loader = _SamplerFreeEdgeLoader(edges, 64, True, seed=0)
    for _ in range(60):
        for batch in loader:
            opt.zero_grad()
            loss, _ = _inbatch_loss(model(batch), batch, 10, dec)
            loss.backward()
            opt.step()
    res = eval_mrr_gathered(model, edges, nodes, 50, (1, 10), torch.Generator().manual_seed(0), dec)
    assert res["mrr"] > 0.5
