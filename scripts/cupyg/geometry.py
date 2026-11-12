"""Edge decoders: the published dot product and a hyperbolic (Lorentz) alternative.

The published artifact scores a link as ``e_u · e_v`` (:class:`DotDecoder`). ``--decoder
hyperbolic`` (:class:`HyperbolicDecoder`) instead places hosts in hyperbolic space -- the
hyperboloid/Lorentz model at curvature -1 -- and scores a link with the Fermi-Dirac decoder
``(margin - d(u, v)) / temperature`` (Krioukov et al. 2010; Chami et al. 2019, HGCN), so the BCE
loss learns "closer than ``margin`` -> linked". Hyperbolic space grows exponentially with radius,
which suits heavy-tailed, hierarchy-like graphs such as the web graph; ``docs/hyperbolic.md``
reports how it compares with ``dot`` on a Common Crawl sample.

**Tangent parametrisation.** The table still stores plain Euclidean vectors ``v``, the tangent
vectors at the hyperboloid origin, and a host's point is ``exp_0(v)``. Everything around the table
is unchanged: the WholeMemory-sharded table and its *Euclidean* sparse Adam, checkpoints and the
chunked export all see an ordinary ``[N, dim]`` float table. No Riemannian optimizer is needed
because the optimisation variable lives in a flat space; the geometry lives entirely inside the
decoder. ``|v|`` is exactly the host's hyperbolic distance from the origin.

**Numerics.** With ``a = |u|``, ``b = |v|`` and ``theta`` the angle between them,

    cosh d(u, v) = cosh(a - b) + sinh(a) sinh(b) (1 - cos theta).

Every term is non-negative, so unlike the textbook ``x0*y0 - <xs, ys>`` there is no catastrophic
cancellation for nearby points far from the origin. ``z = cosh d - 1`` is formed directly
(``cosh(a - b) - 1 = 2 sinh^2((a - b) / 2)``, ``1 - cos theta = |u_hat - v_hat|^2 / 2``) and
``d = log1p(z + sqrt(z) sqrt(z + 2))``, which keeps full precision at small distances. Radii are
clipped to ``max_norm`` (at most :data:`MAX_NORM_LIMIT`), which keeps ``sinh(a) sinh(b)`` far
inside the fp32 range.

**Initialisation.** WholeMemory's ``random_init`` (Xavier-uniform over a ~13M-row shard) and the
dense ``N(0, 0.1)`` both start every host within ~0.01-1 of the origin -- harmless for a dot
product, where a near-zero vector scores ~0 against everyone, but in hyperbolic space the origin is
close to *everything*, so hosts that rarely get pushed apart as negatives (the in-batch negatives
favour high-degree hosts) end up outranking true neighbours. :meth:`HyperbolicDecoder.init_`
instead starts every host on the sphere of radius ``init_radius`` in a random direction, so an
untrained host is far from everyone.

Curvature is not a separate knob. For curvature ``-c`` the logit equals the curvature -1 logit with
the table scaled by ``sqrt(c)`` and margin/temperature by ``sqrt(c)``, so ``--hyp-margin`` and
``--hyp-temp`` (plus the free scale of the table) already span it.

``wgl.hyperbolic`` holds the numpy twin of :meth:`HyperbolicDecoder.distance` for consumers of
the released vectors (this package cannot import ``wgl``; ``tests/test_hyperbolic.py`` keeps the
two in agreement).
"""

from __future__ import annotations

import torch

# sinh(40)^2 ~ 5.5e33: z stays below the fp32 max (3.4e38) with headroom for the log1p argument.
MAX_NORM_LIMIT = 40.0
_EPS = 1e-12


class DotDecoder:
    """``score(u, v) = u · v`` -- the published LINE-1st / PBG-dot decoder."""

    name = "dot"

    def __call__(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Score row pairs, broadcasting over leading dims: ``[..., D] x [..., D] -> [...]``."""
        return (u * v).sum(dim=-1)

    def score_candidates(self, h: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        """Score each ``h[i]`` against its ``K`` candidates: ``[B, D] x [B, K, D] -> [B, K]``."""
        return torch.bmm(h.unsqueeze(1), cands.transpose(1, 2)).squeeze(1)

    def prepare(self, v: torch.Tensor) -> torch.Tensor:
        """Map stored rows to the vectors the decoder actually scores (identity for dot)."""
        return v

    def describe(self) -> dict:
        """Decoder settings for the export sidecar."""
        return {"decoder": self.name}


class HyperbolicDecoder:
    """Fermi-Dirac decoder ``(margin - d(u, v)) / temperature`` on the Lorentz model (curvature -1).

    Rows are tangent vectors at the origin (see the module docstring); ``d`` is the geodesic
    distance between ``exp_0(u)`` and ``exp_0(v)`` with radii clipped to ``max_norm``.
    """

    name = "hyperbolic"

    def __init__(
        self,
        margin: float = 2.0,
        temperature: float = 1.0,
        max_norm: float = 20.0,
        init_radius: float = 2.0,
    ) -> None:
        """Validate and store the decoder hyperparameters."""
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0 < max_norm <= MAX_NORM_LIMIT:
            raise ValueError(f"max_norm must be in (0, {MAX_NORM_LIMIT}], got {max_norm}")
        if not 0 <= init_radius <= max_norm:
            raise ValueError(f"init_radius must be in [0, max_norm], got {init_radius}")
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.max_norm = float(max_norm)
        self.init_radius = float(init_radius)

    def _polar(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split rows into (clipped radius ``[..., 1]``, unit direction ``[..., D]``)."""
        r = v.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS * _EPS).sqrt()
        return r.clamp_max(self.max_norm), v / r

    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Geodesic distance between ``exp_0(u)`` and ``exp_0(v)``, broadcasting like ``u - v``."""
        a, ua = self._polar(u)
        b, vb = self._polar(v)
        half_chord = (ua - vb).pow(2).sum(dim=-1, keepdim=True) * 0.5  # 1 - cos(theta)
        z = 2.0 * torch.sinh((a - b) * 0.5).pow(2) + torch.sinh(a) * torch.sinh(b) * half_chord
        z = z.squeeze(-1).clamp_min(_EPS)
        return torch.log1p(z + z.sqrt() * (z + 2.0).sqrt())

    def __call__(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Score row pairs, broadcasting over leading dims: ``[..., D] x [..., D] -> [...]``."""
        return (self.margin - self.distance(u, v)) / self.temperature

    def score_candidates(self, h: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        """Score each ``h[i]`` against its ``K`` candidates: ``[B, D] x [B, K, D] -> [B, K]``."""
        return self(h.unsqueeze(1), cands)

    def prepare(self, v: torch.Tensor) -> torch.Tensor:
        """Clip stored rows to ``max_norm``: the tangent vectors the decoder actually scores."""
        r = v.pow(2).sum(dim=-1, keepdim=True).sqrt()
        return v * (self.max_norm / r.clamp_min(self.max_norm))

    @torch.no_grad()
    def init_(self, table: torch.Tensor, chunk: int = 1 << 20) -> torch.Tensor:
        """Re-initialise ``table`` in place: random directions at radius ``init_radius``.

        Works on any ``[N, dim]`` tensor -- the dense ``nn.Embedding`` weight or a rank's local
        WholeMemory shard -- in ``chunk``-row blocks so a 13M-row shard never needs a full-size
        temporary. ``init_radius == 0`` keeps the table's own initialisation.
        """
        if self.init_radius <= 0:
            return table
        for start in range(0, table.shape[0], chunk):
            rows = torch.randn(
                (min(chunk, table.shape[0] - start), table.shape[1]),
                device=table.device,
                dtype=torch.float32,
            )
            rows *= self.init_radius / rows.norm(dim=1, keepdim=True).clamp_min(_EPS)
            table[start : start + rows.shape[0]].copy_(rows)
        return table

    def describe(self) -> dict:
        """Decoder settings for the export sidecar (``wgl release shard-export`` reads these)."""
        return {
            "decoder": self.name,
            "hyp_margin": self.margin,
            "hyp_temp": self.temperature,
            "hyp_max_norm": self.max_norm,
            "hyp_init_radius": self.init_radius,
        }


Decoder = DotDecoder | HyperbolicDecoder

# Default for every function that takes a decoder: the published dot product.
DOT = DotDecoder()


def make_decoder(
    name: str,
    *,
    margin: float = 2.0,
    temperature: float = 1.0,
    max_norm: float = 20.0,
    init_radius: float = 2.0,
) -> Decoder:
    """Build the decoder selected by ``--decoder`` (the hyperparameters apply to ``hyperbolic``)."""
    if name == "dot":
        return DotDecoder()
    if name == "hyperbolic":
        return HyperbolicDecoder(
            margin=margin, temperature=temperature, max_norm=max_norm, init_radius=init_radius
        )
    raise ValueError(f"unknown decoder {name!r} (expected 'dot' or 'hyperbolic')")
