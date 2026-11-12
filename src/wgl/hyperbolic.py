"""Numpy helpers for hyperbolic (``--decoder hyperbolic``) embedding tables.

A hyperbolic table stores one **tangent vector at the origin** per host (``v`` in ``R^dim``); the
host's point in hyperbolic space (curvature -1) is ``exp_0(v)`` and ``|v|`` is its distance from
the origin. These rows are *not* comparable by cosine or dot product -- rank neighbours with
:func:`distance` (smaller = closer), or convert to a standard model with :func:`to_lorentz` /
:func:`to_poincare`.

:func:`distance` is the numpy twin of ``cupyg.geometry.HyperbolicDecoder.distance`` (the trainer
cannot import ``wgl``); ``tests/test_hyperbolic.py`` keeps the two in agreement. See
``docs/hyperbolic.md``.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _polar(v: np.ndarray, max_norm: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into (radius ``[..., 1]``, optionally clipped; unit direction ``[..., dim]``)."""
    v = np.asarray(v, dtype=np.float64)
    r = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)
    return (r if max_norm is None else np.minimum(r, max_norm)), v / r


def distance(u: np.ndarray, v: np.ndarray, max_norm: float | None = None) -> np.ndarray:
    """Geodesic distance between ``exp_0(u)`` and ``exp_0(v)``, broadcasting like ``u - v``.

    Uses ``cosh d = cosh(a - b) + sinh(a) sinh(b) (1 - cos theta)`` with ``a = |u|``, ``b = |v|``:
    all terms are non-negative, so there is no cancellation for nearby points far from the origin
    (unlike ``x0*y0 - <xs, ys>`` on Lorentz coordinates). Computed in float64.

    Args:
        u: Tangent vectors ``[..., dim]``.
        v: Tangent vectors ``[..., dim]`` (broadcast against ``u``).
        max_norm: Optional radius clip; exported tables are already clipped, so leave ``None``.

    Returns:
        Distances with the broadcast leading shape.
    """
    a, ua = _polar(u, max_norm)
    b, vb = _polar(v, max_norm)
    half_chord = 0.5 * np.sum((ua - vb) ** 2, axis=-1, keepdims=True)  # 1 - cos(theta)
    z = 2.0 * np.sinh(0.5 * (a - b)) ** 2 + np.sinh(a) * np.sinh(b) * half_chord
    z = np.maximum(z[..., 0], 0.0)
    return np.log1p(z + np.sqrt(z * (z + 2.0)))


def to_lorentz(v: np.ndarray) -> np.ndarray:
    """Map tangent vectors ``[..., dim]`` to hyperboloid points ``[..., dim + 1]``.

    ``x = (cosh |v|, sinh |v| * v / |v|)``, so ``-x0*y0 + <xs, ys> = -cosh d``. Useful with a
    maximum-inner-product index: ranking candidates ``y`` by ``<(x0, -xs), y>`` (largest first) is
    ranking by distance. Prefer :func:`distance` for re-scoring: the inner product loses precision
    in fp32 for nearby points far from the origin.
    """
    r, unit = _polar(v, None)
    return np.concatenate([np.cosh(r), np.sinh(r) * unit], axis=-1)


def to_poincare(v: np.ndarray) -> np.ndarray:
    """Map tangent vectors ``[..., dim]`` to Poincare-ball points ``tanh(|v| / 2) * v / |v|``.

    Handy for plotting (``dim == 2``) and for libraries that expect ball coordinates. Points with
    ``|v|`` above ~37 round onto the ball's boundary in float64 (above ~17 in float32), which is
    why the tangent vectors, not ball coordinates, are the canonical stored form.
    """
    r, unit = _polar(v, None)
    return np.tanh(0.5 * r) * unit
