"""Latent map -- place every explored image in a low-dimensional, navigable space.

Every generated image carries its exact conditioning in an ``.npz`` sidecar, and
the fitted distribution already knows the corpus' principal axes. That is enough
to draw the explored territory without touching a model:

* **corpus frame** -- z-scores along the corpus PCA axes:
  ``z_k = c_k . (x - mean) / pca_std_k``. Stable across runs (the axes are the
  corpus'), comparable between batches, and *invertible*: a point picked in
  this frame maps straight back to a conditioning tensor (:func:`unproject`),
  which is exactly the ``pca`` sampler with its coefficients pinned.
* **explored frame** -- PCA of the explored images themselves (after a fixed
  random sketch, so a 10k-image gallery never needs a (N, 59k) SVD). It spreads
  the explorations out along the directions *you* actually travelled.

``off_manifold`` is the share of an image's deviation from the corpus mean that
lies outside the corpus PCA span -- near 0 for ``pca``-sampled points, near 1 for
``diagonal`` anarchy. The 3D view can only show the first few axes, so this says
how much of each point the picture is leaving out.

Pure NumPy: imported by ``scripts/latent_map.py`` and the tests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .distribution import EmbeddingDistribution

SKETCH_DIM = 256


def _flat_dev(dist: EmbeddingDistribution, x: np.ndarray) -> np.ndarray:
    """``(n, *feature_shape)`` -> centered ``(n, D)`` float32 deviations."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == len(dist.feature_shape):
        x = x[None]
    return x.reshape(len(x), -1) - dist.mean.reshape(1, -1)


def project(dist: EmbeddingDistribution, x: np.ndarray, dims: int) -> np.ndarray:
    """Corpus-frame coordinates ``(n, dims)``: z-scores along the top PCA axes."""
    comps, pstd = dist.pca_components[:dims], dist.pca_std[:dims]
    return (_flat_dev(dist, x) @ comps.T) / np.maximum(pstd, 1e-12)[None]


def unproject(dist: EmbeddingDistribution, z: np.ndarray,
              axes: Optional[list[int]] = None) -> np.ndarray:
    """Inverse of :func:`project` within the PCA span: corpus-frame z-scores
    (one per entry of ``axes``, default the leading axes) -> a conditioning
    tensor of ``feature_shape``. Every axis not named stays at the corpus mean."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    axes = list(range(len(z))) if axes is None else list(axes)
    comps = dist.pca_components[axes].astype(np.float64)
    pstd = dist.pca_std[axes].astype(np.float64)
    flat = dist.mean.reshape(-1).astype(np.float64) + (z * pstd) @ comps
    return flat.reshape(dist.feature_shape).astype(np.float32)


def off_manifold(dist: EmbeddingDistribution, x: np.ndarray) -> np.ndarray:
    """Fraction of each deviation's energy outside the full corpus PCA span."""
    dev = _flat_dev(dist, x)
    inside = dev @ dist.pca_components.T              # (n, k), components orthonormal
    total = np.einsum("ij,ij->i", dev, dev, dtype=np.float64)
    kept = np.einsum("ij,ij->i", inside, inside, dtype=np.float64)
    return np.clip(1.0 - kept / np.maximum(total, 1e-12), 0.0, 1.0)


def explained(dist: EmbeddingDistribution, dims: int) -> np.ndarray:
    """Share of the corpus' total variance carried by each of the top axes."""
    var = dist.pca_std[:dims].astype(np.float64) ** 2
    return var / max(dist.total_var, 1e-12)


SKETCH_BLOCK = 65536   # feature columns per generated slab of the sketch matrix


def sketch(dist: EmbeddingDistribution, x: np.ndarray, k: int = SKETCH_DIM,
           seed: int = 0) -> np.ndarray:
    """z-scored (per-coordinate, like the distance gauge) then sketched: ``(n, k)``.

    A fixed Gaussian random projection (Johnson-Lindenstrauss) keeps pairwise
    distances to within a few percent. The ``(D, k)`` matrix is never held
    whole: it is generated in seeded slabs of :data:`SKETCH_BLOCK` columns, so
    every call sees the same matrix while peak memory stays ~``SKETCH_BLOCK*k``
    floats -- for flow-model conditioning D is ~1M, i.e. a 1 GB matrix.

    Same sigma floor as :meth:`EmbeddingDistribution.distance`: coordinates the
    corpus never varies (CLIP's start token sits at the fit's 1e-4 floor) would
    otherwise turn a tiny nudge -- e.g. a hybrid SLERP -- into a 1e5 z-score
    that owns the whole explored frame."""
    std = np.maximum(dist.std, 0.1 * float(np.mean(dist.std))).reshape(1, -1)
    z = (_flat_dev(dist, x) / std).astype(np.float32)
    out = np.zeros((len(z), k), dtype=np.float64)
    for b, start in enumerate(range(0, z.shape[1], SKETCH_BLOCK)):
        cols = z[:, start:start + SKETCH_BLOCK]
        r = np.random.default_rng([seed, b]).standard_normal((cols.shape[1], k), dtype=np.float32)
        out += cols @ r
    return out / np.sqrt(k)


def explored_frame(sketches: np.ndarray, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA of the explored set (in sketch space). Returns ``(coords (n, dims),
    variance share per axis)``; coordinates are scaled to unit std per axis so
    the frame fills the view the same way the corpus frame does."""
    s = np.asarray(sketches, dtype=np.float64)
    n = len(s)
    if n < 2:
        return np.zeros((n, dims)), np.zeros(dims)
    c = s - s.mean(axis=0, keepdims=True)
    _, sv, vt = np.linalg.svd(c, full_matrices=False)
    m = min(dims, len(sv))
    coords = np.zeros((n, dims))
    sd = sv[:m] / np.sqrt(n - 1)
    coords[:, :m] = (c @ vt[:m].T) / np.maximum(sd, 1e-12)[None]
    # Deterministic orientation: an SVD axis' sign is arbitrary, so point each
    # one so the newest images (last rows) drift positive -- and stay put
    # between rebuilds instead of flipping.
    flip = np.sign(coords[-max(1, n // 10):, :].mean(axis=0))
    coords *= np.where(flip == 0, 1.0, flip)[None]
    share = np.zeros(dims)
    share[:m] = sv[:m] ** 2 / max(float((sv ** 2).sum()), 1e-12)
    return coords, share


def corpus_indices(n_samples: int, retained: int) -> np.ndarray:
    """Which corpus rows :meth:`EmbeddingDistribution.fit` kept as
    ``corpus_embeddings`` -- replays its seeded draw, so retained embeddings can
    be matched back to their prompt text."""
    if n_samples <= retained:
        return np.arange(n_samples)
    return np.random.default_rng(0).choice(n_samples, size=retained, replace=False)


def axis_extremes(z: np.ndarray, labels: list[str], k: int = 3) -> list[dict]:
    """Per axis: the ``k`` corpus prompts sitting furthest along + and -."""
    out = []
    for a in range(z.shape[1]):
        order = np.argsort(z[:, a])
        out.append({"pos": [labels[i] for i in order[::-1][:k]],
                    "neg": [labels[i] for i in order[:k]]})
    return out
