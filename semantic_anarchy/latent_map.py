"""Latent map -- place every explored image in a navigable 3D space.

Three frames, because "near" can mean three different things:

* **look** -- UMAP of the images' CLIP embeddings. Near means *looks alike*.
  On the real gallery this is the only frame with structure: across the gallery
  conditioning neighbours and visual neighbours overlap at chance (1.9% vs 1.5%),
  because samples sit far apart in a very high-dimensional space. Locally the
  conditioning does decide the look -- a small perturbation stays close -- which
  is why sampling a spot means perturbing the image nearest it.
* **conditioning** -- UMAP of the conditioning itself, through a fixed random
  sketch so a 10k-image gallery never needs a (N, 59k) distance matrix. Shows
  where a sample was drawn from, not what it looks like.
* **corpus** -- z-scores along the top corpus PCA axes:
  ``z_k = c_k . (x - mean) / pca_std_k``. Linear and invertible
  (:func:`unproject`), and the only frame whose axes mean something.

``off_manifold`` is the share of an image's deviation from the corpus mean that
lies outside the corpus PCA span -- near 0 for ``pca``-sampled points, near 1 for
``diagonal`` anarchy.

Pure NumPy at module scope; ``umap``/``sklearn`` are imported where they are
used, so the projection math and the tests of it need neither.
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
    that owns the whole layout."""
    std = np.maximum(dist.std, 0.1 * float(np.mean(dist.std))).reshape(1, -1)
    z = (_flat_dev(dist, x) / std).astype(np.float32)
    out = np.zeros((len(z), k), dtype=np.float64)
    for b, start in enumerate(range(0, z.shape[1], SKETCH_BLOCK)):
        cols = z[:, start:start + SKETCH_BLOCK]
        r = np.random.default_rng([seed, b]).standard_normal((cols.shape[1], k), dtype=np.float32)
        out += cols @ r
    return out / np.sqrt(k)


def _unit_scale(y: np.ndarray) -> np.ndarray:
    """Center, and scale to unit std overall (one factor, so shape is kept)."""
    y = y - y.mean(axis=0, keepdims=True)
    return y / max(float(y.std()), 1e-12)


def warm_start(x: np.ndarray, prev: np.ndarray, known: np.ndarray, metric: str,
               k: int = 5) -> np.ndarray:
    """Layout to start UMAP from: known images where they were, each new image
    at the mean of its ``k`` nearest known neighbours in the source space."""
    init = np.array(prev, dtype=np.float64)
    new = np.where(~known)[0]
    if not len(new) or not known.any():
        return init
    src = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12) \
        if metric == "cosine" else x
    ref = np.where(known)[0]
    for i in new:
        d = np.linalg.norm(src[ref] - src[i], axis=1)
        init[i] = prev[ref[np.argsort(d)[:k]]].mean(axis=0)
    return init


def align(y: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rotate/translate/scale ``y`` onto ``target`` over the ``mask``ed rows
    (orthogonal Procrustes, no reflection), so a rebuild doesn't spin the map."""
    if mask.sum() < 4:
        return y
    ya, ta = y[mask], target[mask]
    cy, ct = ya.mean(axis=0), ta.mean(axis=0)
    u, sv, vt = np.linalg.svd((ya - cy).T @ (ta - ct))
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    scale = (sv * [1.0, 1.0, d]).sum() / max(float(((ya - cy) ** 2).sum()), 1e-12)
    return (y - cy) @ r * scale + ct


def umap_3d(x: np.ndarray, metric: str, neighbors: int, min_dist: float,
            prev: Optional[np.ndarray] = None, known: Optional[np.ndarray] = None,
            seed: int = 0) -> np.ndarray:
    """``(n, D)`` -> unit-scale ``(n, 3)``. With ``prev``/``known`` it continues
    from an earlier layout, so adding images doesn't reshuffle the map."""
    import umap

    n = len(x)
    if n < 6:                       # below UMAP's spectral init; nothing to see yet
        return np.zeros((n, 3))
    warm = prev is not None and known is not None and known.sum() >= 6
    init = warm_start(x, prev, known, metric) if warm else "spectral"
    y = umap.UMAP(
        n_components=3, n_neighbors=min(neighbors, n - 1), min_dist=min_dist,
        metric=metric, random_state=seed, init=init, n_jobs=1,
        learning_rate=0.25 if warm else 1.0,
    ).fit_transform(x)
    y = _unit_scale(np.asarray(y, dtype=np.float64))
    return align(y, prev, known) if warm else y


def neighbours_kept(source: np.ndarray, y: np.ndarray, metric: str, k: int = 15,
                    limit: int = 2000, seed: int = 0) -> Optional[float]:
    """Trustworthiness of ``y`` against ``source``: 1 = whoever is near on the
    map is near in the source, 0.5 = chance. ``None`` when there are too few."""
    from sklearn.manifold import trustworthiness

    n = len(y)
    if n < 2 * k + 2:
        return None
    pick = np.random.default_rng(seed).choice(n, limit, replace=False) if n > limit else np.arange(n)
    return float(trustworthiness(source[pick], y[pick], n_neighbors=k, metric=metric))


def find_regions(y: np.ndarray, prev: Optional[np.ndarray] = None) -> np.ndarray:
    """Cluster labels ``(n,)`` of the layout, ``-1`` for unclaimed images. Numbers
    follow ``prev`` (each cluster keeps the id most of its members had), so a
    region doesn't change colour every time an image lands."""
    from sklearn.cluster import HDBSCAN

    n = len(y)
    if n < 16:
        return np.full(n, -1)
    # Not "leaf": it carves plain Gaussian blobs into fragments that aren't there.
    # But the default can flip to one cluster holding nearly everything when the
    # layout shifts slightly (20 regions -> 2 on the real gallery), which says
    # nothing about the images. Only then, cap the cluster size.
    args = dict(min_cluster_size=max(8, n // 80), min_samples=4, copy=True)
    raw = HDBSCAN(**args).fit_predict(y)
    if (raw >= 0).any() and np.bincount(raw[raw >= 0]).max() > n / 3:
        cap = max(2 * args["min_cluster_size"], n // 6)
        raw = HDBSCAN(max_cluster_size=cap, **args).fit_predict(y)
    ids = [c for c in np.unique(raw) if c >= 0]
    ids.sort(key=lambda c: -(raw == c).sum())
    out = np.full(n, -1)
    used: set[int] = set()
    fresh = 0 if prev is None else int(prev.max(initial=-1)) + 1
    for rank, c in enumerate(ids):
        mask = raw == c
        new_id = None
        if prev is not None:
            old, count = np.unique(prev[mask], return_counts=True)
            for o in old[np.argsort(-count)]:
                if o >= 0 and int(o) not in used:
                    new_id = int(o)
                    break
        if new_id is None:
            new_id = rank if prev is None else fresh
            fresh += prev is not None
        used.add(new_id)
        out[mask] = new_id
    return out


def region_faces(y: np.ndarray, labels: np.ndarray, k: int = 4) -> dict[int, list[int]]:
    """Per region, the ``k`` member rows nearest its centroid: its typical images."""
    faces = {}
    for c in np.unique(labels):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        d = np.linalg.norm(y[idx] - y[idx].mean(axis=0), axis=1)
        faces[int(c)] = idx[np.argsort(d)[:k]].tolist()
    return faces


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
