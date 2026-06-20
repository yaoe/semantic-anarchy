"""Pure-numpy tests for the statistical core and the evolution loop.

No torch / diffusers -- these exercise the whole idea minus the SD decode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy import EmbeddingDistribution, evolve_distribution
from semantic_anarchy.aesthetic import RandomScorer, get_scorer


def _make_corpus(n=4000, tokens=8, hidden=12, seed=0):
    """Corpus with a known per-coordinate mean/std the fit should recover."""
    rng = np.random.default_rng(seed)
    shape = (tokens, hidden)
    true_mean = rng.normal(0.0, 1.0, size=shape)
    true_std = rng.uniform(0.3, 1.5, size=shape)
    corpus = true_mean[None] + true_std[None] * rng.standard_normal((n, *shape))
    return corpus.astype(np.float32), true_mean, true_std


def test_fit_recovers_mean_and_std():
    corpus, true_mean, true_std = _make_corpus()
    dist = EmbeddingDistribution.fit(corpus, per_token=True)
    assert dist.feature_shape == corpus.shape[1:]
    assert np.allclose(dist.mean, true_mean, atol=0.05)
    assert np.allclose(dist.std, true_std, atol=0.05)


def test_sample_shape_and_temperature_scaling():
    corpus, _, _ = _make_corpus()
    dist = EmbeddingDistribution.fit(corpus)
    rng = np.random.default_rng(1)
    s = dist.sample(50, temperature=1.0, rng=rng)
    assert s.shape == (50, *dist.feature_shape)

    # Higher temperature -> larger spread *around the distribution mean*.
    # (Total std is dominated by the per-coordinate means, so measure the
    # deviation from dist.mean, which is what temperature actually scales.)
    rng = np.random.default_rng(2)
    low = dist.sample(4000, temperature=0.5, rng=rng)
    rng = np.random.default_rng(2)
    high = dist.sample(4000, temperature=2.0, rng=rng)
    low_dev = (low - dist.mean[None]).std()
    high_dev = (high - dist.mean[None]).std()
    assert high_dev > low_dev * 3.0  # 4x in expectation (0.5 -> 2.0)


def test_truncation_bound_respected():
    corpus, _, _ = _make_corpus()
    dist = EmbeddingDistribution.fit(corpus)
    rng = np.random.default_rng(3)
    trunc = 2.0
    s = dist.sample(2000, truncation=trunc, rng=rng)
    z = (s - dist.mean[None]) / dist.std[None]
    assert np.abs(z).max() <= trunc + 1e-4


def test_save_load_roundtrip(tmp_path):
    corpus, _, _ = _make_corpus()
    dist = EmbeddingDistribution.fit(corpus, per_token=True)
    path = tmp_path / "dist"
    dist.save(path)
    loaded = EmbeddingDistribution.load(path)
    assert loaded.feature_shape == dist.feature_shape
    assert loaded.per_token == dist.per_token
    assert loaded.n_samples == dist.n_samples
    assert np.allclose(loaded.mean, dist.mean)
    assert np.allclose(loaded.std, dist.std)


def test_refit_from_elites_shifts_mean_toward_elites():
    corpus, _, _ = _make_corpus()
    dist = EmbeddingDistribution.fit(corpus)
    rng = np.random.default_rng(4)

    # Elites all offset far in +x; the refit mean should move toward them.
    elites = dist.sample(200, rng=rng) + 5.0
    branch = dist.refit_from_elites(elites)
    # Branch mean is closer to the elite mean than the base mean was.
    elite_mean = elites.mean(axis=0)
    base_gap = np.abs(dist.mean - elite_mean).mean()
    branch_gap = np.abs(branch.mean - elite_mean).mean()
    assert branch_gap < base_gap


def test_project_clips_to_typical_set():
    corpus, _, _ = _make_corpus()
    dist = EmbeddingDistribution.fit(corpus)
    wild = dist.mean + 10.0 * dist.std  # 10 sigma out
    projected = dist.project(wild, truncation=3.0)
    z = (projected - dist.mean) / dist.std
    assert np.abs(z).max() <= 3.0 + 1e-4


def test_evolution_increases_score_on_analytic_objective():
    corpus, _, _ = _make_corpus(hidden=8)
    dist = EmbeddingDistribution.fit(corpus)
    rng = np.random.default_rng(5)
    target = dist.sample(1, temperature=1.5, rng=np.random.default_rng(123))[0]

    def scorer(pop):
        diff = pop.reshape(pop.shape[0], -1) - target.reshape(-1)[None]
        return -np.sqrt((diff ** 2).mean(axis=1))

    evolved, history = evolve_distribution(
        dist, scorer, generations=15, pop_size=128, elite_frac=0.2,
        rng=rng, verbose=False,
    )
    assert history.improved()
    assert history.elite_mean_score[-1] > history.elite_mean_score[0]
    # The evolved mean should land closer to the hidden target.
    base_gap = np.abs(dist.mean - target).mean()
    evolved_gap = np.abs(evolved.mean - target).mean()
    assert evolved_gap < base_gap


def test_random_scorer_and_factory():
    scorer = get_scorer("random", seed=0)
    assert isinstance(scorer, RandomScorer)
    scores = scorer.score([object()] * 5)  # images aren't inspected by RandomScorer
    assert scores.shape == (5,)
    with pytest.raises(ValueError):
        get_scorer("nonsense")


def test_aesthetic_scorer_degrades_gracefully():
    # No weights file / maybe no torch -> must not raise, falls back to random.
    scorer = get_scorer("aesthetic", weights_path="does/not/exist.pth")
    assert scorer.available is False
    scores = scorer.score([object()] * 3)
    assert scores.shape == (3,)


# ----------------------------------------------------------- PCA / samplers ---
def _make_correlated_corpus(n=80, tokens=6, hidden=8, rank=3, seed=0):
    """A low-rank-plus-noise corpus, so PCA has real structure to capture."""
    rng = np.random.default_rng(seed)
    d = tokens * hidden
    basis = rng.standard_normal((rank, d))
    coeffs = rng.standard_normal((n, rank)) * np.array([4.0, 2.0, 1.0])[:rank]
    x = coeffs @ basis + 0.05 * rng.standard_normal((n, d))
    x = x + rng.normal(0, 1, size=d)  # nonzero per-coordinate mean
    return x.reshape(n, tokens, hidden).astype(np.float32)


def test_pca_samples_lie_in_subspace():
    corpus = _make_correlated_corpus()
    dist = EmbeddingDistribution.fit(corpus, n_components=10)
    assert dist.pca_components is not None
    s = dist.sample(2000, sampler="pca", rng=np.random.default_rng(1))
    flat = s.reshape(2000, -1) - dist.mean.reshape(1, -1)
    comps = dist.pca_components.astype(np.float64)
    # Residual variance orthogonal to the PCA subspace must be ~0.
    proj = flat @ comps.T @ comps
    resid = flat - proj
    assert (resid ** 2).mean() / (flat ** 2).mean() < 1e-6


def test_blend_endpoints_match_diagonal_and_pca():
    corpus = _make_correlated_corpus()
    dist = EmbeddingDistribution.fit(corpus, n_components=10)

    # lambda=0 == diagonal stats; lambda=1 == pca stats (same rng draws).
    diag = dist.sample(3000, sampler="diagonal", rng=np.random.default_rng(7))
    b0 = dist.sample(3000, sampler="blend", coherence=0.0, rng=np.random.default_rng(7))
    assert np.allclose(diag, b0, atol=1e-4)

    pca = dist.sample(3000, sampler="pca", rng=np.random.default_rng(9))
    b1 = dist.sample(3000, sampler="blend", coherence=1.0, rng=np.random.default_rng(9))
    assert np.allclose(pca, b1, atol=1e-4)


def test_pca_save_load_roundtrip(tmp_path):
    corpus = _make_correlated_corpus()
    dist = EmbeddingDistribution.fit(corpus, n_components=12)
    path = tmp_path / "dist_pca"
    dist.save(path)
    loaded = EmbeddingDistribution.load(path)
    assert loaded.pca_components is not None
    assert np.allclose(loaded.pca_components, dist.pca_components)
    assert np.allclose(loaded.pca_std, dist.pca_std)
    # PCA sampling reproduces with the same seed across the round-trip.
    a = dist.sample(50, sampler="pca", rng=np.random.default_rng(3))
    b = loaded.sample(50, sampler="pca", rng=np.random.default_rng(3))
    assert np.allclose(a, b, atol=1e-5)
    assert 0.0 <= loaded.pca_variance_fraction() <= 1.0 + 1e-6


def test_pca_sampler_errors_without_pca():
    # A distribution fitted from a single sample has no PCA -> clear error.
    corpus = _make_correlated_corpus(n=1)
    dist = EmbeddingDistribution.fit(corpus)
    assert dist.pca_components is None
    with pytest.raises(ValueError, match="without PCA"):
        dist.sample(2, sampler="pca")


def test_unique_path_avoids_overwrite(tmp_path):
    from semantic_anarchy.io_utils import unique_path

    p = tmp_path / "sheet.png"
    assert unique_path(p) == p          # free name passes through
    p.write_bytes(b"x")
    assert unique_path(p) == tmp_path / "sheet_1.png"
    (tmp_path / "sheet_1.png").write_bytes(b"x")
    assert unique_path(p) == tmp_path / "sheet_2.png"


def test_hybrid_sampler_fuses_corpus():
    corpus = _make_correlated_corpus(n=40)
    dist = EmbeddingDistribution.fit(corpus, max_corpus=20)
    assert dist.corpus_embeddings is not None
    assert dist.corpus_embeddings.shape[0] == 20  # capped from 40

    s = dist.sample(64, sampler="hybrid", rng=np.random.default_rng(2))
    assert s.shape == (64, *dist.feature_shape)
    assert np.isfinite(s).all()
    # SLERP keeps the magnitude scale of real embeddings (not collapsed to 0).
    samp_norm = np.linalg.norm(s.reshape(64, -1), axis=1).mean()
    corp_norm = np.linalg.norm(dist.corpus_embeddings.reshape(20, -1), axis=1).mean()
    assert 0.5 * corp_norm < samp_norm < 1.6 * corp_norm


def test_hybrid_errors_without_corpus():
    import dataclasses

    dist = EmbeddingDistribution.fit(_make_correlated_corpus(n=20))
    stripped = dataclasses.replace(dist, corpus_embeddings=None)
    with pytest.raises(ValueError, match="corpus_embeddings"):
        stripped.sample(2, sampler="hybrid")


def test_corpus_embeddings_save_load_roundtrip(tmp_path):
    corpus = _make_correlated_corpus(n=30)
    dist = EmbeddingDistribution.fit(corpus, max_corpus=16)
    path = tmp_path / "dist_corp"
    dist.save(path)
    loaded = EmbeddingDistribution.load(path)
    assert loaded.corpus_embeddings is not None
    assert np.allclose(loaded.corpus_embeddings, dist.corpus_embeddings)
    # hybrid reproduces across the round-trip with the same seed.
    a = dist.sample(20, sampler="hybrid", rng=np.random.default_rng(5))
    b = loaded.sample(20, sampler="hybrid", rng=np.random.default_rng(5))
    assert np.allclose(a, b, atol=1e-5)


def test_slerp_endpoints():
    from semantic_anarchy.distribution import _slerp

    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    assert np.allclose(_slerp(a, b, 0.0), a, atol=1e-6)
    assert np.allclose(_slerp(a, b, 1.0), b, atol=1e-6)
    # midpoint lies on the unit circle between them (norm ~1).
    assert abs(np.linalg.norm(_slerp(a, b, 0.5)) - 1.0) < 1e-6
