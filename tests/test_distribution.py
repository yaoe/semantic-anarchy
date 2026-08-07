"""Pure-numpy tests for the statistical core and the evolution loop.

No torch / diffusers -- these exercise the whole idea minus the SD decode.
"""

from __future__ import annotations

import json
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


def test_pca_matches_a_direct_svd():
    """The Gram trick is an optimisation, not a different answer.

    With fewer samples than coordinates (every real corpus) fit() eigendecomposes
    the (N,N) Gram matrix instead of SVD-ing the (N,D) data. That must give the
    same spectrum and the same subspace -- component SIGNS are arbitrary in both,
    so axes are compared by |cos|.
    """
    corpus = _make_correlated_corpus(n=120, tokens=10, hidden=60)
    n, d = corpus.shape[0], corpus[0].size
    assert n < d                                     # the Gram branch
    dist = EmbeddingDistribution.fit(corpus, n_components=20)

    flat = corpus.reshape(n, -1).astype(np.float64) - dist.mean.reshape(1, -1)
    _, s, vt = np.linalg.svd(flat, full_matrices=False)

    assert np.allclose(dist.pca_std, s[:20] / np.sqrt(n - 1), rtol=1e-4)
    cos = np.abs((dist.pca_components.astype(np.float64) * vt[:20]).sum(axis=1))
    assert cos.min() > 1 - 1e-6
    comps = dist.pca_components.astype(np.float64)
    assert np.allclose(comps @ comps.T, np.eye(20), atol=1e-5)


def test_pca_svd_branch_still_works_with_more_samples_than_coords():
    """The other side of the branch: N > D falls back to the direct SVD."""
    corpus = _make_correlated_corpus(n=200, tokens=4, hidden=6)
    assert corpus.shape[0] > corpus[0].size          # the SVD branch
    dist = EmbeddingDistribution.fit(corpus, n_components=5)
    assert dist.pca_components.shape == (5, 24)
    s = dist.sample(500, sampler="pca", rng=np.random.default_rng(0))
    flat = s.reshape(500, -1) - dist.mean.reshape(1, -1)
    comps = dist.pca_components.astype(np.float64)
    resid = flat - flat @ comps.T @ comps
    assert (resid ** 2).mean() / (flat ** 2).mean() < 1e-6


def test_n_components_caps_the_retained_rank():
    """A cap keeps the LEADING axes -- the ones the spectrum actually needs."""
    corpus = _make_correlated_corpus(n=60, tokens=8, hidden=8)
    capped = EmbeddingDistribution.fit(corpus, n_components=5)
    full = EmbeddingDistribution.fit(corpus)
    assert capped.pca_components.shape[0] == 5
    assert full.pca_components.shape[0] == 59              # N-1
    # the retained axes are the top of the same spectrum
    assert np.allclose(capped.pca_std, full.pca_std[:5], rtol=1e-4)
    # asking for more than the rank is clamped, not an error
    assert EmbeddingDistribution.fit(
        corpus, n_components=10_000).pca_components.shape[0] == 59


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


# --------------------------------------------------------------------------- #
# The corpus-autopsy corrections (docs/TODO/01_exploration_plan.md phase 1).
# Every one of these is opt-in: the "unchanged by default" tests below are the
# load-bearing ones, because a broken sampler is a texture the project keeps.
# --------------------------------------------------------------------------- #
def _make_length_corpus(n=600, tokens=10, hidden=6, seed=0):
    """A corpus whose token rows switch to a DIFFERENT lobe past each prompt's EOS.

    Mirrors what CLIP actually does: content rows come from one population,
    padding rows from another, and the boundary moves per prompt. The pooled
    per-position Gaussian therefore peaks between the two lobes -- the exact
    failure the length-conditional fit exists to remove.
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(2, tokens, size=n)
    mu_content = rng.normal(0.0, 1.0, size=(tokens, hidden))
    mu_pad = mu_content + 5.0
    x = np.empty((n, tokens, hidden), dtype=np.float32)
    for i, ln in enumerate(lengths):
        for t in range(tokens):
            base = mu_content[t] if t < ln else mu_pad[t]
            x[i, t] = base + 0.2 * rng.standard_normal(hidden)
    return x, lengths.astype(np.int32), mu_content, mu_pad


def test_length_fit_recovers_both_lobes():
    corpus, lengths, mu_c, mu_p = _make_length_corpus()
    dist = EmbeddingDistribution.fit(corpus, lengths=lengths, n_components=20)
    assert dist.has_length_stats
    # Middle positions see both populations, so both means are recoverable.
    assert np.allclose(dist.mean_content[5], mu_c[5], atol=0.1)
    assert np.allclose(dist.mean_pad[5], mu_p[5], atol=0.1)
    # ...and the pooled fit sits in the gap between them, which is the bug.
    assert np.abs(dist.mean[5] - mu_c[5]).mean() > 1.0


def test_length_conditional_sampling_lands_on_the_right_lobe():
    corpus, lengths, mu_c, mu_p = _make_length_corpus()
    dist = EmbeddingDistribution.fit(corpus, lengths=lengths, n_components=20)
    lens = dist.draw_lengths(400, np.random.default_rng(0), "fixed", 6)
    s = dist.sample(400, rng=np.random.default_rng(1), lengths=lens)
    assert np.allclose(s[:, 3].mean(axis=0), mu_c[3], atol=0.15)   # before EOS
    assert np.allclose(s[:, 8].mean(axis=0), mu_p[8], atol=0.15)   # after EOS
    # Without the length knob the same position lands in neither lobe.
    plain = dist.sample(400, rng=np.random.default_rng(1))
    assert np.abs(plain[:, 3].mean(axis=0) - mu_c[3]).mean() > 1.0


def test_length_mode_corpus_bootstraps_the_histogram():
    corpus, lengths, _, _ = _make_length_corpus()
    dist = EmbeddingDistribution.fit(corpus, lengths=lengths, n_components=20)
    drawn = dist.draw_lengths(5000, np.random.default_rng(2), "corpus")
    assert set(np.unique(drawn)).issubset(set(np.unique(lengths)))
    assert abs(drawn.mean() - lengths.mean()) < 0.2


def test_length_ignored_when_the_fit_predates_it():
    corpus, lengths, _, _ = _make_length_corpus()
    dist = EmbeddingDistribution.fit(corpus, n_components=20)   # no lengths
    assert not dist.has_length_stats
    a = dist.sample(8, rng=np.random.default_rng(3), lengths=np.full(8, 5))
    b = dist.sample(8, rng=np.random.default_rng(3))
    assert np.allclose(a, b)


def test_rho_sets_row_coherence_without_moving_the_marginals():
    corpus = _make_correlated_corpus(n=200, tokens=6, hidden=8)
    dist = EmbeddingDistribution.fit(corpus, n_components=10)

    def row_corr(s):
        z = (s - dist.mean) / dist.std
        return float(np.mean([np.corrcoef(z[:, 0, h], z[:, 4, h])[0, 1]
                              for h in range(z.shape[2])]))

    for rho in (0.0, 0.4, 0.8, 1.0):
        s = dist.sample(4000, sampler="diagonal", rho=rho,
                        rng=np.random.default_rng(11))
        assert abs(row_corr(s) - rho) < 0.06
        # Marginals are untouched: sqrt(rho)^2 + sqrt(1-rho)^2 == 1 by design.
        z = (s - dist.mean) / dist.std
        assert abs(z.std() - 1.0) < 0.05


def test_rho_shares_its_deviation_only_inside_the_content_span():
    corpus, lengths, _, _ = _make_length_corpus(tokens=10)
    dist = EmbeddingDistribution.fit(corpus, lengths=lengths, n_components=20)
    lens = dist.draw_lengths(3000, np.random.default_rng(4), "fixed", 6)
    s = dist.sample(3000, rho=1.0, lengths=lens, rng=np.random.default_rng(5))
    z = (s - dist.mean_content) / dist.std_content
    inside = np.mean([np.corrcoef(z[:, 1, h], z[:, 4, h])[0, 1]
                      for h in range(z.shape[2])])
    across = np.mean([np.corrcoef(z[:, 1, h], z[:, 8, h])[0, 1]
                      for h in range(z.shape[2])])
    assert inside > 0.9        # both rows are content: fully shared
    assert abs(across) < 0.1   # row 8 is padding: independent


def test_empirical_head_reproduces_a_bimodal_leading_axis():
    # A corpus whose PC1 coefficient is two lobes with a hole at zero -- exactly
    # what the report found for the real PC1 (prompt length in disguise).
    rng = np.random.default_rng(0)
    d, n = 48, 800
    basis = np.linalg.qr(rng.standard_normal((d, 3)))[0].T
    lobe = np.where(rng.random(n) < 0.5, -3.0, 3.0) + 0.3 * rng.standard_normal(n)
    coeffs = np.stack([lobe, rng.standard_normal(n), rng.standard_normal(n)], 1)
    corpus = (coeffs @ basis).reshape(n, 6, 8).astype(np.float32)
    dist = EmbeddingDistribution.fit(corpus, n_components=3)
    assert dist.pca_head is not None

    def pc1(s):
        flat = s.reshape(len(s), -1) - dist.mean.reshape(1, -1)
        return (flat @ dist.pca_components[0]) / dist.pca_std[0]

    gauss = pc1(dist.sample(4000, sampler="pca", rng=np.random.default_rng(6)))
    emp = pc1(dist.sample(4000, sampler="pca", empirical_head=1,
                          rng=np.random.default_rng(6)))
    # The Gaussian puts its densest mass exactly where the corpus has none.
    assert np.mean(np.abs(gauss) < 0.5) > 0.3
    assert np.mean(np.abs(emp) < 0.5) < 0.05


def test_radius_band_is_wider_than_the_samplers_spike():
    corpus = _make_correlated_corpus(n=200)
    dist = EmbeddingDistribution.fit(corpus, n_components=10)
    assert dist.corpus_distance is not None
    radii = dist.sample_radii(500, np.random.default_rng(7))
    assert radii.std() > 0
    # retarget honours a per-sample band, not just a scalar shell.
    s = dist.sample(5, rng=np.random.default_rng(8))
    pinned = dist.retarget(s, radii[:5])
    got = [dist.distance(x) for x in pinned]
    assert np.allclose(got, radii[:5], rtol=1e-3)
    # ...and still honours a scalar.
    assert np.allclose([dist.distance(x) for x in dist.retarget(s, 1.7)], 1.7,
                       rtol=1e-3)


def test_split_sampler_decomposes_the_diagonal_deviation():
    corpus = _make_correlated_corpus(n=200)
    dist = EmbeddingDistribution.fit(corpus, n_components=10)
    comps = dist.pca_components.astype(np.float64)

    def parts(s):
        flat = s.reshape(len(s), -1) - dist.mean.reshape(1, -1)
        on = flat @ comps.T @ comps
        return (on ** 2).mean(), ((flat - on) ** 2).mean()

    # temp_on=temp_off=1 is exactly the diagonal it was built from.
    diag = dist.sample(64, sampler="diagonal", rng=np.random.default_rng(9))
    both = dist.sample(64, sampler="split", rng=np.random.default_rng(9))
    assert np.allclose(diag, both, atol=1e-4)

    on_only = dist.sample(400, sampler="split", temp_on=1.0, temp_off=0.0,
                          rng=np.random.default_rng(10))
    off_only = dist.sample(400, sampler="split", temp_on=0.0, temp_off=1.0,
                           rng=np.random.default_rng(10))
    assert parts(on_only)[1] < 1e-8      # nothing off-manifold
    assert parts(off_only)[0] < 1e-8     # nothing on-manifold
    # Doubling one half doubles that half's energy and leaves the other alone.
    hot_on = dist.sample(400, sampler="split", temp_on=2.0, temp_off=1.0,
                         rng=np.random.default_rng(10))
    base_on, base_off = parts(dist.sample(400, sampler="split",
                                          rng=np.random.default_rng(10)))
    assert abs(parts(hot_on)[0] / base_on - 4.0) < 0.05
    assert abs(parts(hot_on)[1] / base_off - 1.0) < 0.05


def test_noise_floor_is_below_the_fitted_rank_and_caps_equalize():
    corpus = _make_correlated_corpus(n=120, tokens=6, hidden=8, rank=3)
    dist = EmbeddingDistribution.fit(corpus)     # full rank
    floor = dist.noise_floor_axes()
    assert floor is not None and 0 < floor < dist.pca_std.shape[0]
    # Equalising with no explicit `components` now stops at the floor instead of
    # spending the whole budget amplifying stored noise.
    s = dist.sample(200, sampler="pca", equalize=True, rng=np.random.default_rng(12))
    flat = s.reshape(200, -1) - dist.mean.reshape(1, -1)
    comps = dist.pca_components.astype(np.float64)
    tail = flat @ comps[floor:].T
    assert (tail ** 2).mean() < 1e-8
    # An explicit `components` still reaches as far as it is told to.
    wide = dist.sample(200, sampler="pca", equalize=True,
                       components=dist.pca_std.shape[0],
                       rng=np.random.default_rng(12))
    tail_wide = (wide.reshape(200, -1) - dist.mean.reshape(1, -1)) @ comps[floor:].T
    assert (tail_wide ** 2).mean() > 1e-6


def test_defaults_reproduce_the_original_samplers_bit_for_bit():
    """The whole point: every correction is opt-in, nothing shifts underneath."""
    corpus = _make_correlated_corpus(n=120)
    dist = EmbeddingDistribution.fit(corpus, n_components=10)
    for sampler in ("diagonal", "pca", "blend", "hybrid"):
        a = dist.sample(16, sampler=sampler, rng=np.random.default_rng(3))
        b = dist.sample(16, sampler=sampler, rng=np.random.default_rng(3),
                        rho=0.0, lengths=None, empirical_head=0,
                        temp_on=1.0, temp_off=1.0)
        assert np.array_equal(a, b), sampler


def test_new_arrays_survive_the_save_load_roundtrip(tmp_path):
    corpus, lengths, _, _ = _make_length_corpus(n=200)
    dist = EmbeddingDistribution.fit(corpus, lengths=lengths, n_components=12)
    path = tmp_path / "dist_len"
    dist.save(path)
    loaded = EmbeddingDistribution.load(path)
    assert loaded.has_length_stats
    assert loaded.pca_head is not None and loaded.corpus_distance is not None
    assert np.array_equal(loaded.lengths, dist.lengths)
    lens = np.full(12, 5)
    a = dist.sample(12, lengths=lens, rho=0.5, rng=np.random.default_rng(2))
    b = loaded.sample(12, lengths=lens, rho=0.5, rng=np.random.default_rng(2))
    assert np.allclose(a, b, atol=1e-5)
    meta = json.loads((path.with_suffix(".meta.json")).read_text())
    assert meta["has_length_stats"] and meta["has_radius_band"]


def test_load_of_a_pre_correction_npz_still_works(tmp_path):
    """A distribution mined before any of this existed must load and sample."""
    corpus = _make_correlated_corpus(n=40)
    dist = EmbeddingDistribution.fit(corpus, n_components=8)
    path = tmp_path / "old"
    dist.save(path)
    # Strip the new arrays from the npz, the way an older mine would have.
    with np.load(str(path) + ".npz") as z:
        keep = {k: z[k] for k in z.files
                if k in ("mean", "std", "pca_components", "pca_std",
                         "corpus_embeddings")}
    np.savez_compressed(str(path) + ".npz", **keep)
    old = EmbeddingDistribution.load(path)
    assert not old.has_length_stats and old.pca_head is None
    assert old.sample(4, sampler="pca", empirical_head=4).shape[0] == 4
    with pytest.raises(ValueError, match="radius band"):
        old.sample_radii(4)


def test_blend_endpoints_stay_exact_under_length_conditioning():
    """blend(0) is a diagonal draw and must keep being treated as one."""
    corpus, lengths, _, _ = _make_length_corpus(n=200)
    dist = EmbeddingDistribution.fit(corpus, lengths=lengths, n_components=12)
    lens = np.full(64, 5)
    diag = dist.sample(64, sampler="diagonal", lengths=lens,
                       rng=np.random.default_rng(7))
    b0 = dist.sample(64, sampler="blend", coherence=0.0, lengths=lens,
                     rng=np.random.default_rng(7))
    assert np.allclose(diag, b0, atol=1e-4)
    pca = dist.sample(64, sampler="pca", lengths=lens, rng=np.random.default_rng(8))
    b1 = dist.sample(64, sampler="blend", coherence=1.0, lengths=lens,
                     rng=np.random.default_rng(8))
    assert np.allclose(pca, b1, atol=1e-4)


def test_length_conditioning_does_not_double_count_the_length_shift():
    """A subspace deviation already carries a length shift; it gets projected out."""
    corpus, lengths, _, _ = _make_length_corpus(n=400)
    dist = EmbeddingDistribution.fit(corpus, lengths=lengths, n_components=20)
    lens = np.full(300, 5)
    s = dist.sample(300, sampler="pca", lengths=lens, rng=np.random.default_rng(9))
    mu, _, _ = dist._length_fields(lens, 300)
    delta = (mu - dist.mean[None]).reshape(300, -1)
    dev = s.reshape(300, -1) - mu.reshape(300, -1)
    along = np.einsum("ij,ij->i", dev, delta) / np.linalg.norm(delta, axis=1)
    assert np.abs(along).max() < 1e-3
