"""The chunked fit for flow-model conditioning (~2M coordinates per sample).

A 4k-prompt krea2 corpus is 15GiB as fp16, and the fit used to hold two more
float32 copies of it, which drove a 31GiB box into swap for hours. These pin that
the chunked path agrees with the textbook answer, however it is chunked, and that
it really stays out of the full-size copies.
"""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy import distribution as dist_mod
from semantic_anarchy.distribution import EmbeddingDistribution, _floor_std
from semantic_anarchy.pipeline import _encode_stacked

SHAPE = (4, 3, 18_000)          # 216,000 coords: just over WIDE_FEATURES


def _corpus(n=24, dtype=np.float16, seed=0):
    """Low-rank signal + noise, so the top axes are well separated and comparable."""
    rng = np.random.default_rng(seed)
    d = int(np.prod(SHAPE))
    basis = rng.standard_normal((6, d)) * 0.5
    coef = rng.standard_normal((n, 6)) * np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    x = coef @ basis + 0.2 * rng.standard_normal((n, d)) + 1.0
    return x.reshape(n, *SHAPE).astype(dtype)


def _chunk_rows(monkeypatch, rows):
    monkeypatch.setattr(dist_mod, "WIDE_CHUNK_BYTES", rows * 4 * int(np.prod(SHAPE)))


@pytest.mark.parametrize("rows", [1, 5, 7, 100])
def test_wide_fit_matches_the_direct_computation(monkeypatch, rows):
    _chunk_rows(monkeypatch, rows)
    corpus = _corpus()
    d = EmbeddingDistribution.fit(corpus, n_components=4)

    x = corpus.astype(np.float64).reshape(len(corpus), -1)
    mean = x.mean(axis=0)
    centered = x - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)

    np.testing.assert_allclose(d.mean.ravel(), mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(d.std.ravel(), x.std(axis=0), rtol=1e-4, atol=1e-4)
    assert d.total_var == pytest.approx((centered ** 2).sum() / (len(x) - 1), rel=1e-5)
    np.testing.assert_allclose(d.pca_std, s[:4] / np.sqrt(len(x) - 1), rtol=1e-4)
    for got, want in zip(d.pca_components, vt[:4]):
        assert abs(float(got.astype(np.float64) @ want)) == pytest.approx(1.0, abs=1e-3)
    want_radius = np.sqrt(np.mean((centered / _floor_std(d.std).ravel()) ** 2, axis=1))
    np.testing.assert_allclose(d.corpus_distance, want_radius, rtol=1e-3)


def test_wide_fit_does_not_depend_on_the_chunking(monkeypatch):
    corpus = _corpus()
    _chunk_rows(monkeypatch, 5)
    a = EmbeddingDistribution.fit(corpus, n_components=4)
    _chunk_rows(monkeypatch, 100)
    b = EmbeddingDistribution.fit(corpus, n_components=4)
    np.testing.assert_allclose(a.pca_std, b.pca_std, rtol=1e-5)
    np.testing.assert_allclose(a.pca_head, b.pca_head, atol=1e-3)
    np.testing.assert_allclose(a.std, b.std, rtol=1e-6)


def test_wide_fit_samples_and_round_trips(tmp_path):
    d = EmbeddingDistribution.fit(_corpus(), n_components=4)
    assert d.mean.dtype == np.float32 and d.pca_components.dtype == np.float32
    assert d.corpus_embeddings.shape == (24, *SHAPE)
    assert d.sample(2, sampler="pca").shape == (2, *SHAPE)
    d.save(tmp_path / "wide")
    again = EmbeddingDistribution.load(tmp_path / "wide")
    np.testing.assert_array_equal(again.pca_std, d.pca_std)


def test_wide_fit_pools_over_tokens_on_a_3d_feature_shape():
    corpus = _corpus()
    d = EmbeddingDistribution.fit(corpus, per_token=False, n_components=2)
    assert d.mean.shape == SHAPE
    np.testing.assert_array_equal(d.mean[0], d.mean[3])
    x = corpus.astype(np.float64)
    np.testing.assert_allclose(d.mean[0], x.mean(axis=(0, 1)), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(d.std[0], x.std(axis=(0, 1)), rtol=1e-4, atol=1e-4)


def test_wide_fit_memory_does_not_grow_with_the_corpus(monkeypatch):
    """Twice the rows must not cost anything like twice the working memory."""
    _chunk_rows(monkeypatch, 2)

    def peak(n):
        corpus = _corpus(n=n)
        tracemalloc.start()
        EmbeddingDistribution.fit(corpus, n_components=2, max_corpus=2)
        out = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return out

    extra_rows_as_float32 = 40 * 4 * int(np.prod(SHAPE))
    assert peak(80) - peak(40) < extra_rows_as_float32 / 10


def test_wide_fit_reports_progress():
    seen = []
    EmbeddingDistribution.fit(_corpus(), n_components=2, log=seen.append)
    assert any("gram" in m for m in seen) and any("PCA axes" in m for m in seen)


def test_marginal_plot_only_casts_the_plotted_coordinates(tmp_path):
    pytest.importorskip("matplotlib")
    from semantic_anarchy import plot_marginals
    corpus = _corpus(n=40)
    tracemalloc.start()
    out = plot_marginals(corpus, n=2, out_path=tmp_path / "m.png")
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert out.exists()
    assert peak < corpus.size * 8 / 2


def test_encode_stacked_keeps_order_and_dtype():
    seen = []
    def one(chunk):
        return np.array([[int(p)] * 2 for p in chunk], dtype=np.float16)
    out = _encode_stacked([str(i) for i in range(7)], 3, one, seen.append)
    assert out.dtype == np.float16 and out.shape == (7, 2)
    np.testing.assert_array_equal(out[:, 0], np.arange(7))
    assert seen == [3, 6, 7]
