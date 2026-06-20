"""Torch-free tests for the backend abstraction (no model load required).

We exercise the numpy-only verbs -- fit/sample/save/load -- on synthetic named
tensors for both backends, plus the sampler dispatch. The model-touching verbs
(encode/generate) need torch/diffusers and are not covered here.
"""

import numpy as np
import pytest

from semantic_anarchy import dist_backend, make_backend, BACKEND_DEFAULTS
from semantic_anarchy.distribution import EmbeddingDistribution


def _sdxl_named(n=30):
    return {
        "prompt_embeds": np.random.randn(n, 77, 2048).astype("float32"),
        "pooled": np.random.randn(n, 1280).astype("float32"),
    }


def _sd15_named(n=30):
    return {"embeds": np.random.randn(n, 77, 768).astype("float32")}


def test_backend_tensor_names():
    assert dist_backend("sd15").tensor_names == ("embeds",)
    assert dist_backend("sdxl").tensor_names == ("prompt_embeds", "pooled")


def test_make_backend_rejects_unknown():
    with pytest.raises(ValueError):
        dist_backend("sd3")
    with pytest.raises(ValueError):
        make_backend("nope")


def test_backend_defaults_present():
    for name in ("sd15", "sdxl"):
        d = BACKEND_DEFAULTS[name]
        assert {"steps", "guidance", "height", "width"} <= set(d)


def test_sdxl_fit_sample_shapes():
    b = dist_backend("sdxl")
    dists = b.fit(_sdxl_named(), per_token=True, n_components=8)
    assert set(dists) == {"prompt_embeds", "pooled"}
    assert all(isinstance(d, EmbeddingDistribution) for d in dists.values())

    out = b.sample(dists, n=3, temperature=1.0, rng=np.random.default_rng(0))
    assert out["prompt_embeds"].shape == (3, 77, 2048)
    assert out["pooled"].shape == (3, 1280)


def test_sd15_fit_sample_shapes():
    b = dist_backend("sd15")
    dists = b.fit(_sd15_named(), per_token=True, n_components=8)
    out = b.sample(dists, n=4, temperature=1.0, rng=np.random.default_rng(0))
    assert out["embeds"].shape == (4, 77, 768)


@pytest.mark.parametrize("sampler", ["diagonal", "pca", "blend", "hybrid"])
def test_sampler_dispatch_both_tensors(sampler):
    """Every sampler runs on BOTH SDXL tensors and preserves shape."""
    b = dist_backend("sdxl")
    dists = b.fit(_sdxl_named(), per_token=True, n_components=8)
    out = b.sample(dists, n=2, temperature=1.5, sampler=sampler,
                   coherence=0.5, rng=np.random.default_rng(1))
    assert out["prompt_embeds"].shape == (2, 77, 2048)
    assert out["pooled"].shape == (2, 1280)


def test_pca_temperature_extrapolates_no_clipping():
    """pca deviation magnitude scales LINEARLY with temperature (extrapolation)."""
    b = dist_backend("sdxl")
    dists = b.fit({"pooled": np.random.randn(60, 1280).astype("float32")},
                  per_token=True, n_components=10)
    d = dists["pooled"]
    norms = {}
    for t in (1.0, 2.0, 4.0):
        s = d.sample(n=200, temperature=t, sampler="pca",
                     rng=np.random.default_rng(0))
        norms[t] = float(np.linalg.norm(s - d.mean[None], axis=1).mean())
    # 4x temperature -> ~4x deviation, i.e. no clipping back to the corpus range.
    assert abs(norms[4.0] / norms[1.0] - 4.0) < 0.1


def test_save_load_roundtrip(tmp_path):
    b = dist_backend("sdxl")
    dists = b.fit(_sdxl_named(), per_token=True, n_components=8)
    prefix = tmp_path / "clip_dist_sdxl"
    written = b.save_dists(dists, prefix)
    assert len(written) == 2 and all(p.exists() for p in written)

    loaded = b.load_dists(prefix)
    assert set(loaded) == {"prompt_embeds", "pooled"}
    np.testing.assert_allclose(loaded["pooled"].mean, dists["pooled"].mean)


def test_sd15_single_tensor_save_layout(tmp_path):
    """sd15 saves the lone tensor at <prefix> directly (original layout)."""
    b = dist_backend("sd15")
    dists = b.fit(_sd15_named(), per_token=True, n_components=8)
    prefix = tmp_path / "dist"
    written = b.save_dists(dists, prefix)
    assert written == [tmp_path / "dist.npz"]
    loaded = b.load_dists(prefix)
    np.testing.assert_allclose(loaded["embeds"].mean, dists["embeds"].mean)
