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


# --------------------------------------------------------- CFG negative ---

def test_sd15_neg_mode_dispatch():
    """``_negative`` maps neg_mode to an embedding without touching torch.

    ``None`` means "let SDModel use its own negative_prompt" -- that's the
    ``text`` default, the one path that needs a loaded text encoder.
    """
    b = dist_backend("sd15")
    named = _sd15_named(4)
    dists = b.fit(named, per_token=True, n_components=4)

    assert b._negative(named, dists, "text") is None
    zeros = b._negative(named, dists, "zeros")
    assert zeros.shape == (1, 77, 768) and not zeros.any()
    mean = b._negative(named, dists, "mean")
    np.testing.assert_allclose(mean[0], dists["embeds"].mean.reshape(77, 768))
    # mean with no fitted distribution degrades to the default, never crashes
    assert b._negative(named, None, "mean") is None


def test_default_sd15_negative_env(monkeypatch):
    from semantic_anarchy.pipeline import SD15_NEGATIVE_PROMPT, default_sd15_negative

    monkeypatch.delenv("SA_SD15_NEGATIVE", raising=False)
    assert default_sd15_negative() == SD15_NEGATIVE_PROMPT
    monkeypatch.setenv("SA_SD15_NEGATIVE", "blurry, jpeg artifacts")
    assert default_sd15_negative() == "blurry, jpeg artifacts"
    # explicitly empty = no negative text, back to the empty-prompt encoding
    monkeypatch.setenv("SA_SD15_NEGATIVE", "  ")
    assert default_sd15_negative() is None


def test_sd15_neg_mode_default_is_text():
    """resolve_gen_defaults picks the house negative for sd15, mean for sdxl."""
    import argparse

    from semantic_anarchy.cli_args import resolve_gen_defaults

    def resolved(backend):
        a = argparse.Namespace(backend=backend, steps=None, guidance=None,
                               neg_mode=None, height=None, width=None)
        resolve_gen_defaults(a)
        return a.neg_mode

    assert resolved("sd15") == "text"
    assert resolved("sdxl") == "mean"
    assert resolved("sd2") == "empty"


def test_effective_negative_only_for_text_mode():
    """The sidecar records the words, not just the mode -- and only when used."""
    from types import SimpleNamespace

    from semantic_anarchy.cli_args import effective_negative

    sd15 = SimpleNamespace(model=SimpleNamespace(negative_prompt="ugly, blurry"))
    sdxl = SimpleNamespace(model=SimpleNamespace())        # tensor negative, no text

    assert effective_negative(sd15, "text") == "ugly, blurry"
    for mode in ("empty", "mean", "zeros"):
        assert effective_negative(sd15, mode) is None
    assert effective_negative(sdxl, "text") is None


# --------------------------------------------------------------------------- #
# Phase-1 knobs at the backend seam: one length draw, one radius band and one
# knob dict shared by every named tensor, so a multi-tensor conditioning set
# stays coherent.
# --------------------------------------------------------------------------- #
def _lengths(n=30, seq=77, seed=0):
    return np.random.default_rng(seed).integers(3, seq, size=n).astype(np.int32)


def test_fit_adds_the_length_split_only_where_there_is_a_token_axis():
    be = dist_backend("sdxl")
    named = _sdxl_named(n=60)
    dists = be.fit(named, n_components=8, lengths=_lengths(60))
    assert dists["prompt_embeds"].has_length_stats     # (77, 2048)
    assert not dists["pooled"].has_length_stats        # (1280,) -- no token axis


def test_one_length_draw_is_shared_by_every_tensor():
    be = dist_backend("sdxl")
    dists = be.fit(_sdxl_named(n=60), n_components=8, lengths=_lengths(60))
    rng = np.random.default_rng(1)
    lens = be.draw_lengths(dists, 4, rng=rng, mode="fixed", length=20)
    assert lens.tolist() == [20, 20, 20, 20]
    named = be.sample(dists, n=4, rng=np.random.default_rng(2), lengths=lens)
    # pooled has no token axis and simply ignores the knob rather than erroring.
    assert named["prompt_embeds"].shape == (4, 77, 2048)
    assert named["pooled"].shape == (4, 1280)


def test_draw_lengths_returns_none_without_a_length_fit():
    be = dist_backend("sd15")
    dists = be.fit(_sd15_named(n=20), n_components=4)   # no lengths
    assert be.draw_lengths(dists, 4, mode="corpus") is None


def test_sample_radii_averages_the_per_tensor_bands():
    be = dist_backend("sdxl")
    dists = be.fit(_sdxl_named(n=40), n_components=8)
    radii = be.sample_radii(dists, 6, rng=np.random.default_rng(3))
    assert radii.shape == (6,) and (radii > 0).all()
    # And retarget accepts that per-sample band, pinning the reported gauge.
    named = be.sample(dists, n=6, rng=np.random.default_rng(4))
    pinned = be.retarget(dists, named, radii)
    got = [be.distance(dists, {k: v[i] for k, v in pinned.items()})
           for i in range(6)]
    assert np.allclose(got, radii, rtol=1e-3)


def test_length_conditional_flag_matches_the_conditioning_layout():
    for name in ("sd15", "sd2", "sdxl"):
        assert dist_backend(name).length_conditional
    for name in ("flux2", "krea2"):
        assert not dist_backend(name).length_conditional


def test_sampler_kwargs_covers_every_sample_knob():
    """cli_args is the single source of CLI truth -- so it must stay in sync."""
    import argparse
    import inspect

    from semantic_anarchy.cli_args import add_backend_args, sampler_kwargs

    p = argparse.ArgumentParser()
    add_backend_args(p)
    args = p.parse_args([])
    packed = sampler_kwargs(args)
    accepted = set(inspect.signature(EmbeddingDistribution.sample).parameters)
    assert set(packed).issubset(accepted)
    # Every knob sample() takes (bar the plumbing) has a flag behind it.
    assert accepted - set(packed) == {"self", "n", "temperature", "rng"}


def test_neg_dists_kwarg_reaches_every_backend_with_a_tensor_negative():
    """The `mean`/`zeros` neg-modes are a no-op without the fitted dists."""
    import argparse

    from semantic_anarchy.cli_args import neg_dists_kwarg

    dists = {"embeds": object()}
    for name in ("sd15", "sd2", "sdxl"):
        assert neg_dists_kwarg(argparse.Namespace(backend=name), dists) == {"dists": dists}
    for name in ("flux2", "krea2"):
        assert neg_dists_kwarg(argparse.Namespace(backend=name), dists) == {}
