"""Torch-free tests for fitting a distribution to a hand-picked set of images.

The property that matters is the one the old evolve branch did NOT have: a
``pca`` draw from a selection fit stays inside the affine span of the latents
that were selected. Everything else here guards the plumbing around it — which
sidecars are usable, what a stray one does, and where the files land.
"""

import json

import numpy as np
import pytest

from semantic_anarchy.dist_paths import dist_files
from semantic_anarchy.distribution import EmbeddingDistribution
from semantic_anarchy.selection_fit import (
    MANIFEST_SUFFIX, fit_base, fit_latents, latents_for, list_fits,
    manifest_path, read_manifest, save_fit, slug_name, stack_latents,
    write_manifest,
)


def _write_image(dirpath, stem, embeds, *, meta=None, ext=".jpg"):
    """A generated image as the gallery knows it: pixels + .npz + .json."""
    img = dirpath / f"{stem}{ext}"
    img.write_bytes(b"not really a jpeg")
    if embeds is not None:
        np.savez(dirpath / f"{stem}.npz", embeds=np.asarray(embeds, dtype=np.float32))
    (dirpath / f"{stem}.json").write_text(json.dumps(meta or {"backend": "sd15"}))
    return img


# ------------------------------------------------------------------ naming --
def test_slug_name_is_filename_safe():
    assert slug_name("my keepers · week 3") == "my-keepers-week-3"
    assert slug_name("  --weird--  ") == "weird"


def test_fit_base_and_manifest_sit_together(tmp_path):
    base = fit_base("keepers", tmp_path)
    assert base == str(tmp_path / "keepers")
    assert manifest_path(base) == tmp_path / ("keepers" + MANIFEST_SUFFIX)


def test_empty_name_is_rejected():
    with pytest.raises(ValueError):
        fit_base("···")


# ---------------------------------------------------------------- gathering --
def test_latents_for_follows_the_refine_chain(tmp_path):
    """An upscale has no conditioning of its own — it contributes its
    ancestor's, which is what makes an upscaled keeper selectable."""
    _write_image(tmp_path, "orig", np.zeros((2, 3)))
    up = _write_image(tmp_path, "up", None,
                      meta={"kind": "refine", "refined_from": "orig.jpg"})
    assert latents_for(up) == tmp_path / "orig.npz"


def test_latents_for_gives_up_on_a_dead_chain(tmp_path):
    lonely = _write_image(tmp_path, "lonely", None, meta={"kind": "refine"})
    assert latents_for(lonely) is None


def test_stack_latents_skips_the_stray_shapes(tmp_path):
    """A selection is assembled by eye, so one image of the wrong shape (or one
    missing the tensor entirely) must cost that image, not the whole fit."""
    good = [tmp_path / f"g{i}.npz" for i in range(3)]
    for p in good:
        np.savez(p, embeds=np.random.default_rng(0).normal(size=(4, 5)))
    odd = tmp_path / "odd.npz"
    np.savez(odd, embeds=np.zeros((7, 5)))       # different feature shape
    empty = tmp_path / "empty.npz"
    np.savez(empty, pooled=np.zeros(4))          # no 'embeds' at all

    stacked, used, skipped = stack_latents([*good, odd, empty], ["embeds"])
    assert stacked["embeds"].shape == (3, 4, 5)
    assert used == good
    assert [p for p, _ in skipped] == [odd, empty]
    assert "expected" in dict(skipped)[odd]


def test_stack_latents_needs_every_named_tensor(tmp_path):
    """sdxl fits two tensors; a sidecar holding only one can't serve either."""
    half = tmp_path / "half.npz"
    np.savez(half, prompt_embeds=np.zeros((4, 5)))
    stacked, used, skipped = stack_latents([half], ["prompt_embeds", "pooled"])
    assert used == [] and stacked == {}
    assert "pooled" in skipped[0][1]


# -------------------------------------------------------------------- fitting --
def _selection(n=6, shape=(4, 5), seed=0):
    rng = np.random.default_rng(seed)
    return {"embeds": rng.normal(size=(n, *shape))}


def test_fit_keeps_the_selections_full_rank():
    dists = fit_latents(_selection(n=6))
    d = dists["embeds"]
    assert d.n_samples == 6
    assert d.pca_std.shape == (5,)          # N-1 axes: the affine span, exactly


def test_pca_samples_stay_in_the_span_of_the_selection():
    """The whole point. A pca draw is a combination of the picked latents, so it
    lies in their affine span — unlike a branch that grafts a corpus basis onto
    a moved centre, whose draws leave the span in every direction at once."""
    sel = _selection(n=5, shape=(3, 4))
    d = fit_latents(sel)["embeds"]
    x = sel["embeds"].reshape(5, -1)
    centered = x - x.mean(axis=0)
    # Basis of the span, from the data rather than from the fit under test.
    q = np.linalg.svd(centered, full_matrices=False)[2][:4]

    drawn = d.sample(8, sampler="pca", rng=np.random.default_rng(1)).reshape(8, -1)
    dev = drawn - x.mean(axis=0)
    residual = dev - dev @ q.T @ q
    assert np.abs(residual).max() < 1e-4


def test_fit_needs_more_than_one_sample():
    with pytest.raises(ValueError):
        fit_latents({"embeds": np.zeros((1, 4, 5))})


def test_components_cap_is_honoured():
    d = fit_latents(_selection(n=10), n_components=3)["embeds"]
    assert d.pca_std.shape == (3,)


# ------------------------------------------------------------------ on disk --
def test_saved_fit_is_an_ordinary_distribution(tmp_path):
    """It has to load through the same path a mined corpus does — that is what
    makes it selectable as a base distribution with every sampler working."""
    base = fit_base("keepers", tmp_path)
    written = save_fit(fit_latents(_selection()), base, "sd15")
    assert written == dist_files(base, "sd15")
    reloaded = EmbeddingDistribution.load(written[0])
    assert reloaded.n_samples == 6
    assert reloaded.pca_components is not None


def test_manifest_round_trip_and_listing(tmp_path):
    base = fit_base("keepers", tmp_path)
    save_fit(fit_latents(_selection()), base, "sd15")
    write_manifest(base, "sd15", ["outputs/generated/a", "outputs/generated/b"],
                   name="keepers", note="week 3", models=["juggernaut", "juggernaut"])

    man = read_manifest(base)
    assert man["n_samples"] == 2 and man["models"] == ["juggernaut"]

    rows = list_fits(tmp_path, "sd15")
    assert [r["name"] for r in rows] == ["keepers"]
    assert rows[0]["ready"] is True


def test_listing_flags_a_fit_whose_files_are_gone(tmp_path):
    base = fit_base("ghost", tmp_path)
    write_manifest(base, "sd15", ["a", "b", "c"], name="ghost")
    assert list_fits(tmp_path, "sd15")[0]["ready"] is False


def test_listing_an_empty_dir(tmp_path):
    assert list_fits(tmp_path / "nope", "sd15") == []
