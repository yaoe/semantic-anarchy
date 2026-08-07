"""Torch-free tests for the output-naming + image-format rules (io_utils.py).

Renders are JPEG and everything mined before the switch is PNG, so the load-bearing
rule is that the *stem* is an image's identity: a new name must not land on an
older render's sidecars, and a lookup must find the file whatever its extension.
"""

import pytest

from semantic_anarchy.io_utils import (
    IMAGE_EXTS, find_image, image_ext, jpeg_quality, unique_image_path,
    unique_path, with_image_ext,
)


def test_default_format_is_jpg(monkeypatch):
    monkeypatch.delenv("SA_IMAGE_FORMAT", raising=False)
    assert image_ext() == ".jpg"


@pytest.mark.parametrize("raw,want", [
    ("png", ".png"), (".png", ".png"), ("PNG", ".png"),
    ("jpeg", ".jpg"), ("webp", ".webp"), ("gif", ".jpg"), ("", ".jpg"),
])
def test_format_override(monkeypatch, raw, want):
    monkeypatch.setenv("SA_IMAGE_FORMAT", raw)
    assert image_ext() == want


def test_quality_is_clamped_and_falls_back(monkeypatch):
    monkeypatch.setenv("SA_JPEG_QUALITY", "80")
    assert jpeg_quality() == 80
    monkeypatch.setenv("SA_JPEG_QUALITY", "400")
    assert jpeg_quality() == 100
    monkeypatch.setenv("SA_JPEG_QUALITY", "high")
    assert jpeg_quality() == 95


def test_with_image_ext(monkeypatch, tmp_path):
    monkeypatch.setenv("SA_IMAGE_FORMAT", "jpg")
    assert with_image_ext(tmp_path / "a.png").name == "a.jpg"


def test_unique_image_path_free_name(monkeypatch, tmp_path):
    monkeypatch.setenv("SA_IMAGE_FORMAT", "jpg")
    p = unique_image_path(tmp_path / "anarchy_sd15_7_000.jpg")
    assert p.name == "anarchy_sd15_7_000.jpg"


@pytest.mark.parametrize("taken", [".png", ".jpg", ".webp", ".npz", ".json"])
def test_unique_image_path_never_lands_on_a_claimed_stem(tmp_path, taken):
    """A .jpg looks free next to an older .png -- and would clobber its sidecars."""
    (tmp_path / f"anarchy_sd15_7_000{taken}").write_bytes(b"x")
    p = unique_image_path(tmp_path / "anarchy_sd15_7_000.jpg")
    assert p.name == "anarchy_sd15_7_000_1.jpg"
    assert not p.exists()


def test_unique_image_path_counts_up(tmp_path):
    for stem in ("anarchy_sd15_7_000", "anarchy_sd15_7_000_1", "anarchy_sd15_7_000_2"):
        (tmp_path / f"{stem}.png").write_bytes(b"x")
    assert unique_image_path(tmp_path / "anarchy_sd15_7_000.jpg").name == \
        "anarchy_sd15_7_000_3.jpg"


def test_unique_path_still_only_looks_at_the_exact_name(tmp_path):
    """The general helper is unchanged: non-image writes key on the full name."""
    (tmp_path / "dist.npz").write_bytes(b"x")
    assert unique_path(tmp_path / "dist.npz").name == "dist_1.npz"
    assert unique_path(tmp_path / "dist.meta.json").name == "dist.meta.json"


def test_find_image_resolves_across_formats(tmp_path):
    png = tmp_path / "anarchy_sd15_7_000.png"
    png.write_bytes(b"x")
    # a persisted .jpg rel-path (label, favorite, bookmark) still finds the PNG
    assert find_image(tmp_path / "anarchy_sd15_7_000.jpg") == png
    assert find_image(png) == png
    assert find_image(tmp_path / "nope.jpg") is None


def test_find_image_prefers_the_current_format(monkeypatch, tmp_path):
    monkeypatch.setenv("SA_IMAGE_FORMAT", "jpg")
    for e in IMAGE_EXTS:
        (tmp_path / f"a{e}").write_bytes(b"x")
    assert find_image(tmp_path / "a.tmp").suffix == ".jpg"
    monkeypatch.setenv("SA_IMAGE_FORMAT", "png")
    assert find_image(tmp_path / "a.tmp").suffix == ".png"
