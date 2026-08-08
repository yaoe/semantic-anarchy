"""Torch-free tests for the hires-fix arithmetic (semantic_anarchy/upscale.py)."""

import json

import numpy as np
import pytest

from semantic_anarchy.upscale import (
    LATENT_MULTIPLE, clamp_denoise, conditioning_source, denoise_steps, round_to,
    target_size,
)


def test_target_size_snaps_to_16():
    """Every axis lands on the diffusion grid, whatever the factor."""
    for w, h, f in [(512, 512, 2.0), (768, 512, 1.5), (1024, 1024, 1.37),
                    (640, 896, 2.25), (513, 777, 1.0)]:
        tw, th = target_size(w, h, f)
        assert tw % LATENT_MULTIPLE == 0 and th % LATENT_MULTIPLE == 0
        # ...within half a grid cell of the requested size
        assert abs(tw - w * f) <= LATENT_MULTIPLE / 2
        assert abs(th - h * f) <= LATENT_MULTIPLE / 2


def test_target_size_exact_when_already_aligned():
    """A 16-aligned source at an integer factor is not nudged at all."""
    assert target_size(1024, 768, 2.0) == (2048, 1536)
    assert target_size(512, 512, 1.0) == (512, 512)


def test_target_size_cap_keeps_aspect():
    """--max-side shrinks both axes together, so the crop ratio survives."""
    tw, th = target_size(1024, 512, 4.0, max_side=2048)
    assert max(tw, th) <= 2048
    assert abs((tw / th) - 2.0) < 0.05
    # under the cap, nothing changes
    assert target_size(1024, 512, 1.5, max_side=2048) == (1536, 768)


def test_target_size_never_degenerate():
    assert target_size(8, 8, 0.1) == (LATENT_MULTIPLE, LATENT_MULTIPLE)
    with pytest.raises(ValueError):
        target_size(512, 512, 0.0)


def test_round_to():
    assert round_to(1002) == 1008          # nearest multiple of 16
    assert round_to(1020) == 1024
    assert round_to(1024) == 1024
    assert round_to(1, 16) == 16           # never below one cell


def test_denoise_steps_is_the_tail_of_the_schedule():
    """denoise=0.3 of a 50-step original = the last 15 steps (diffusers' math)."""
    assert denoise_steps(50, 0.3) == 15
    assert denoise_steps(40, 0.25) == 10
    assert denoise_steps(30, 1.0) == 30
    assert denoise_steps(30, 0.0) == 0     # ...which clamp_denoise exists to avoid


def test_clamp_denoise_guarantees_a_pass():
    """A denoise too small to buy a step is nudged to exactly one step."""
    assert denoise_steps(20, clamp_denoise(20, 0.01)) == 1
    assert clamp_denoise(50, 0.3) == 0.3   # sane values pass through untouched
    assert clamp_denoise(50, 2.0) == 1.0   # and it never exceeds a full re-render


def _png(path, size=(64, 64)):
    """A real (tiny) PNG so the header/size readers have something to chew on."""
    from struct import pack
    from zlib import crc32

    def chunk(tag, data):
        return pack(">I", len(data)) + tag + data + pack(">I", crc32(tag + data))

    ihdr = pack(">IIBBBBB", size[0], size[1], 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b""))


def test_conditioning_source_finds_the_image_itself(tmp_path):
    src = tmp_path / "anarchy_sd15_7_000.png"
    _png(src)
    np.savez(src.with_suffix(".npz"), embeds=np.zeros((77, 768), "float32"))
    src.with_suffix(".json").write_text(json.dumps({"kind": "generate", "steps": 50}))

    origin, meta = conditioning_source(src)
    assert origin == src
    assert meta["steps"] == 50


def test_conditioning_source_walks_a_chain_of_upscales(tmp_path):
    """2x then 2x again: the third image traces back to the original's latents."""
    orig = tmp_path / "anarchy_sd15_7_000.png"
    _png(orig)
    np.savez(orig.with_suffix(".npz"), embeds=np.zeros((77, 768), "float32"))
    orig.with_suffix(".json").write_text(
        json.dumps({"kind": "generate", "steps": 50, "scheduler": "ddim"}))

    up1 = tmp_path / "anarchy_sd15_7_000_hires2p0.png"
    _png(up1)
    up1.with_suffix(".json").write_text(
        json.dumps({"kind": "refine", "refined_from": orig.name}))

    up2 = tmp_path / "anarchy_sd15_7_000_hires2p0_hires2p0.png"
    _png(up2)
    up2.with_suffix(".json").write_text(
        json.dumps({"kind": "refine", "refined_from": up1.name}))

    origin, meta = conditioning_source(up2)
    assert origin == orig
    assert meta["scheduler"] == "ddim"


def test_conditioning_source_raises_when_the_chain_runs_dry(tmp_path):
    lone = tmp_path / "some_upload.png"
    _png(lone)
    with pytest.raises(FileNotFoundError):
        conditioning_source(lone)

    # a dangling refined_from is a dead end, not an infinite loop
    orphan = tmp_path / "anarchy_sd15_7_000_hires2p0.png"
    _png(orphan)
    orphan.with_suffix(".json").write_text(
        json.dumps({"kind": "refine", "refined_from": "gone.png"}))
    with pytest.raises(FileNotFoundError):
        conditioning_source(orphan)
