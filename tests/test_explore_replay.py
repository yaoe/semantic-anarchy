"""explore.py renders children the way their source was rendered.

The conditioning is only half of an image's look: the same .npz through another
checkpoint, size, scheduler or CFG negative is a different picture. These pin
that every render knob not passed explicitly is copied from the source sidecar.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PIL = pytest.importorskip("PIL.Image")
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def explore():
    spec = importlib.util.spec_from_file_location("sa_explore_script", REPO / "scripts" / "explore.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _src(tmp_path, **meta):
    ckpt = tmp_path / "juggernaut.safetensors"
    ckpt.write_bytes(b"")
    src = tmp_path / "anarchy_sd15_1_000.jpg"
    PIL.new("RGB", (640, 960)).save(src)
    src.with_suffix(".json").write_text(json.dumps({
        "model": str(ckpt), "steps": 40, "guidance": 6.5, "scheduler": "euler_a",
        "neg_mode": "empty", "negative": None, "image_seed": 99, **meta}))
    return src, ckpt


def _args(**kw):
    base = dict(steps=None, guidance=None, scheduler="default", neg_mode=None,
                negative=None, height=None, width=None, ckpt=None, model=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_unset_knobs_are_copied_from_the_source(explore, tmp_path):
    src, ckpt = _src(tmp_path)
    args = _args()
    explore.replay_source_params(args, src)
    assert (args.steps, args.guidance, args.scheduler, args.neg_mode) == (40, 6.5, "euler_a", "empty")
    assert (args.width, args.height) == (640, 960)      # the TRUE size, off the pixels
    assert args.ckpt == str(ckpt)


def test_explicit_flags_win_unless_stay_local(explore, tmp_path):
    src, _ = _src(tmp_path)
    args = _args(steps=12, scheduler="ddim", ckpt="other.safetensors")
    explore.replay_source_params(args, src)
    assert (args.steps, args.scheduler, args.ckpt) == (12, "ddim", "other.safetensors")
    explore.replay_source_params(args, src, overwrite=True)
    assert (args.steps, args.scheduler) == (40, "euler_a")


def test_a_missing_checkpoint_is_not_copied(explore, tmp_path):
    src, ckpt = _src(tmp_path)
    ckpt.unlink()
    args = _args()
    explore.replay_source_params(args, src)
    assert args.ckpt is None
