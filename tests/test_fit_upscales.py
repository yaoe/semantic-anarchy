"""🧬 Fit: an upscale/refine has no .npz of its own but fits through its original's.

Regression: the candidate grid checked only the image's OWN sidecar, so every
tagged FLUX upscale (``*_flux1p5.jpg``) vanished from a universe the moment the
Fit tab filtered on it, while the gallery still showed it.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("sa_webui_app_fitup", REPO / "webui" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gen(app, monkeypatch, tmp_path):
    out = tmp_path / "outputs"
    g = out / "generated"
    g.mkdir(parents=True)
    for i in range(3):                                   # three originals with latents
        stem = g / f"anarchy_sd15_7_00{i}"
        stem.with_suffix(".jpg").write_bytes(b"x")
        stem.with_suffix(".npz").write_bytes(b"x")
        stem.with_suffix(".json").write_text(json.dumps({"backend": "sd15"}))
    # a flux upscale of #0, and an upscale of that upscale — no .npz, no backend
    up = g / "anarchy_sd15_7_000_flux1p5"
    up.with_suffix(".jpg").write_bytes(b"x")
    up.with_suffix(".json").write_text(json.dumps({"refined_from": "anarchy_sd15_7_000.jpg"}))
    up2 = g / "anarchy_sd15_7_000_flux1p5_flux1p5"
    up2.with_suffix(".jpg").write_bytes(b"x")
    up2.with_suffix(".json").write_text(
        json.dumps({"refined_from": "anarchy_sd15_7_000_flux1p5.jpg"}))
    # an upscale whose original is gone: genuinely no latents
    dead = g / "anarchy_sd15_9_000_flux1p5"
    dead.with_suffix(".jpg").write_bytes(b"x")
    dead.with_suffix(".json").write_text(json.dumps({"refined_from": "anarchy_sd15_9_000.jpg"}))

    tags = {f"generated/{n}.jpg": ["minimal"] for n in (
        "anarchy_sd15_7_000_flux1p5", "anarchy_sd15_7_000_flux1p5_flux1p5",
        "anarchy_sd15_7_001", "anarchy_sd15_9_000_flux1p5")}
    seen = {}

    class Job:
        id = 1

    monkeypatch.setattr(app, "OUTPUTS", out)
    monkeypatch.setattr(app, "REPO", tmp_path)
    monkeypatch.setattr(app, "FIT_DIR", tmp_path / "fits")
    monkeypatch.setattr(app, "_ROW_CACHE", {})
    monkeypatch.setattr(app, "_SIDECAR_CACHE", {})
    monkeypatch.setattr(app, "current_tags", lambda: tags)
    monkeypatch.setattr(app, "labeled_scores", lambda: {})
    monkeypatch.setattr(app, "load_favs", lambda: set())
    monkeypatch.setattr(app.RUNNER, "submit",
                        lambda kind, label, argv, **kw: seen.update(argv=argv) or Job())
    return seen


def test_tagged_upscales_are_fit_candidates(app, gen):
    r = json.loads(app.api_fit_candidates(tag="minimal").body)
    rels = {row["rel"] for row in r["rows"]}
    assert rels == {"generated/anarchy_sd15_7_000_flux1p5.jpg",
                    "generated/anarchy_sd15_7_000_flux1p5_flux1p5.jpg",
                    "generated/anarchy_sd15_7_001.jpg"}   # the dead chain stays out
    assert r["backends"] == ["sd15"]


def test_an_upscale_and_its_original_are_one_sample(app, gen):
    rels = [f"generated/{n}.jpg" for n in (
        "anarchy_sd15_7_000", "anarchy_sd15_7_000_flux1p5", "anarchy_sd15_7_000_flux1p5_flux1p5",
        "anarchy_sd15_7_001", "anarchy_sd15_7_002")]
    r = json.loads(app.api_fit(app.FitRequest(rels=rels, name="minimal")).body)
    assert r["n"] == 3 and r["backend"] == "sd15"
    argv = gen["argv"]
    src = json.loads(Path(argv[argv.index("--from-file") + 1]).read_text())
    assert len(src) == 3
