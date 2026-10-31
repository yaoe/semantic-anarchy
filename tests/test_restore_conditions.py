"""POST /api/restore: find what sampled an image and re-select its model + fit."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("sa_webui_app_restore", REPO / "webui" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def world(app_module, tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    (out / "generated").mkdir(parents=True)
    monkeypatch.setattr(app_module, "OUTPUTS", out)
    monkeypatch.setattr(app_module, "REPO", tmp_path)
    monkeypatch.setattr(app_module, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(app_module, "browse_roots", lambda: [tmp_path])
    ckpt = tmp_path / "juggernaut.safetensors"
    ckpt.write_bytes(b"x")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a prompt\n")

    def image(name, **meta):
        p = out / "generated" / name
        p.write_bytes(b"x")
        p.with_suffix(".json").write_text(json.dumps(meta))
        return f"generated/{name}"

    class W:
        pass
    w = W()
    w.tmp, w.ckpt, w.corpus, w.image = tmp_path, ckpt, corpus, image
    w.fit_base = app_module.dist_base_for("sd15", "prompts", str(corpus), None)
    w.call = lambda rel: json.loads(
        app_module.api_restore(app_module.RestoreRequest(rel=rel)).body)
    w.config = lambda: json.loads((tmp_path / "config.json").read_text())
    return w


def gen(w, name="anarchy_sd15_7_000.jpg", **over):
    return w.image(name, **{"kind": "generate", "backend": "sd15", "model": str(w.ckpt),
                            "sampler": "pca", "temperature": 1.7, "steps": 20,
                            "scheduler": "ddim", "height": 640, "width": 960, **over})


def test_generate_selects_checkpoint_and_corpus_fit(world):
    # The fit is named for the checkpoint, so it only resolves once that is active.
    base = world.tmp / world.corpus.stem
    fit = Path(f"{base}__juggernaut.npz")
    fit.write_bytes(b"x")
    rel = gen(world, dist=str(fit.with_suffix("")))
    res = world.call(rel)
    assert res["source"] == rel and res["warnings"] == []
    assert res["params"]["sampler"] == "pca"
    cfg = world.config()
    assert cfg["models"]["sd15"] == str(world.ckpt)
    assert cfg["dists"]["sd15"] == {"kind": "prompts", "path": str(world.corpus)}


def test_saved_fit_is_selected_as_a_file(world):
    fit = world.tmp / "outputs" / "dist_fits"
    fit.mkdir()
    (fit / "keepers.npz").write_bytes(b"x")
    world.call(gen(world, dist="outputs/dist_fits/keepers"))
    assert world.config()["dists"]["sd15"] == {
        "kind": "file", "path": str(fit / "keepers.npz")}


def test_missing_checkpoint_and_dist_warn_but_still_return_knobs(world):
    rel = gen(world, model="/nope/gone.safetensors", dist="outputs/dist_deleted")
    res = world.call(rel)
    assert res["params"]["temperature"] == 1.7
    assert any("gone.safetensors" in w for w in res["warnings"])
    assert any("dist_deleted" in w for w in res["warnings"])


def test_upscale_chain_reaches_the_generating_image(world):
    src = gen(world, dist=None)
    up = world.image("anarchy_sd15_7_000_hires2p0.jpg", kind="refine", engine="hires",
                     backend="sd15", steps=99, refined_from=Path(src).name)
    up2 = world.image("anarchy_sd15_7_000_hires2p0_hires2p0.jpg", kind="refine",
                      backend="sd15", steps=98, refined_from=Path(up).name)
    res = world.call(up2)
    assert res["source"] == src
    assert res["params"]["steps"] == 20, "the upscale's own steps are not its sampling"


def test_explore_keeps_ancestor_sampler_but_its_own_render_knobs(world):
    src = gen(world, dist=None)
    kid = world.image("anarchy_sd15_8_000.jpg", kind="explore", mode="walk", backend="sd15",
                      model=str(world.ckpt), parent=Path(src).name, steps=50, scheduler="euler")
    res = world.call(kid)
    assert res["params"]["sampler"] == "pca"
    assert res["params"]["steps"] == 50 and res["params"]["scheduler"] == "euler"
    assert any("walk" in w for w in res["warnings"])


def test_explore_with_deleted_parent_says_so(world):
    kid = world.image("anarchy_sd15_8_000.jpg", kind="explore", backend="sd15",
                      model=str(world.ckpt), parent="anarchy_sd15_gone_000.jpg", steps=50)
    res = world.call(kid)
    assert res["params"]["steps"] == 50
    assert any("ancestor is gone" in w for w in res["warnings"])


def test_image_without_sampling_record_is_refused(world):
    rel = world.image("anarchy_sd15_9_000.jpg", kind="from_prompt", backend="sd15")
    with pytest.raises(HTTPException) as err:
        world.call(rel)
    assert err.value.status_code == 400


def test_a_refine_cycle_terminates(world):
    a = world.image("anarchy_sd15_1.jpg", kind="refine", refined_from="anarchy_sd15_2.jpg")
    world.image("anarchy_sd15_2.jpg", kind="refine", refined_from="anarchy_sd15_1.jpg")
    with pytest.raises(HTTPException):
        world.call(a)
