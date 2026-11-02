"""POST /api/image/delete: removes one image's files and every trace derived from it."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REL = "generated/anarchy_sd15_7_000.jpg"
OTHER = "generated/anarchy_sd15_7_001.jpg"


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("sa_webui_app", REPO / "webui" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def world(app_module, tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    gen = out / "generated"
    gen.mkdir(parents=True)
    for rel in (REL, OTHER):
        p = out / rel
        for ext in (".jpg", ".json", ".npz"):
            p.with_suffix(ext).write_bytes(b"{}")
    (out / "favorites.json").write_text(json.dumps([REL, OTHER]))
    for cache in ("scores.json", "novelty.json", "resonance.json"):
        (out / cache).write_text(json.dumps({REL: 1.0, OTHER: 2.0}))
    np.savez_compressed(out / "clip_embeds.npz", names=np.array([REL, OTHER]),
                        vecs=np.zeros((2, 4), np.float16))
    tags = tmp_path / "tags.jsonl"
    app_module.tagset.append_record(app_module.tagset.make_record(REL, ["ember"]), tags)
    labels = tmp_path / "labels.jsonl"
    labels.write_text('{"rel": "%s", "score": 7}\n' % REL)
    monkeypatch.setattr(app_module, "OUTPUTS", out)
    monkeypatch.setattr(app_module, "FAVS_FILE", out / "favorites.json")
    monkeypatch.setattr(app_module, "TAGS_FILE", tags)
    monkeypatch.setattr(app_module, "LABELS_FILE", labels)
    return out, labels


def delete(mod, rel):
    return json.loads(mod.api_image_delete(mod.DeleteImageRequest(rel=rel)).body)


def test_removes_image_and_sidecars_only(app_module, world):
    out, _ = world
    assert delete(app_module, REL)["files"] == 3
    assert not list((out / "generated").glob("anarchy_sd15_7_000.*"))
    assert len(list((out / "generated").glob("anarchy_sd15_7_001.*"))) == 3


def test_every_extension_of_the_stem_goes(app_module, world):
    out, _ = world
    (out / REL).with_suffix(".png").write_bytes(b"x")
    assert delete(app_module, REL)["files"] == 4
    assert not list((out / "generated").glob("anarchy_sd15_7_000.*"))


def test_purges_derived_state_but_keeps_the_label(app_module, world):
    out, labels = world
    before = labels.read_text()
    delete(app_module, REL)
    assert json.loads((out / "favorites.json").read_text()) == [OTHER]
    for cache in ("scores.json", "novelty.json", "resonance.json"):
        assert set(json.loads((out / cache).read_text())) == {OTHER}
    assert [str(n) for n in np.load(out / "clip_embeds.npz")["names"]] == [OTHER]
    assert app_module.current_tags().get(REL) is None
    assert labels.read_text() == before


def test_deleting_twice_is_not_an_error(app_module, world):
    delete(app_module, REL)
    assert delete(app_module, REL)["files"] == 0


@pytest.mark.parametrize("rel", [
    "../outside/anarchy_sd15_7_000.jpg",     # escapes outputs/
    "generated/anarchy_sd15_7_000.npz",      # not an image path
    "analysis/plots/05_spectrum.png",        # an image, but not a gallery one
])
def test_refuses_anything_but_a_gallery_image(app_module, world, tmp_path, rel):
    out, _ = world
    for path in (out / rel, tmp_path / "outside" / "anarchy_sd15_7_000.jpg"):
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
        path.resolve().write_bytes(b"x")
    with pytest.raises(HTTPException) as e:
        delete(app_module, rel)
    assert e.value.status_code == 404
    assert (out / rel).resolve().exists()
