"""/api/wipe: the filters choose what goes, and survivors keep their ancestry."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException

REPO = Path(__file__).resolve().parent.parent
LOW, HIGH, RAW, OLD, FAV = (f"generated/anarchy_sd15_{n}.jpg" for n in
                            ("low", "high", "raw", "old", "fav"))
OTHER = "generated/anarchy_sdxl_other.jpg"
ORIGIN, UPSCALE = "generated/anarchy_sd15_orig.jpg", "generated/anarchy_sd15_orig_hires.jpg"
T_OLD, T_NEW = 1_000_000.0, 2_000_000.0


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("sa_webui_app_wipe", REPO / "webui" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def out(app_module, tmp_path, monkeypatch):
    out = tmp_path / "outputs"
    (out / "generated").mkdir(parents=True)

    def image(rel, when=T_NEW, **meta):
        p = out / rel
        for ext in (".jpg", ".npz"):
            p.with_suffix(ext).write_bytes(b"x")
        p.with_suffix(".json").write_text(json.dumps(
            {"backend": "sd15", "model": "/m/juggernaut.safetensors", **meta}))
        os.utime(p, (when, when))

    for rel in (LOW, HIGH, RAW, FAV, ORIGIN):
        image(rel)
    image(OLD, when=T_OLD)
    image(OTHER, backend="sdxl", model="/m/juggernaut_xl.safetensors")
    image(UPSCALE, refined_from=Path(ORIGIN).name)

    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({"rel": r, "score": s}) + "\n" for r, s in (
        (LOW, 2), (HIGH, 8), (OLD, 2), (FAV, 1), (OTHER, 2), (UPSCALE, 9))))
    (out / "favorites.json").write_text(json.dumps([FAV]))
    tags = tmp_path / "tags.jsonl"
    app_module.tagset.append_record(app_module.tagset.make_record(LOW, ["ember"]), tags)

    monkeypatch.setattr(app_module, "OUTPUTS", out)
    monkeypatch.setattr(app_module, "FAVS_FILE", out / "favorites.json")
    monkeypatch.setattr(app_module, "LABELS_FILE", labels)
    monkeypatch.setattr(app_module, "TAGS_FILE", tags)
    monkeypatch.setattr(app_module, "_ROW_CACHE", {})
    return out


def doomed(mod, **kw):
    return set(mod._wipe_plan(mod.WipeRequest(**kw))["doomed"])


def test_defaults_are_the_original_wipe_below_5_on_labels(app_module, out):
    assert doomed(app_module) == {LOW, OLD, OTHER}


@pytest.mark.parametrize("cutoff,expected", [
    (0, set()),                               # no labeled image scores below 0
    (2, set()),                               # "below" is strict
    (3, {LOW, OLD, OTHER}),
    (9, {LOW, OLD, OTHER, HIGH}),
])
def test_cutoff_is_a_strict_threshold_on_the_label(app_module, out, cutoff, expected):
    assert doomed(app_module, cutoff=cutoff) == expected


def test_unlabeled_images_only_go_when_asked(app_module, out):
    assert RAW not in doomed(app_module)
    assert doomed(app_module, cutoff=0, include_unlabeled=True) >= {RAW}


def test_model_and_time_filters_narrow_the_match(app_module, out):
    assert doomed(app_module, backend="sdxl") == {OTHER}
    assert doomed(app_module, ckpt="juggernaut") == {LOW, OLD}
    assert doomed(app_module, until=T_OLD + 1) == {OLD}
    assert doomed(app_module, since=T_OLD + 1) == {LOW, OTHER}


def test_starred_images_are_spared_unless_included(app_module, out):
    plan = app_module._wipe_plan(app_module.WipeRequest())
    assert FAV not in plan["doomed"] and plan["kept_starred"] == 1
    assert FAV in doomed(app_module, include_starred=True)


def test_a_survivors_original_is_spared_even_if_it_matches(app_module, out):
    plan = app_module._wipe_plan(app_module.WipeRequest(include_unlabeled=True))
    assert UPSCALE not in plan["doomed"]              # scored 9, survives
    assert ORIGIN not in plan["doomed"]               # unlabeled, but the upscale needs it
    assert plan["kept_lineage"] == 1


def test_a_cycle_in_the_lineage_cannot_hang(app_module, out):
    (out / UPSCALE).with_suffix(".json").write_text(
        json.dumps({"backend": "sd15", "refined_from": Path(UPSCALE).name}))
    app_module._ROW_CACHE.clear()
    doomed(app_module)


def test_everything_can_really_be_deleted(app_module, out):
    res = json.loads(app_module.api_wipe(app_module.WipeRequest(
        cutoff=app_module.WIPE_CUTOFF_ALL, include_unlabeled=True,
        include_starred=True)).body)
    assert res["deleted"] == 8
    assert not list((out / "generated").iterdir())


def test_wipe_removes_files_and_traces_but_keeps_labels(app_module, out):
    labels_before = app_module.LABELS_FILE.read_text()
    res = json.loads(app_module.api_wipe(app_module.WipeRequest(cutoff=3)).body)
    assert set(res["rels"]) == {LOW, OLD, OTHER}
    assert res["files"] == 9
    assert not list((out / "generated").glob("anarchy_sd15_low.*"))
    assert (out / HIGH).is_file()
    assert app_module.current_tags().get(LOW) is None
    assert app_module.LABELS_FILE.read_text() == labels_before


def test_preview_reports_what_the_wipe_then_does(app_module, out):
    p = json.loads(app_module.api_wipe_preview(cutoff=3, include_unlabeled=True).body)
    assert (p["labeled"], p["unlabeled"]) == (3, 1)   # RAW; ORIGIN is spared
    assert p["count"] == 4 and p["kept_lineage"] == 1 and p["kept_starred"] == 1
    res = json.loads(app_module.api_wipe(app_module.WipeRequest(
        cutoff=3, include_unlabeled=True, until=p["as_of"])).body)
    assert res["deleted"] == 4


@pytest.mark.parametrize("cutoff", [-1, 11])
def test_refuses_a_cutoff_outside_the_scale(app_module, out, cutoff):
    with pytest.raises(HTTPException) as e:
        app_module.api_wipe(app_module.WipeRequest(cutoff=cutoff))
    assert e.value.status_code == 400
