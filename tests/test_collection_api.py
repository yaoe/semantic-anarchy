"""POST /api/collection/sample: validation + the argv it queues."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("sa_webui_app_coll", REPO / "webui" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def setup(app, monkeypatch, tmp_path):
    gen = tmp_path / "generated"
    gen.mkdir()
    for i in range(3):
        for ext in (".jpg", ".npz"):
            (gen / f"anarchy_sd15_7_00{i}{ext}").write_bytes(b"x")
    (gen / "anarchy_sdxl_1_000.jpg").write_bytes(b"x")
    (gen / "anarchy_sdxl_1_000.npz").write_bytes(b"x")
    seen = {}

    class Job:
        id = 3

    monkeypatch.setattr(app, "OUTPUTS", tmp_path)
    monkeypatch.setattr(app, "COLLECTION_DIR", tmp_path / "collections")
    monkeypatch.setattr(app, "resolve_dist_base", lambda b: "outputs/dist")
    monkeypatch.setattr(app.RUNNER, "submit",
                        lambda kind, label, argv, **kw: seen.update(argv=argv) or Job())
    return seen


def _body(app, **kw):
    base = dict(rels=[f"generated/anarchy_sd15_7_00{i}.jpg" for i in range(3)], spread=0.4, n=6)
    base.update(kw)
    return app.CollectionSampleRequest(**base)


def test_queues_the_script_with_a_sources_file(app, setup):
    r = json.loads(app.api_collection_sample(_body(app)).body)
    argv = setup["argv"]
    assert "scripts/sample_collection.py" in argv and r["n_members"] == 3
    src = Path(argv[argv.index("--from-file") + 1])
    assert len(json.loads(src.read_text())) == 3
    assert argv[argv.index("--spread") + 1] == "0.4"


@pytest.mark.parametrize("kw", [dict(method="x; rm"), dict(spread=9), dict(n=0), dict(rels=[])])
def test_rejects_bad_input(app, setup, kw):
    with pytest.raises(app.HTTPException):
        app.api_collection_sample(_body(app, **kw))


def test_rejects_mixed_backends(app, setup):
    rels = [f"generated/anarchy_sd15_7_00{i}.jpg" for i in range(3)] + ["generated/anarchy_sdxl_1_000.jpg"]
    with pytest.raises(app.HTTPException) as e:
        app.api_collection_sample(_body(app, rels=rels))
    assert "mixes backends" in e.value.detail
