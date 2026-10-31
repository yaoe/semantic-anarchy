"""POST /api/explore samples against the selected distribution, not outputs/dist."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("sa_webui_app_explore", REPO / "webui" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_explore_passes_the_selected_dist(app_module, tmp_path, monkeypatch):
    gen = tmp_path / "generated"
    gen.mkdir()
    for ext in (".jpg", ".npz"):
        (gen / f"anarchy_sd15_7_000{ext}").write_bytes(b"x")
    submitted = []

    def submit(action, label, argv):
        submitted.append(argv)
        return SimpleNamespace(id="1")

    monkeypatch.setattr(app_module, "OUTPUTS", tmp_path)
    monkeypatch.setattr(app_module, "_model_flags", lambda req: [])
    monkeypatch.setattr(app_module, "resolve_dist_base", lambda backend: f"picked_{backend}")
    monkeypatch.setattr(app_module.RUNNER, "submit", submit)

    res = app_module.api_explore(app_module.ExploreRequest(src="generated/anarchy_sd15_7_000.jpg"))

    assert json.loads(res.body)["job_id"] == "1"
    argv = submitted[0]
    assert argv[argv.index("--dist") + 1] == "picked_sd15"
