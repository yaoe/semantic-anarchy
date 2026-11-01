"""Dashboard orchestration: server routing and the multi-lane job runner.

webui/app.py imports fastapi but no torch, so these stay in the torch-free tier.
"""

import importlib.util
import re
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("sa_webui_app", REPO / "webui" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolve annotations through it
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


@pytest.fixture
def servers(app, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "COMFY_SERVERS_FILE", tmp_path / "none.json")
    monkeypatch.setenv("SA_COMFY_SERVERS", "gpu1=http://gpu1.example:8188/,local=http://x,"
                                           "bad key=http://y,gpu2=ftp://z,gpu3=https://gpu3.example")
    monkeypatch.delenv("SA_COMFY_URL", raising=False)
    return app


# ---- server allow-list and routing -----------------------------------------
def test_server_list_keeps_only_valid_keys_and_http_urls(servers):
    assert servers.comfy_servers() == {"gpu1": "http://gpu1.example:8188",
                                       "gpu3": "https://gpu3.example"}


def test_process_default_is_offered_and_picked(servers, monkeypatch):
    monkeypatch.setenv("SA_COMFY_URL", "https://gpu3.example")
    assert servers.default_server() == "gpu3"
    monkeypatch.setenv("SA_COMFY_URL", "https://other.example")
    assert servers.comfy_servers()["default"] == "https://other.example"


def test_routing_names_keys_never_urls(servers):
    app = servers
    assert app.resolve_server("sdxl", "gpu1") == ("local", None)       # only sd15 leaves
    assert app.resolve_server("sd15", "local") == ("local", None)
    assert app.resolve_server("sd15", "gpu1") == ("gpu1", "http://gpu1.example:8188")
    assert app.resolve_server("sd15", None) == ("local", None)         # no SA_COMFY_URL
    with pytest.raises(app.HTTPException):
        app.resolve_server("sd15", "http://evil.example")


# ---- V1: work that drives the local pipeline never goes to ComfyUI --------
def test_local_pipeline_jobs_never_get_the_comfy_checkpoint(servers, monkeypatch, tmp_path):
    app = servers
    ckpt = tmp_path / "local-sd15.safetensors"
    ckpt.write_bytes(b"")
    monkeypatch.setitem(app.SINGLE_FILE_CKPT, "sd15", str(ckpt))
    monkeypatch.setattr(app, "selected_model", lambda backend: None)
    monkeypatch.setenv("SA_COMFY_URL", "https://gpu3.example")
    remote = app._model_flags(app.RunRequest(action="generate", backend="sd15"))
    local = app._model_flags(app.RunRequest(action="refine", backend="sd15",
                                            server=app.LOCAL_SERVER))
    assert remote == ["--ckpt", app.COMFY_SD15_CKPT]
    assert local == ["--ckpt", str(ckpt)]
    assert app.job_env(None) == {"SA_COMFY_URL": None}


def test_make_backend_honours_remote_ok(monkeypatch):
    from semantic_anarchy import backend as B
    monkeypatch.setenv("SA_COMFY_URL", "http://gpu.example")
    monkeypatch.setattr(B.ComfySD15Backend, "load", classmethod(lambda cls, *a, **k: "comfy"))
    monkeypatch.setattr(B.SD15Backend, "load", classmethod(lambda cls, *a, **k: "local"))
    assert B.make_backend("sd15") == "comfy"
    assert B.make_backend("sd15", remote_ok=False) == "local"


def test_scripts_that_drive_the_pipeline_opt_out_of_remote():
    """Any script touching ``backend.model`` (refine_image, .pipe, ...) must load
    with remote_ok=False -- the ComfyUI backend has no local model."""
    offenders = []
    for p in sorted((REPO / "scripts").glob("*.py")):
        src = p.read_text()
        if re.search(r"\b(backend|be)\.model\.(pipe|refine_image|device)", src):
            if not re.search(r"(load_backend|make_backend)\([^)]*remote_ok=False", src):
                offenders.append(p.name)
    assert not offenders, f"load the backend with remote_ok=False in: {offenders}"


# ---- V3/V5: the runner ---------------------------------------------------
def _wait(runner, jobs, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if all(runner.get(j.id).status in ("done", "error", "cancelled") for j in jobs):
            return
        time.sleep(0.05)
    raise AssertionError("jobs did not finish")


def _py(code):
    return [sys.executable, "-c", code]


def test_job_env_overrides_and_unsets(app, monkeypatch):
    monkeypatch.setenv("SA_COMFY_URL", "http://inherited.example")
    r = app.Runner()
    show = "import os; print('URL=' + str(os.environ.get('SA_COMFY_URL')))"
    a = r.submit("t", "unset", _py(show), env={"SA_COMFY_URL": None})
    b = r.submit("t", "set", _py(show), env={"SA_COMFY_URL": "http://gpu1.example"})
    _wait(r, [a, b])
    assert "URL=None" in a.log and "URL=http://gpu1.example" in b.log


def test_lanes_run_in_parallel_and_snapshot_lists_them(app):
    r = app.Runner()
    jobs = [r.submit("t", f"lane {l}", _py("import time; time.sleep(0.8)"), lane=l)
            for l in ("local", "gpu1", "gpu2")]
    time.sleep(0.4)
    snap = r.snapshot()
    assert len(snap["running_ids"]) == 3 and snap["running"] in snap["running_ids"]
    _wait(r, jobs)
    starts = [j.started for j in jobs]
    assert max(starts) - min(starts) < 0.5


def test_exclusive_job_never_overlaps_other_lanes(app):
    r = app.Runner()
    first = r.submit("t", "running", _py("import time; time.sleep(0.6)"), lane="gpu1")
    time.sleep(0.15)
    mine = r.submit("mine", "mine", _py("import time; time.sleep(0.3)"), lane="gpu2",
                    exclusive=True)
    time.sleep(0.05)
    later = r.submit("t", "later", _py("pass"), lane="local")
    _wait(r, [first, mine, later])
    assert mine.started >= first.ended            # waited for the running lane
    assert later.started >= mine.ended            # and held the queue behind it


def test_cancelling_a_job_waiting_at_the_gate_frees_it(app):
    r = app.Runner()
    busy = r.submit("t", "busy", _py("import time; time.sleep(1.0)"), lane="gpu1")
    time.sleep(0.15)
    mine = r.submit("mine", "mine", _py("pass"), lane="gpu2", exclusive=True)
    time.sleep(0.1)
    assert r.cancel(mine.id)
    _wait(r, [busy, mine])
    assert mine.status == "cancelled" and mine.started is None
    after = r.submit("t", "after", _py("pass"), lane="local")
    _wait(r, [after], timeout=5)                   # the gate is open again
    assert after.status == "done"
