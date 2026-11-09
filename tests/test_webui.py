"""Dashboard orchestration: server routing and the multi-lane job runner.

webui/app.py imports fastapi but no torch, so these stay in the torch-free tier.
"""

import importlib.util
import json
import re
import subprocess
import sys
import threading
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


# ---- 🌌 map: sampling that stays local ---------------------------------------
def _explore_argv(app, monkeypatch, tmp_path, ckpt_on_disk=True, **body):
    """argv /api/explore would queue for a real source image with this sidecar."""
    import json as _json
    ckpt = tmp_path / "source_model.safetensors"
    if ckpt_on_disk:
        ckpt.write_bytes(b"")
    src = tmp_path / "anarchy_sd15_1_000.jpg"
    src.write_bytes(b"")
    src.with_suffix(".npz").write_bytes(b"")
    src.with_suffix(".json").write_text(_json.dumps({"model": str(ckpt)}))
    seen = {}

    class Job:
        id = 1

    def submit(kind, label, argv, **kw):
        seen["argv"] = argv
        return Job()

    monkeypatch.setattr(app.RUNNER, "submit", submit)
    monkeypatch.setattr(app, "_resolve_output_image", lambda rel: src)
    monkeypatch.setattr(app, "resolve_dist_base", lambda backend: "outputs/dist")
    monkeypatch.setattr(app, "resolve_server", lambda backend, server: ("local", None))
    monkeypatch.setattr(app, "_model_flags", lambda req: ["--ckpt", "selected.safetensors"])
    app.api_explore(app.ExploreRequest(src=src.name, **body))
    return seen["argv"], str(ckpt)


def test_stay_local_renders_with_the_sources_checkpoint(app, monkeypatch, tmp_path):
    argv, ckpt = _explore_argv(app, monkeypatch, tmp_path, stay_local=True)
    assert "--stay-local" in argv
    assert argv[argv.index("--ckpt") + 1] == ckpt


def test_plain_explore_renders_with_the_sources_checkpoint_too(app, monkeypatch, tmp_path):
    # A child rendered through another checkpoint than its parent stops looking
    # like it, so explore follows the source, not the sidebar's pick.
    argv, ckpt = _explore_argv(app, monkeypatch, tmp_path)
    assert "--stay-local" not in argv
    assert argv[argv.index("--ckpt") + 1] == ckpt
    # ...and forces no render knob of its own: explore.py copies them from src.
    assert "--scheduler" not in argv and "--steps" not in argv


def test_explore_falls_back_to_the_selected_checkpoint_when_the_sources_is_gone(
        app, monkeypatch, tmp_path):
    argv, _ = _explore_argv(app, monkeypatch, tmp_path, ckpt_on_disk=False)
    assert argv[argv.index("--ckpt") + 1] == "selected.safetensors"


def test_map_rejects_an_unknown_focus(app):
    with pytest.raises(app.HTTPException) as e:
        app.api_latent_map("sd15", focus="; rm -rf")
    assert e.value.status_code == 400


def _map_setup(app, monkeypatch, tmp_path, builds):
    """A map endpoint over tmp_path whose build is a stub writing doc number N."""
    monkeypatch.setattr(app, "OUTPUTS", tmp_path)
    monkeypatch.setattr(app, "resolve_dist_base", lambda b: "outputs/dist")
    monkeypatch.setattr(app, "current_dist", lambda b: {"kind": "base", "path": None})
    monkeypatch.setattr(app, "_map_building", set())
    monkeypatch.setattr(app, "_map_failed", {})
    out = tmp_path / "latent_map_sd15_balanced.json"

    def build(cmd, **kw):
        result = builds(len(calls))
        calls.append(cmd)
        if result == 0:
            out.write_text(json.dumps({"backend": "sd15", "built": len(calls), "nodes": [],
                                       "warning": None}))
        return subprocess.CompletedProcess(cmd, result, "", "boom")

    calls: list = []
    monkeypatch.setattr(app.subprocess, "run", build)
    return calls


def _map_get(app):
    return json.loads(app.api_latent_map("sd15", "balanced").body)


def _wait_map_idle(app):
    for _ in range(500):
        with app._map_jobs_lock:
            if not app._map_building:
                return
        time.sleep(0.01)
    raise AssertionError("background map build never finished")


def test_first_map_is_waited_for_then_served_from_cache(app, monkeypatch, tmp_path):
    calls = _map_setup(app, monkeypatch, tmp_path, lambda n: 0)
    assert _map_get(app)["rebuilding"] is False
    assert _map_get(app)["built"] == 1
    assert len(calls) == 1


def test_stale_map_answers_at_once_and_rebuilds_behind_it(app, monkeypatch, tmp_path):
    release = threading.Event()
    def builds(n):
        if n:
            assert release.wait(5)
        return 0
    calls = _map_setup(app, monkeypatch, tmp_path, builds)
    _map_get(app)

    (tmp_path / "new_image.npz").write_bytes(b"")
    stale = _map_get(app)       # returning at all proves it didn't wait for the build
    assert (stale["built"], stale["rebuilding"]) == (1, True)
    assert _map_get(app)["rebuilding"] is True     # and one build, however many polls
    release.set()
    _wait_map_idle(app)

    assert len(calls) == 2
    fresh = _map_get(app)
    assert (fresh["built"], fresh["rebuilding"]) == (2, False)


def test_failed_rebuild_keeps_the_old_map_without_retrying(app, monkeypatch, tmp_path):
    calls = _map_setup(app, monkeypatch, tmp_path, lambda n: 0 if n == 0 else 1)
    _map_get(app)
    (tmp_path / "new_image.npz").write_bytes(b"")
    _map_get(app)
    _wait_map_idle(app)

    doc = _map_get(app)
    assert (doc["built"], doc["rebuilding"]) == (1, False)
    assert "boom" in doc["warning"]
    assert len(calls) == 2      # polling a broken build must not spin it


def test_map_thumbnails_follow_the_atlas_on_disk(app, monkeypatch, tmp_path):
    """Another focus' build can regrid the shared atlas under an older document."""
    monkeypatch.setattr(app, "OUTPUTS", tmp_path)
    out = tmp_path / "latent_map_sd15_balanced.json"
    out.write_text(json.dumps({"atlas": {"cols": 1, "tile": 96},
                               "nodes": [{"rel": "a.jpg", "a": 0}, {"rel": "gone.jpg", "a": 1}]}))
    (tmp_path / "latent_map_sd15.atlas.jpg").write_bytes(b"x")
    (tmp_path / "latent_map_sd15.atlas.json").write_text(
        json.dumps({"rels": ["new.jpg", "a.jpg"], "cols": 2, "tile": 48}))
    doc = app._read_map_doc(out, "sd15")
    assert [n["a"] for n in doc["nodes"]] == [1, -1]
    assert (doc["atlas"]["cols"], doc["atlas"]["tile"]) == (2, 48)


def test_taste_model_does_not_invalidate_the_map(app, monkeypatch, tmp_path):
    """resonance.py rewrites it every run; counting it rebuilt the map forever."""
    monkeypatch.setattr(app, "OUTPUTS", tmp_path)
    (tmp_path / "a.npz").write_bytes(b"")
    before = app._map_signature("x")
    (tmp_path / "taste_model.npz").write_bytes(b"")
    assert app._map_signature("x") == before
    (tmp_path / "b.npz").write_bytes(b"")
    assert app._map_signature("x") != before
