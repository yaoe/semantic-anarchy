#!/usr/bin/env python3
"""Semantic Anarchy — explorer dashboard.

A self-contained FastAPI app that drives the project's *existing* CLI scripts
(mine_distribution / generate / temperature_sweep / sampler_sweep) as
subprocesses and surfaces them in the browser:

* Generate controls — pick backend/model/sampler/temperature/n/seed/... and run.
* Sweep grids       — temperature & sampler contact sheets.
* Mine              — (re)build a backend's conditioning distribution.
* Live job log      — streamed stdout of the running script, with a job history.
* Gallery           — every artifact in ``outputs/`` (generated images, sweep
                      sheets, mined marginal plots), newest first.

One GPU => one job at a time: submissions are queued and run by a single worker
thread. The browser polls for state; no websockets needed.

Run via ``webui/run.sh`` (binds to the Tailscale IP). Nothing here imports
torch/diffusers — the heavy work happens in the subprocesses, which use the
venv python passed in ``SA_PYTHON``.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"
# Folder of "good init images" -- when init injection is on, each generation
# starts img2img from a RANDOM one (entropy injection). Drop images in here.
INIT_DIR = Path(os.path.expanduser(os.environ.get("SA_INIT_DIR", str(REPO / "init_images"))))
INIT_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
FAVS_FILE = OUTPUTS / "favorites.json"   # persisted list of favorited image rel-paths

# rel-path -> distance gauge (or None), read once from each image's json sidecar.
_DIST_CACHE: dict = {}


def _distance_for(rel: str) -> Optional[float]:
    if rel not in _DIST_CACHE:
        val = None
        j = OUTPUTS / Path(rel).with_suffix(".json")
        if j.is_file():
            try:
                val = json.loads(j.read_text()).get("distance")
            except Exception:
                pass
        _DIST_CACHE[rel] = val
    return _DIST_CACHE[rel]


def init_images_count() -> int:
    """Total init images anywhere under INIT_DIR (recursive = the 'any folder' pool)."""
    if INIT_DIR.is_dir():
        return sum(1 for p in INIT_DIR.rglob("*") if p.suffix.lower() in INIT_EXTS)
    return 0


def init_folders() -> list:
    """Subfolders of INIT_DIR that contain images, with per-folder counts."""
    out = []
    if INIT_DIR.is_dir():
        # images sitting directly in the root (not in a subfolder)
        root_n = sum(1 for p in INIT_DIR.iterdir()
                     if p.is_file() and p.suffix.lower() in INIT_EXTS)
        if root_n:
            out.append({"name": "(root)", "path": "", "count": root_n})
        for d in sorted(p for p in INIT_DIR.iterdir() if p.is_dir()):
            n = sum(1 for p in d.rglob("*") if p.suffix.lower() in INIT_EXTS)
            if n:
                out.append({"name": d.name, "path": d.name, "count": n})
    return out


def resolve_init_dir(folder: Optional[str]) -> Path:
    """Map a requested folder name to a safe path under INIT_DIR.

    ``None``/``""``/``"__any__"`` -> INIT_DIR root (recursive = pick from any
    folder). Otherwise the named subfolder, sandboxed against traversal.
    """
    base = INIT_DIR.resolve()
    if not folder or folder in ("__any__", "(root)"):
        return base
    cand = (INIT_DIR / folder).resolve()
    if base not in cand.parents and cand != base:
        raise HTTPException(400, f"init folder outside init_images/: {folder}")
    if not cand.is_dir():
        raise HTTPException(400, f"init folder not found: {folder}")
    return cand
_favs_lock = threading.Lock()


def load_favs() -> set:
    try:
        return set(json.loads(FAVS_FILE.read_text()))
    except Exception:
        return set()


def save_favs(favs: set) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FAVS_FILE.write_text(json.dumps(sorted(favs), indent=0))
# Interpreter used to run the heavy scripts (a venv with torch+diffusers).
PYTHON = os.environ.get("SA_PYTHON") or os.sys.executable
# Flow-model jobs (flux2 / krea2) run in the dedicated flux venv (diffusers>=0.39).
FLUX_PYTHON = os.environ.get("SA_FLUX_PYTHON", str(REPO / ".venv-flux" / "bin" / "python"))


def python_for(backend: str) -> str:
    return FLUX_PYTHON if backend in ("flux2", "krea2") else PYTHON
# SD1.5 single-file checkpoint (downloaded per the README).
SD15_CKPT = os.path.expanduser(
    os.environ.get("SA_SD15_CKPT", "~/models/v1-5-pruned-emaonly.safetensors")
)
# SD 2.1 (768, v-pred) single-file checkpoint (mirror), also via --ckpt.
SD2_CKPT = os.path.expanduser(
    os.environ.get("SA_SD2_CKPT", "~/models/v2-1_768-ema-pruned.safetensors")
)
SINGLE_FILE_CKPT = {"sd15": SD15_CKPT, "sd2": SD2_CKPT}
# SDXL repos already cached in ~/.cache/huggingface (HF ids). base-1.0 is the
# DEFAULT — it takes CFG and actually expresses the sampled drift. turbo is a
# 1-step/no-CFG distilled model that mode-collapses to generic output, so it's
# a poor fit for this method and only offered as a fast preview.
SDXL_MODELS = {
    "sdxl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
}
SDXL_DEFAULT_MODEL = "sdxl-base-1.0"
# Per-model generation defaults applied when the user leaves steps/guidance
# blank. base needs real CFG + steps; turbo is 1-step/no-CFG by construction.
SDXL_MODEL_DEFAULTS = {
    "sdxl-base-1.0": {"steps": 30, "guidance": 7.0},
    "sdxl-turbo": {"steps": 1, "guidance": 0.0},
}

# Allow-lists so user input can never become an arbitrary command.
BACKENDS = {"sd15", "sd2", "sdxl", "flux2", "krea2"}
SAMPLERS = {"diagonal", "pca", "blend", "hybrid"}
NEG_MODES = {"mean", "empty", "zeros"}
SCHEDULERS = {"default", "ddim", "euler", "euler_a", "dpm"}

# Gallery buckets keyed by filename prefix.
GALLERY_BUCKETS = [
    ("generated", "anarchy_"),
    ("temperature", "temperature_sweep"),
    ("sampler", "sampler_sweep"),
    ("marginals", "mined_marginals"),
]


# --------------------------------------------------------------------------- #
# Job model + single-worker queue
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: int
    action: str
    label: str
    argv: list[str]
    status: str = "queued"           # queued | running | done | error | cancelled
    rc: Optional[int] = None
    started: Optional[float] = None
    ended: Optional[float] = None
    log: list[str] = field(default_factory=list)
    _proc: Optional[subprocess.Popen] = None


class Runner:
    """Serialises jobs onto one worker thread (one GPU)."""

    def __init__(self) -> None:
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._jobs: dict[int, Job] = {}
        self._seq = 0
        self._lock = threading.Lock()
        self._current: Optional[Job] = None
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, action: str, label: str, argv: list[str]) -> Job:
        with self._lock:
            self._seq += 1
            job = Job(id=self._seq, action=action, label=label, argv=argv)
            self._jobs[job.id] = job
        self._q.put(job)
        return job

    def get(self, job_id: int) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: int) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status == "queued":
            job.status = "cancelled"
            return True
        if job.status == "running" and job._proc:
            job._proc.terminate()
            return True
        return False

    def snapshot(self) -> dict:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)
        return {
            "running": self._current.id if self._current else None,
            "jobs": [
                {
                    "id": j.id,
                    "action": j.action,
                    "label": j.label,
                    "status": j.status,
                    "rc": j.rc,
                    "started": j.started,
                    "ended": j.ended,
                    "lines": len(j.log),
                    "cmd": " ".join(shlex.quote(a) for a in j.argv),
                }
                for j in jobs
            ],
        }

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            if job.status == "cancelled":
                continue
            self._current = job
            job.status = "running"
            job.started = time.time()
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            try:
                job._proc = subprocess.Popen(
                    job.argv, cwd=str(REPO), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in job._proc.stdout:          # live stream
                    job.log.append(line.rstrip("\n"))
                    if len(job.log) > 5000:
                        del job.log[:1000]
                job.rc = job._proc.wait()
                job.status = "done" if job.rc == 0 else "error"
            except Exception as exc:                    # noqa: BLE001
                job.log.append(f"[webui] launch failed: {exc!r}")
                job.status = "error"
                job.rc = -1
            finally:
                job.ended = time.time()
                job._proc = None
                self._current = None


RUNNER = Runner()


# --------------------------------------------------------------------------- #
# Build argv for each action from validated params
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    action: str                       # generate | temp_sweep | sampler_sweep | mine
    backend: str = "sd15"
    model: Optional[str] = None       # sdxl model key (SDXL_MODELS)
    sampler: str = "diagonal"
    temperature: Optional[float] = None
    n: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    coherence: Optional[float] = None
    components: Optional[int] = None
    truncation: Optional[float] = None
    neg_mode: Optional[str] = None
    temps: Optional[str] = None       # sweep: "0.5,1.0,1.5"
    seeds: Optional[str] = None       # sweep: "0,1,2"
    scheduler: Optional[str] = None   # default | ddim | euler | euler_a | dpm
    width: Optional[int] = None
    height: Optional[int] = None
    comp_lo: Optional[int] = None     # pca: first axis (skip dominant/standard ones)
    equalize: bool = False            # pca: express selected axes at equal strength
    dist: str = "base"                # base corpus | evolved ★ branch
    target_distance: Optional[float] = None  # shell sampling: pin the distance gauge
    min_distance: Optional[float] = None      # floor: never below this distance
    init: bool = False                # start from a random good init image
    init_mode: str = "img2img"        # img2img (latent) | embedding (IP-Adapter)
    init_strength: float = 0.7        # img2img denoise from the init (0.6-0.8)
    ip_scale: float = 0.7             # IP-Adapter image-embedding strength
    init_folder: Optional[str] = None # which subfolder to pick from ("" / "__any__" = any)


def _model_flags(req: RunRequest) -> list[str]:
    """--ckpt for single-file backends (sd15/sd2), --model for sdxl (cached HF id)."""
    if req.backend in SINGLE_FILE_CKPT:
        ckpt = SINGLE_FILE_CKPT[req.backend]
        p = Path(ckpt)
        if not p.exists():
            raise HTTPException(400, f"{req.backend} checkpoint not found: {ckpt}")
        # A diffusers *folder* loads via from_pretrained (--model); a single-file
        # .ckpt/.safetensors loads via from_single_file (--ckpt).
        return ["--model", ckpt] if p.is_dir() else ["--ckpt", ckpt]
    if req.backend == "flux2":
        return ["--model", os.environ.get("SA_FLUX2_MODEL",
                                          "black-forest-labs/FLUX.2-klein-4B")]
    if req.backend == "krea2":
        return ["--model", os.environ.get("SA_KREA2_MODEL", "krea/Krea-2-Raw")]
    key = req.model or SDXL_DEFAULT_MODEL
    if key not in SDXL_MODELS:
        raise HTTPException(400, f"unknown sdxl model {key!r}")
    return ["--model", SDXL_MODELS[key]]


def _common_sampler_flags(req: RunRequest) -> list[str]:
    a: list[str] = ["--sampler", req.sampler]
    if req.coherence is not None:
        a += ["--coherence", str(req.coherence)]
    if req.components is not None:
        a += ["--components", str(req.components)]
    if req.truncation is not None:
        a += ["--truncation", str(req.truncation)]
    if req.comp_lo:
        a += ["--comp-lo", str(int(req.comp_lo))]
    if req.equalize:
        a += ["--equalize"]
    return a


def _gen_flags(req: RunRequest) -> list[str]:
    # For SDXL, never let steps/guidance fall through to the script's turbo
    # family defaults (1 step / no CFG) — backfill from the chosen model so base
    # always runs with real CFG. sd15 keeps the script's own defaults.
    steps, guidance = req.steps, req.guidance
    if req.backend == "sdxl":
        d = SDXL_MODEL_DEFAULTS.get(req.model or SDXL_DEFAULT_MODEL, {})
        if steps is None:
            steps = d.get("steps")
        if guidance is None:
            guidance = d.get("guidance")
    a: list[str] = []
    if steps is not None:
        a += ["--steps", str(steps)]
    if guidance is not None:
        a += ["--guidance", str(guidance)]
    if req.neg_mode:
        a += ["--neg-mode", req.neg_mode]
    if req.scheduler and req.scheduler != "default":
        a += ["--scheduler", req.scheduler]
    return a


def build_argv(req: RunRequest) -> tuple[str, list[str]]:
    if req.backend not in BACKENDS:
        raise HTTPException(400, f"bad backend {req.backend!r}")
    if req.sampler not in SAMPLERS:
        raise HTTPException(400, f"bad sampler {req.sampler!r}")
    if req.neg_mode and req.neg_mode not in NEG_MODES:
        raise HTTPException(400, f"bad neg_mode {req.neg_mode!r}")
    if req.scheduler and req.scheduler not in SCHEDULERS:
        raise HTTPException(400, f"bad scheduler {req.scheduler!r}")

    base = [python_for(req.backend), "-u"]
    model = _model_flags(req)
    # Which distribution to sample: the base corpus, or the evolved ★ branch
    # (written by scripts/evolve_favorites.py, backend-namespaced).
    if req.dist == "evolved":
        suffix = "" if req.backend == "sd15" else f"_{req.backend}"
        tensor = "__prompt_embeds" if req.backend == "sdxl" else ""
        if not (OUTPUTS / f"dist_evolved{suffix}{tensor}.npz").is_file():
            raise HTTPException(400, f"no evolved branch for {req.backend} yet — run 🧪 Evolve ★ first")
        dist_base = "outputs/dist_evolved"
    else:
        dist_base = "outputs/dist"
    common = ["--backend", req.backend, *model, "--dist", dist_base]

    if req.action == "mine":
        argv = base + ["scripts/mine_distribution.py", "--backend", req.backend,
                       *model, "--prompts", "prompts_1000.txt", "--out", "outputs/dist"]
        comps = req.components
        if comps is None and req.backend in ("flux2", "krea2"):
            comps = 256   # Qwen embeddings are huge; full-rank PCA would be GBs
        if comps is not None:
            argv += ["--components", str(comps)]
        label = f"mine · {req.backend}"
        return label, argv

    if req.action == "generate":
        argv = base + ["scripts/generate.py", *common, *_common_sampler_flags(req),
                       *_gen_flags(req)]
        if req.n is not None:
            argv += ["--n", str(req.n)]
        if req.temperature is not None:
            argv += ["--temperature", str(req.temperature)]
        if req.target_distance is not None:
            argv += ["--target-distance", str(req.target_distance)]
        if req.min_distance is not None:
            argv += ["--min-distance", str(req.min_distance)]
        if req.seed is not None:
            argv += ["--seed", str(req.seed)]
        if req.width:
            argv += ["--width", str(int(req.width))]
        if req.height:
            argv += ["--height", str(int(req.height))]
        # init-image injection (only if enabled AND images are present)
        if req.init and init_images_count() > 0:
            if req.init_mode not in ("img2img", "embedding"):
                raise HTTPException(400, f"bad init_mode {req.init_mode!r}")
            d = resolve_init_dir(req.init_folder)
            argv += ["--init-dir", str(d), "--init-mode", req.init_mode]
            if req.init_mode == "embedding":
                argv += ["--ip-scale", str(req.ip_scale)]
            else:
                argv += ["--init-strength", str(req.init_strength)]
        label = f"generate · {req.backend} · {req.sampler} · T={req.temperature or 1.0} · n={req.n or 8}"
        return label, argv

    if req.action == "temp_sweep":
        argv = base + ["scripts/temperature_sweep.py", *common,
                       *_common_sampler_flags(req), *_gen_flags(req)]
        if req.temps:
            argv += ["--temps", _clean_csv(req.temps)]
        if req.seeds:
            argv += ["--seeds", _clean_csv(req.seeds)]
        label = f"temp-sweep · {req.backend} · {req.sampler}"
        return label, argv

    if req.action == "sampler_sweep":
        argv = base + ["scripts/sampler_sweep.py", *common, *_gen_flags(req)]
        if req.temperature is not None:
            argv += ["--temperature", str(req.temperature)]
        if req.coherence is not None:
            argv += ["--coherence", str(req.coherence)]
        if req.seeds:
            argv += ["--seeds", _clean_csv(req.seeds)]
        label = f"sampler-sweep · {req.backend}"
        return label, argv

    raise HTTPException(400, f"unknown action {req.action!r}")


def _clean_csv(s: str) -> str:
    """Keep only number-ish comma lists (defends the sweep flags)."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        try:
            float(p)
        except ValueError:
            raise HTTPException(400, f"bad number in list: {p!r}")
    return ",".join(parts)


# --------------------------------------------------------------------------- #
# App + routes
# --------------------------------------------------------------------------- #
app = FastAPI(title="Semantic Anarchy Explorer")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/api/run")
def api_run(req: RunRequest) -> JSONResponse:
    label, argv = build_argv(req)
    job = RUNNER.submit(req.action, label, argv)
    return JSONResponse({"job_id": job.id, "label": label})


class RefineRequest(BaseModel):
    src: str                          # filename (or outputs-relative path) of a PNG
    scale: float = 1.5
    steps: Optional[int] = None
    strength: float = 0.35
    scheduler: Optional[str] = None   # default ddim in refine.py when unset
    tiled: bool = True                # tiled native-res detail pass (Ultimate-SD-Upscale style)
    overlap: int = 128
    engine: str = "flux"              # flux (klein reference-regen) | sd (tiled img2img)
    prompt: Optional[str] = None      # flux engine: override the upscale instruction


@app.post("/api/refine")
def api_refine(req: RefineRequest) -> JSONResponse:
    # Resolve + sandbox the source to a PNG under outputs/.
    base = OUTPUTS.resolve()
    src = (base / Path(req.src)).resolve()
    # Fall back to outputs/generated/ when given a bare filename (gallery cards
    # used to send just the name, and that's where generated images live).
    if not src.is_file():
        alt = (base / "generated" / Path(req.src).name).resolve()
        if alt.is_file():
            src = alt
    if base not in src.parents or src.suffix.lower() != ".png" or not src.is_file():
        raise HTTPException(404, f"source not found under outputs/: {req.src}")
    if not (0.0 < req.scale <= 3.0):
        raise HTTPException(400, "scale must be in (0, 3]")
    if not (0.0 < req.strength <= 1.0):
        raise HTTPException(400, "strength must be in (0, 1]")

    if "anarchy_sdxl" in src.name:
        backend = "sdxl"
    elif "anarchy_sd2" in src.name:
        backend = "sd2"
    else:
        backend = "sd15"
    model = _model_flags(RunRequest(action="refine", backend=backend))
    if req.engine == "flux":
        argv = [FLUX_PYTHON, "-u", "scripts/refine_flux.py", "--src", str(src),
                "--scale", str(req.scale)]
        if req.steps is not None:
            argv += ["--steps", str(int(req.steps))]
        if req.prompt:
            argv += ["--prompt", req.prompt]
        label = f"refine · flux-klein · {src.name} · x{req.scale}"
        job = RUNNER.submit("refine", label, argv)
        return JSONResponse({"job_id": job.id, "label": label})
    if req.scheduler and req.scheduler not in SCHEDULERS:
        raise HTTPException(400, f"bad scheduler {req.scheduler!r}")
    argv = [PYTHON, "-u", "scripts/refine.py", "--backend", backend, *model,
            "--src", str(src), "--scale", str(req.scale), "--strength", str(req.strength)]
    if req.steps is not None:
        argv += ["--steps", str(int(req.steps))]
    if req.scheduler:
        argv += ["--scheduler", req.scheduler]
    if req.tiled:
        argv += ["--tiled", "--overlap", str(int(req.overlap))]
    mode = "tiled" if req.tiled else "single"
    label = f"refine · {backend} · {src.name} · x{req.scale} · d{req.strength} · {mode}"
    job = RUNNER.submit("refine", label, argv)
    return JSONResponse({"job_id": job.id, "label": label})


class ExploreRequest(BaseModel):
    src: str                          # anchor image (outputs-relative or bare name)
    mode: str = "neighborhood"        # neighborhood | breed | walk
    b: Optional[str] = None           # second parent (breed)
    radius: float = 0.3
    mutate: float = 0.15
    direction: str = "outward"        # walk: outward | random | axis
    step: float = 0.15                # walk: per-frame step
    axis: Optional[int] = None        # walk: which principal axis
    n: int = 6
    steps: Optional[int] = None
    guidance: Optional[float] = None


def _resolve_output_png(rel: str) -> Path:
    """Resolve an outputs-relative (or bare) PNG name, sandboxed to outputs/."""
    base = OUTPUTS.resolve()
    p = (base / Path(rel)).resolve()
    if not p.is_file():
        alt = (base / "generated" / Path(rel).name).resolve()
        if alt.is_file():
            p = alt
    if base not in p.parents or p.suffix.lower() != ".png" or not p.is_file():
        raise HTTPException(404, f"image not found under outputs/: {rel}")
    return p


@app.post("/api/explore")
def api_explore(req: ExploreRequest) -> JSONResponse:
    if req.mode not in ("neighborhood", "breed", "walk"):
        raise HTTPException(400, f"bad mode {req.mode!r}")
    if req.direction not in ("outward", "random", "axis"):
        raise HTTPException(400, f"bad direction {req.direction!r}")
    src = _resolve_output_png(req.src)
    if not src.with_suffix(".npz").is_file():
        raise HTTPException(400, f"{src.name} has no conditioning sidecar (too old to explore)")
    backend = "sd15"
    for b in ("flux2", "krea2", "sdxl", "sd2"):
        if f"anarchy_{b}_" in src.name:
            backend = b
            break
    model = _model_flags(RunRequest(action="explore", backend=backend))
    argv = [python_for(backend), "-u", "scripts/explore.py", "--backend", backend, *model,
            "--mode", req.mode, "--src", str(src),
            "--n", str(int(req.n)), "--scheduler", "ddim"]
    if req.mode == "breed":
        b = _resolve_output_png(req.b or "")
        if not b.with_suffix(".npz").is_file():
            raise HTTPException(400, f"{b.name} has no conditioning sidecar")
        if backend not in b.name:
            raise HTTPException(400, "breed parents must be from the same backend")
        argv += ["--b", str(b), "--mutate", str(req.mutate)]
        label = f"breed · {backend} · {src.name} × {b.name}"
    elif req.mode == "walk":
        argv += ["--direction", req.direction, "--step", str(req.step)]
        if req.axis is not None:
            argv += ["--axis", str(int(req.axis))]
        label = f"walk · {backend} · {src.name} · {req.direction} ×{req.n}"
    else:
        argv += ["--radius", str(req.radius)]
        label = f"explore · {backend} · {src.name} · r={req.radius}"
    # Backfill steps/guidance so SDXL never falls through to the turbo family
    # defaults (1 step / no CFG) -- children must render with real CFG.
    d = SDXL_MODEL_DEFAULTS.get(SDXL_DEFAULT_MODEL, {}) if backend == "sdxl" else {}
    steps = req.steps if req.steps is not None else d.get("steps")
    guidance = req.guidance if req.guidance is not None else d.get("guidance")
    if steps is not None:
        argv += ["--steps", str(int(steps))]
    if guidance is not None:
        argv += ["--guidance", str(guidance)]
    if backend == "sdxl":
        argv += ["--neg-mode", "mean"]
    job = RUNNER.submit("explore", label, argv)
    return JSONResponse({"job_id": job.id, "label": label})


class EvolveRequest(BaseModel):
    backend: Optional[str] = None     # None = backend with most starred images
    n: int = 8
    temperature: float = 1.0
    base_blend: float = 0.25


def _most_starred_backend() -> str:
    favs = load_favs()
    counts = {b: sum(1 for r in favs if f"anarchy_{b}_" in Path(r).name)
              for b in BACKENDS}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise HTTPException(400, "no starred images to evolve from")
    return best


@app.post("/api/evolve")
def api_evolve(req: EvolveRequest) -> JSONResponse:
    backend = req.backend or _most_starred_backend()
    if backend not in BACKENDS:
        raise HTTPException(400, f"bad backend {backend!r}")
    model = _model_flags(RunRequest(action="evolve", backend=backend))
    argv = [python_for(backend), "-u", "scripts/evolve_favorites.py", "--backend", backend, *model,
            "--n", str(int(req.n)), "--temperature", str(req.temperature),
            "--base-blend", str(req.base_blend), "--scheduler", "ddim"]
    d = SDXL_MODEL_DEFAULTS.get(SDXL_DEFAULT_MODEL, {}) if backend == "sdxl" else {}
    if d.get("steps") is not None:
        argv += ["--steps", str(d["steps"]), "--guidance", str(d["guidance"]),
                 "--neg-mode", "mean"]
    label = f"evolve★ · {backend} · T={req.temperature}"
    job = RUNNER.submit("evolve", label, argv)
    return JSONResponse({"job_id": job.id, "label": label})


@app.post("/api/resonance")
def api_resonance() -> JSONResponse:
    """Queue the resonance engine: embed new images, novelty + taste model."""
    argv = [PYTHON, "-u", "scripts/resonance.py"]
    job = RUNNER.submit("resonance", "analyze · novelty + resonance", argv)
    return JSONResponse({"job_id": job.id})


@app.get("/api/tasteband")
def api_tasteband() -> JSONResponse:
    """Distance stats of the starred images -- where your keepers live."""
    favs = load_favs()
    ds = []
    for rel in favs:
        j = OUTPUTS / Path(rel).with_suffix(".json")
        if j.is_file():
            try:
                d = json.loads(j.read_text()).get("distance")
                if d is not None:
                    ds.append(float(d))
            except Exception:
                pass
    if not ds:
        return JSONResponse({"count": 0})
    ds.sort()
    return JSONResponse({
        "count": len(ds),
        "mean": round(sum(ds) / len(ds), 2),
        "p25": round(ds[len(ds) // 4], 2),
        "p75": round(ds[(3 * len(ds)) // 4], 2),
    })


@app.post("/api/cancel/{job_id}")
def api_cancel(job_id: int) -> JSONResponse:
    return JSONResponse({"ok": RUNNER.cancel(job_id)})


@app.get("/api/state")
def api_state() -> JSONResponse:
    return JSONResponse(RUNNER.snapshot())


@app.get("/api/log/{job_id}", response_class=PlainTextResponse)
def api_log(job_id: int) -> str:
    job = RUNNER.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return "\n".join(job.log)


@app.get("/api/images")
def api_images() -> JSONResponse:
    favs = load_favs()
    def _load_json(name):
        try:
            return json.loads((OUTPUTS / name).read_text())
        except Exception:
            return {}
    scores = _load_json("scores.json")
    novelty = _load_json("novelty.json")
    resonance = _load_json("resonance.json")
    buckets: dict[str, list] = {k: [] for k, _ in GALLERY_BUCKETS}
    buckets["favorites"] = []
    buckets["top"] = []
    buckets["frontier"] = []
    by_rel: dict[str, dict] = {}
    if OUTPUTS.is_dir():
        # generate.py writes to outputs/generated/; sweeps + marginals to
        # outputs/ root. rglob covers both.
        for p in OUTPUTS.rglob("*.png"):
            name = p.name
            rel = p.relative_to(OUTPUTS).as_posix()
            for key, prefix in GALLERY_BUCKETS:
                if name.startswith(prefix):
                    st = p.stat()
                    item = {
                        "name": name,
                        "rel": rel,
                        "url": f"/img?path={rel}",
                        "mtime": st.st_mtime,
                        "size": st.st_size,
                        "fav": rel in favs,
                        "score": scores.get(rel),
                        "dist": _distance_for(rel),
                        "nov": novelty.get(rel),
                        "res": resonance.get(rel),
                    }
                    buckets[key].append(item)
                    by_rel[rel] = item
                    break
    # The favorites bucket pulls the matching items (skipping any deleted files).
    buckets["favorites"] = [by_rel[r] for r in favs if r in by_rel]
    # Top rated: every scored generated image, best first.
    scored = [it for it in buckets["generated"] if it["score"] is not None]
    buckets["top"] = sorted(scored, key=lambda d: d["score"], reverse=True)[:80]
    # 🎯 Frontier: Pareto front of novelty x resonance (fall back to the generic
    # aesthetic score until the taste model exists). "Nothing else in the
    # gallery is both newer AND more you."
    cand = [it for it in buckets["generated"]
            if it["nov"] is not None and (it["res"] or it["score"]) is not None]
    cand.sort(key=lambda d: d["nov"], reverse=True)
    front, best = [], -1e9
    for it in cand:
        r = it["res"] if it["res"] is not None else it["score"]
        if r > best:
            front.append(it)
            best = r
    front.sort(key=lambda d: (d["res"] if d["res"] is not None else d["score"]),
               reverse=True)
    buckets["frontier"] = front[:150]
    for key in buckets:
        if key not in ("top", "frontier"):
            buckets[key].sort(key=lambda d: d["mtime"], reverse=True)
    return JSONResponse(buckets)


@app.post("/api/score")
def api_score() -> JSONResponse:
    """Queue an aesthetic-scoring pass over all generated images."""
    argv = [PYTHON, "-u", "scripts/score_images.py"]
    job = RUNNER.submit("score", "score · aesthetic (all images)", argv)
    return JSONResponse({"job_id": job.id})


class FavRequest(BaseModel):
    rel: str
    on: bool = True


@app.post("/api/favorite")
def api_favorite(req: FavRequest) -> JSONResponse:
    # Only allow favoriting PNGs that actually live under outputs/.
    base = OUTPUTS.resolve()
    target = (base / req.rel).resolve()
    if base not in target.parents or target.suffix.lower() != ".png":
        raise HTTPException(404, "not an outputs png")
    rel = target.relative_to(base).as_posix()
    with _favs_lock:
        favs = load_favs()
        favs.add(rel) if req.on else favs.discard(rel)
        save_favs(favs)
    return JSONResponse({"ok": True, "fav": req.on, "count": len(favs)})


@app.get("/api/meta")
def api_meta(path: str) -> JSONResponse:
    """Return the param sidecar (``<stem>.json``) for an image, or {}."""
    base = OUTPUTS.resolve()
    target = (base / path).resolve()
    if base not in target.parents:
        raise HTTPException(404, "not found")
    j = target.with_suffix(".json")
    if j.is_file():
        try:
            return JSONResponse(json.loads(j.read_text()))
        except Exception:
            pass
    return JSONResponse({})


@app.get("/img")
def api_img(path: str) -> FileResponse:
    # Serve PNGs anywhere under outputs/ (incl. outputs/generated/), guarding
    # against path traversal by confirming the resolved file stays inside it.
    base = OUTPUTS.resolve()
    target = (base / path).resolve()
    if base not in target.parents or target.suffix.lower() != ".png" or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(target)


@app.get("/api/config")
def api_config() -> JSONResponse:
    return JSONResponse({
        "python": PYTHON,
        "sd15_ckpt": SD15_CKPT,
        "sd15_ckpt_exists": Path(SD15_CKPT).exists(),
        "sd2_ckpt": SD2_CKPT,
        "sd2_ckpt_exists": Path(SD2_CKPT).exists(),
        "sdxl_models": SDXL_MODELS,
        "init_dir": str(INIT_DIR),
        "init_count": init_images_count(),
        "init_folders": init_folders(),
        "repo": str(REPO),
    })


# --------------------------------------------------------------------------- #
# Frontend (single inline page; vanilla JS, polls /api/state + /api/images)
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Anarchy — Explorer</title>
<style>
  :root { --bg:#0d0e12; --panel:#16181f; --panel2:#1d2029; --line:#2a2e3a;
          --ink:#e7e9ee; --dim:#9aa0ad; --accent:#e0533d; --ok:#4caf50;
          --run:#e0a13d; --err:#e0533d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:14px; }
  header h1 { font-size:17px; margin:0; letter-spacing:.3px; }
  header .sub { color:var(--dim); font-size:12px; }
  .layout { display:grid; grid-template-columns:320px 1fr; gap:0; min-height:calc(100vh - 52px); }
  .side { border-right:1px solid var(--line); padding:16px; overflow-y:auto;
          max-height:calc(100vh - 52px); position:sticky; top:0; }
  .main { padding:16px 20px; overflow-y:auto; max-height:calc(100vh - 52px); }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.8px;
       color:var(--dim); margin:18px 0 8px; }
  label { display:block; font-size:12px; color:var(--dim); margin:10px 0 3px; }
  input, select { width:100%; background:var(--panel2); color:var(--ink);
                  border:1px solid var(--line); border-radius:7px; padding:7px 9px; font:inherit; }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .seg { display:flex; gap:6px; flex-wrap:wrap; }
  .seg button { flex:1; }
  button { background:var(--panel2); color:var(--ink); border:1px solid var(--line);
           border-radius:7px; padding:8px 10px; font:inherit; cursor:pointer; }
  button:hover { border-color:#3a4150; }
  button.sel { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.run { background:var(--accent); border-color:var(--accent); color:#fff;
               width:100%; padding:11px; font-weight:600; margin-top:14px; }
  button.run:disabled { opacity:.5; cursor:not-allowed; }
  .hint { font-size:11px; color:var(--dim); margin-top:4px; }
  .warn { color:var(--err); }
  .jobs { display:flex; flex-direction:column; gap:6px; }
  .job { background:var(--panel); border:1px solid var(--line); border-radius:8px;
         padding:8px 10px; cursor:pointer; }
  .job.active { border-color:var(--accent); }
  .job .t { display:flex; justify-content:space-between; gap:8px; }
  .job .lbl { font-size:12px; }
  .badge { font-size:10px; padding:1px 7px; border-radius:20px; text-transform:uppercase;
           letter-spacing:.5px; white-space:nowrap; }
  .b-running{background:var(--run);color:#000;} .b-done{background:#2e7d32;color:#fff;}
  .b-error{background:var(--err);color:#fff;} .b-queued{background:#3a4150;color:#cfd3dc;}
  .b-cancelled{background:#555;color:#fff;}
  pre.log { background:#07080b; border:1px solid var(--line); border-radius:8px;
            padding:12px; height:240px; overflow:auto; font:12px/1.5 ui-monospace,Menlo,monospace;
            white-space:pre-wrap; color:#cfd3dc; margin:0; }
  .tabs { display:flex; gap:6px; margin:6px 0 12px; flex-wrap:wrap; }
  .tabs button.sel { background:var(--accent); border-color:var(--accent); }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          overflow:hidden; }
  .card img { width:100%; display:block; background:#000; cursor:zoom-in; }
  .card .meta { padding:6px 9px; font-size:11px; color:var(--dim);
                display:flex; justify-content:space-between; gap:6px;
                flex-wrap:wrap; row-gap:4px; }
  .card .meta .nm { flex:1 1 100%; }
  .card .meta .nm { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .sheet img { cursor:zoom-in; }
  .empty { color:var(--dim); padding:30px; text-align:center; }
  #lightbox { position:fixed; inset:0; background:rgba(0,0,0,.92); display:none;
              align-items:center; justify-content:center; z-index:50; padding:20px; }
  #lightbox img { max-width:96vw; max-height:94vh; }
  .statusline { font-size:12px; color:var(--dim); margin-top:6px; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
  .dot.idle{background:#3a4150;} .dot.busy{background:var(--run);}
</style>
</head>
<body>
<header>
  <h1>Semantic&nbsp;Anarchy</h1>
  <span class="sub">promptless explorer · <span id="host"></span></span>
  <span class="statusline" style="margin-left:auto"><span id="busydot" class="dot idle"></span><span id="busytxt">idle</span></span>
</header>

<div class="layout">
  <!-- ------------------------------------------------ controls ---------- -->
  <aside class="side">
    <h2>Action</h2>
    <div class="seg" id="actions">
      <button data-act="generate" class="sel">Generate</button>
      <button data-act="temp_sweep">Temp sweep</button>
      <button data-act="sampler_sweep">Sampler sweep</button>
      <button data-act="mine">Mine</button>
    </div>

    <h2>Model</h2>
    <div class="seg" id="backends">
      <button data-be="sd15" class="sel">SD 1.5</button>
      <button data-be="sd2">SD 2.1</button>
      <button data-be="sdxl">SDXL</button>
      <button data-be="flux2">FLUX.2</button>
      <button data-be="krea2">Krea 2</button>
    </div>
    <div id="sdxlModelWrap" style="display:none">
      <label>SDXL checkpoint</label>
      <select id="model">
        <option value="sdxl-base-1.0">sdxl-base-1.0 (30 steps, CFG) — recommended</option>
        <option value="sdxl-turbo">sdxl-turbo (1 step, no CFG — fast preview, generic)</option>
      </select>
    </div>
    <div id="ckptHint" class="hint"></div>
    <label title="which learned distribution to sample from">Distribution</label>
    <select id="distSel">
      <option value="base">base corpus</option>
      <option value="evolved">evolved ★ branch (from 🧪)</option>
    </select>

    <div class="genonly">
      <h2>Sampler</h2>
      <select id="sampler">
        <option value="diagonal">diagonal — independent coords (raw)</option>
        <option value="pca">pca — on the corpus manifold (T&gt;1 extrapolates)</option>
        <option value="blend">blend — interpolate diagonal/pca</option>
        <option value="hybrid">hybrid — SLERP two real concepts</option>
      </select>

      <div class="row">
        <div><label>Temperature</label><input id="temperature" type="number" step="0.1" placeholder="1.0"></div>
        <div class="coh"><label>Coherence λ</label><input id="coherence" type="number" step="0.1" placeholder="0.5"></div>
      </div>
      <label title="shell sampling: pin every sample's distance gauge to this ring (overrides how far temperature lands)">Target distance</label>
      <input id="target_distance" type="number" step="0.1" placeholder="off">
      <div class="hint" id="bandHint"></div>
      <label>Sampler / scheduler</label>
      <select id="scheduler">
        <option value="default">default scheduler</option>
        <option value="ddim">DDIM (smooth, for high-step renders)</option>
        <option value="euler">Euler</option>
        <option value="euler_a">Euler ancestral</option>
        <option value="dpm">DPM++ 2M</option>
      </select>
    </div>

    <div class="genimg">
      <div class="row">
        <div class="ngen"><label>Images (n)</label><input id="n" type="number" placeholder="8"></div>
        <div class="nseed"><label>Seed</label><input id="seed" type="number" placeholder="random"></div>
      </div>
      <div class="row">
        <div><label>Steps</label><input id="steps" type="number" placeholder="auto (try 50)"></div>
        <div><label>Guidance</label><input id="guidance" type="number" step="0.5" placeholder="auto"></div>
      </div>
      <label>Aspect ratio</label>
      <select id="aspect">
        <option value="">default (square)</option>
        <option value="1:1">square 1:1</option>
        <option value="3:2">landscape 3:2</option>
        <option value="2:3">portrait 2:3</option>
        <option value="4:3">landscape 4:3</option>
        <option value="3:4">portrait 3:4</option>
        <option value="16:9">wide 16:9</option>
        <option value="9:16">tall 9:16</option>
        <option value="21:9">cinematic 21:9</option>
      </select>
      <div class="hint" id="aspectHint"></div>
      <div class="row">
        <div><label>Width</label><input id="width" type="number" placeholder="auto"></div>
        <div><label>Height</label><input id="height" type="number" placeholder="auto"></div>
      </div>
      <div class="row">
        <div><label title="start from a random good init image">Init folder</label>
          <select id="initFolder"><option value="off">off</option></select></div>
        <div><label>Init mode</label>
          <select id="initMode">
            <option value="img2img">img2img (structure)</option>
            <option value="embedding">image-embedding (content)</option>
          </select></div>
      </div>
      <div class="row">
        <div><label>Strength / scale</label><input id="init_strength" type="number" step="0.05" placeholder="0.7"></div>
        <div></div>
      </div>
      <div class="hint" id="initHint"></div>
    </div>

    <div class="sweeponly" style="display:none">
      <div class="tempsweeponly">
        <label>Temperatures (csv)</label><input id="temps" placeholder="0.5,1.0,1.5,2.0">
      </div>
      <label>Seeds (csv)</label><input id="seeds" placeholder="0,1,2">
    </div>

    <details>
      <summary style="cursor:pointer;color:var(--dim);font-size:12px;margin-top:14px">Advanced</summary>
      <div class="row">
        <div><label>Components</label><input id="components" type="number" placeholder="all"></div>
        <div><label>Truncation σ</label><input id="truncation" type="number" step="0.5" placeholder="off"></div>
      </div>
      <div class="row">
        <div><label title="skip the dominant/standard PCA axes; higher = stranger subjects">Comp-lo (weird axis)</label>
          <input id="comp_lo" type="number" placeholder="0"></div>
        <div><label>Equalize</label>
          <select id="equalize"><option value="">off</option><option value="1">on (express minor axes)</option></select></div>
      </div>
      <div class="hint">For non-standard subjects: sampler <b>pca</b>, comp-lo ~40–200, equalize <b>on</b>, temp ~1.1–1.4.</div>
      <label>SDXL neg-mode</label>
      <select id="neg_mode">
        <option value="">auto</option><option value="mean">mean</option>
        <option value="empty">empty</option><option value="zeros">zeros</option>
      </select>
    </details>

    <button class="run" id="runBtn">Run ▶</button>
    <div class="hint" id="runHint"></div>

    <h2>Jobs</h2>
    <div class="jobs" id="jobs"></div>
  </aside>

  <!-- ------------------------------------------------ main -------------- -->
  <main class="main">
    <h2 style="margin-top:0">Job log <span id="logTitle" style="color:var(--ink);text-transform:none;letter-spacing:0"></span></h2>
    <pre class="log" id="log">select or start a job…</pre>

    <h2>Gallery</h2>
    <div class="tabs" id="tabs">
      <button data-tab="generated" class="sel">Generated</button>
      <button data-tab="frontier" title="Pareto front of novelty × your resonance">🎯 Frontier</button>
      <button data-tab="top">🏆 Top rated</button>
      <button data-tab="favorites">★ Favorites</button>
      <button data-tab="temperature">Temp sweeps</button>
      <button data-tab="sampler">Sampler sweeps</button>
      <button data-tab="marginals">Marginals</button>
      <select id="sortBy" title="gallery order" style="margin-left:auto;width:150px">
        <option value="new">newest first</option>
        <option value="score">score ↓</option>
        <option value="dist">distance ↓</option>
        <option value="dist_asc">distance ↑</option>
        <option value="nov">novelty ↓</option>
        <option value="res">resonance ↓</option>
      </select>
      <button id="evolveBtn" title="refit a distribution branch around your ★ favorites and sample it">🧪 Evolve ★</button>
      <button id="analyzeBtn" title="embed new images, recompute novelty + retrain the taste model from your ★">🎯 Analyze</button>
      <button id="scoreBtn" title="aesthetic-score all images">Score all ▶</button>
    </div>
    <div id="refineBar" style="display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap;
         margin:0 0 12px;padding:10px 12px;background:var(--panel);border:1px solid var(--line);border-radius:8px">
      <div><label style="margin-top:0">Upscale ×</label>
        <select id="rfScale" style="width:90px"><option>1.25</option><option selected>1.5</option><option>2.0</option></select></div>
      <div><label style="margin-top:0">Steps</label>
        <input id="rfSteps" type="number" placeholder="40" style="width:80px"></div>
      <div><label style="margin-top:0">Engine</label>
        <select id="rfEngine" style="width:150px">
          <option value="flux" selected>FLUX klein (best)</option>
          <option value="sd">SD img2img</option></select></div>
      <div><label style="margin-top:0">SD mode</label>
        <select id="rfMode" style="width:140px"><option value="tiled" selected>Detail (tiled)</option>
          <option value="single">Standard (1 pass)</option></select></div>
      <div><label style="margin-top:0">Denoise</label>
        <input id="rfStrength" type="number" step="0.05" placeholder="0.45" style="width:90px"></div>
      <div><label style="margin-top:0">Scheduler</label>
        <select id="rfSched" style="width:110px"><option value="ddim" selected>DDIM</option>
          <option value="default">default</option><option value="dpm">DPM++ 2M</option></select></div>
      <div><label style="margin-top:0" title="FLUX engine: instruction given with the reference image">Flux prompt</label>
        <select id="rfPromptSel" style="width:170px">
          <option value="faithful" selected>faithful upscale</option>
          <option value="recreate">creative re-render</option>
          <option value="custom">custom…</option>
        </select></div>
      <div id="rfPromptCustomWrap" style="display:none"><label style="margin-top:0">Custom prompt</label>
        <input id="rfPrompt" placeholder="your instruction" style="width:220px"></div>
      <div style="border-left:1px solid var(--line);align-self:stretch"></div>
      <div><label style="margin-top:0" title="perturbation size for 🧭 Explore (fraction of corpus spread)">Explore radius</label>
        <input id="exRadius" type="number" step="0.05" placeholder="0.3" style="width:90px"></div>
      <div><label style="margin-top:0">Children</label>
        <input id="exN" type="number" placeholder="6" style="width:70px"></div>
      <div><label style="margin-top:0" title="🚶 walk: distance growth per frame (0.15 = +15% further out each step)">Walk step</label>
        <input id="wkStep" type="number" step="0.05" placeholder="0.15" style="width:80px"></div>
      <span class="hint" style="margin:0 0 6px" id="exploreHint"><b>🧭 Explore</b> samples around an image's latent point (hill-climb on taste); <b>🧬 Breed</b>: click on two images to SLERP-cross them.</span>
    </div>
    <div id="gallery"></div>
    <div id="galleryFoot" style="text-align:center;color:var(--dim);font-size:12px;padding:14px 0 30px"></div>
  </main>
</div>

<div id="lightbox">
  <img id="lightimg">
  <div id="lightmeta" style="position:fixed;left:0;right:0;bottom:0;background:rgba(7,8,11,.92);
       border-top:1px solid var(--line);padding:10px 16px;font:12px/1.6 ui-monospace,Menlo,monospace;
       color:#cfd3dc;max-height:38vh;overflow:auto"></div>
</div>

<script>
const $ = s => document.querySelector(s);
let action = "generate", backend = "sd15", tab = "generated";
const PAGE = 200; let shown = PAGE;   // gallery pagination: render this many, +PAGE on scroll
let selectedJob = null, images = {}, lastBusy = null;

document.getElementById("host").textContent = location.host;

// ---- segmented controls ----
function wireSeg(containerSel, attr, cb) {
  document.querySelectorAll(containerSel + " button").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll(containerSel + " button").forEach(x => x.classList.remove("sel"));
      b.classList.add("sel"); cb(b.dataset[attr]);
    };
  });
}
wireSeg("#actions", "act", a => { action = a; syncForm(); });
wireSeg("#backends", "be", b => { backend = b; syncForm(); });
wireSeg("#tabs", "tab", t => { tab = t; shown = PAGE; $(".main").scrollTop = 0; renderGallery(); });

function syncForm() {
  const sweep = action === "temp_sweep" || action === "sampler_sweep";
  const gen = action === "generate";
  $("#sdxlModelWrap").style.display = backend === "sdxl" ? "" : "none";
  document.querySelectorAll(".genonly").forEach(e => e.style.display = (gen || sweep) ? "" : "none");
  document.querySelectorAll(".genimg").forEach(e => e.style.display = gen ? "" : "none");
  document.querySelectorAll(".sweeponly").forEach(e => e.style.display = sweep ? "" : "none");
  document.querySelectorAll(".tempsweeponly").forEach(e => e.style.display = action === "temp_sweep" ? "" : "none");
  document.querySelectorAll(".coh").forEach(e => e.style.display =
    ($("#sampler").value === "blend") ? "" : "none");
  // sampler_sweep ignores the sampler dropdown (it sweeps all three itself)
  $("#sampler").parentElement.querySelector("label")?.remove?.();
  $("#runBtn").textContent = action === "mine" ? "Mine ▶" : "Run ▶";
  syncModelDefaults();
  if (typeof applyAspect === "function") applyAspect();  // recompute dims for the backend
  refreshCkptHint();
  const c = window._cfg || {};
  const ih = $("#initHint");
  if (ih) ih.innerHTML = (c.init_count > 0)
    ? `${c.init_count} init image(s) across ${(c.init_folders||[]).length} folder(s) in <code>init_images/</code>`
    : `<span class="warn">no init images yet — drop folders/images in <code>${c.init_dir || 'init_images/'}</code></span>`;
}
function populateInitFolders() {
  const sel = $("#initFolder"); if (!sel) return;
  const c = window._cfg || {}; const folders = c.init_folders || [];
  const cur = sel.value;
  let html = `<option value="off">off</option>`;
  if (folders.length) html += `<option value="__any__">any folder (${c.init_count})</option>`;
  for (const f of folders) html += `<option value="${f.path}">${f.name} (${f.count})</option>`;
  sel.innerHTML = html;
  if ([...sel.options].some(o => o.value === cur)) sel.value = cur;  // keep selection
}
$("#sampler").onchange = syncForm;

// Aspect-ratio presets -> width/height sized for the current backend's native
// resolution (sd15 512, sd2 768, sdxl 1024), kept ~1:1 area and multiples of 64.
const NATIVE = {sd15: 512, sd2: 768, sdxl: 1024, flux2: 1024, krea2: 1024};
function applyAspect() {
  const v = $("#aspect").value;
  if (!v) { $("#width").value = ""; $("#height").value = ""; $("#aspectHint").textContent = ""; return; }
  const base = NATIVE[backend] || 1024, A = base * base;
  const [rw, rh] = v.split(":").map(Number), r = rw / rh;
  const round64 = x => Math.max(64, Math.round(x / 64) * 64);
  const w = round64(Math.sqrt(A * r)), h = round64(Math.sqrt(A / r));
  $("#width").value = w; $("#height").value = h;
  $("#aspectHint").textContent = `${w}×${h} (${backend} native ${base})`;
}
$("#aspect").onchange = applyAspect;

// SDXL: reflect the chosen model's step/guidance defaults in the placeholders
// so it's obvious base runs with CFG and turbo doesn't.
const MODEL_DEFAULTS = {"sdxl-base-1.0":{steps:30,guidance:7}, "sdxl-turbo":{steps:1,guidance:0}};
function syncModelDefaults() {
  if (backend !== "sdxl") { $("#steps").placeholder="auto"; $("#guidance").placeholder="auto"; return; }
  const d = MODEL_DEFAULTS[$("#model").value] || {};
  $("#steps").placeholder = "default " + d.steps;
  $("#guidance").placeholder = "default " + d.guidance;
}
$("#model").onchange = syncModelDefaults;

function refreshCkptHint() {
  const el = $("#ckptHint"); const c = window._cfg || {};
  if (backend === "sd15") {
    el.innerHTML = c.sd15_ckpt && !c.sd15_ckpt_exists
      ? `<span class="warn">SD1.5 checkpoint missing: ${c.sd15_ckpt}</span>`
      : `single-file ckpt → <code>--ckpt</code> (512²)`;
  } else if (backend === "sd2") {
    el.innerHTML = c.sd2_ckpt && !c.sd2_ckpt_exists
      ? `<span class="warn">SD2.1 checkpoint missing: ${c.sd2_ckpt}</span>`
      : `single-file 768 v-pred ckpt → <code>--ckpt</code> (768²)`;
  } else if (backend === "flux2") {
    el.textContent = "FLUX.2 klein (flow model, Qwen3 encoder) — mine first, then generate";
  } else if (backend === "krea2") {
    el.textContent = "Krea 2 Raw — use sampler diagonal (T 1.0–1.3) or blend λ0.6–0.7; pure pca looks washed (256-comp mine). Slow.";
  } else {
    el.textContent = "cached HF repo → --model (1024²)";
  }
}

// ---- run ----
function numOrNull(id) { const v = $("#"+id).value.trim(); return v === "" ? null : Number(v); }
$("#runBtn").onclick = async () => {
  const body = {
    action, backend,
    model: $("#model").value,
    sampler: $("#sampler").value,
    temperature: numOrNull("temperature"),
    n: numOrNull("n"), seed: numOrNull("seed"),
    steps: numOrNull("steps"), guidance: numOrNull("guidance"),
    coherence: numOrNull("coherence"), components: numOrNull("components"),
    truncation: numOrNull("truncation"),
    neg_mode: $("#neg_mode").value || null,
    dist: $("#distSel").value,
    target_distance: numOrNull("target_distance"),
    temps: $("#temps").value.trim() || null,
    seeds: $("#seeds").value.trim() || null,
    scheduler: $("#scheduler").value || null,
    width: numOrNull("width"), height: numOrNull("height"),
    comp_lo: numOrNull("comp_lo"), equalize: $("#equalize").value === "1",
    init: $("#initFolder").value !== "off",
    init_folder: $("#initFolder").value === "off" ? null : $("#initFolder").value,
    init_mode: $("#initMode").value,
    init_strength: $("#init_strength").value.trim() === "" ? 0.7 : Number($("#init_strength").value),
    ip_scale: $("#init_strength").value.trim() === "" ? 0.7 : Number($("#init_strength").value),
  };
  $("#runHint").textContent = "submitting…";
  try {
    const r = await fetch("/api/run", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) { $("#runHint").innerHTML = `<span class="warn">${j.detail||"error"}</span>`; return; }
    selectedJob = j.job_id;
    $("#runHint").textContent = "queued: " + j.label;
    poll();
  } catch (e) { $("#runHint").innerHTML = `<span class="warn">${e}</span>`; }
};

// ---- polling ----
async function poll() {
  try {
    const st = await (await fetch("/api/state")).json();
    renderJobs(st);
    const busy = st.running !== null;
    $("#busydot").className = "dot " + (busy ? "busy" : "idle");
    $("#busytxt").textContent = busy ? "running job #" + st.running : "idle";
    if (selectedJob == null && st.jobs.length) selectedJob = st.jobs[0].id;
    if (selectedJob != null) loadLog(selectedJob);
    // refresh gallery whenever a job just finished, or periodically
    if (lastBusy === true && !busy) refreshImages();
    lastBusy = busy;
  } catch (e) {}
}
function renderJobs(st) {
  const box = $("#jobs"); box.innerHTML = "";
  st.jobs.slice(0, 12).forEach(j => {
    const d = document.createElement("div");
    d.className = "job" + (j.id === selectedJob ? " active" : "");
    d.innerHTML = `<div class="t"><span class="lbl">#${j.id} ${j.label}</span>
      <span class="badge b-${j.status}">${j.status}</span></div>`;
    d.onclick = () => { selectedJob = j.id; loadLog(j.id); renderJobs(st); };
    box.appendChild(d);
  });
}
async function loadLog(id) {
  try {
    const txt = await (await fetch("/api/log/" + id)).text();
    $("#logTitle").textContent = "#" + id;
    const el = $("#log");
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    el.textContent = txt || "(no output yet)";
    if (atBottom) el.scrollTop = el.scrollHeight;
  } catch (e) {}
}

// ---- gallery ----
async function refreshImages() {
  try { images = await (await fetch("/api/images")).json(); renderGallery(); } catch (e) {}
}
function fmtSize(b) { return b > 1e6 ? (b/1e6).toFixed(1)+"M" : Math.round(b/1e3)+"K"; }
function makeCard(im, sheet) {
  const c = document.createElement("div");
  c.className = "card";
  const src = im.url + "&t=" + Math.floor(im.mtime);
  const isAnarchy = (im.name || "").startsWith("anarchy_");
  const star = `<button class="fav" title="favorite" style="padding:2px 7px;font-size:12px;${im.fav?'color:#ffcf4d;border-color:#ffcf4d':''}">${im.fav?'★':'☆'}</button>`;
  const up = isAnarchy
    ? `<button class="up" title="upscale + more steps" style="padding:2px 8px;font-size:11px;white-space:nowrap">⤴ Upscale</button>` : "";
  const nav = isAnarchy
    ? `<button class="ex" title="explore around this (neighborhood)" style="padding:2px 8px;font-size:11px">🧭</button>` +
      `<button class="wk" title="walk outward toward the periphery from this point" style="padding:2px 8px;font-size:11px">🚶</button>` +
      `<button class="br" title="breed: click this then another image" style="padding:2px 8px;font-size:11px${pendingBreed===im.rel?';background:var(--accent);border-color:var(--accent)':''}">🧬</button>` : "";
  const sc = (im.score != null)
    ? `<span title="aesthetic score" style="color:#7fae7f">${im.score.toFixed(2)}</span>` : "";
  const dd = (im.dist != null)
    ? `<span title="distance from corpus center" style="color:#b48ee0">d${im.dist.toFixed(2)}</span>` : "";
  const nr = (tab === "frontier")
    ? `${im.nov != null ? `<span title="novelty (NN distance in gallery)" style="color:#e0a13d">n${im.nov.toFixed(2)}</span>` : ""}` +
      `${im.res != null ? `<span title="resonance P(you'd star it)" style="color:#e07ab8">r${im.res.toFixed(2)}</span>` : ""}` : "";
  c.innerHTML = `<img loading="lazy" src="${src}">
    <div class="meta"><span class="nm">${im.name}</span>${sc}${dd}${nr}${star}${nav}${up}<span>${fmtSize(im.size)}</span></div>`;
  c.querySelector("img").onclick = () => openLight(src, im.rel, im.score);
  const ub = c.querySelector(".up");
  if (ub) ub.onclick = () => refineImage(im.rel || im.name, ub);
  const ex = c.querySelector(".ex");
  if (ex) ex.onclick = () => exploreImage(im.rel, ex);
  const wk = c.querySelector(".wk");
  if (wk) wk.onclick = () => walkImage(im.rel, wk);
  const br = c.querySelector(".br");
  if (br) br.onclick = () => breedClick(im.rel);
  c.querySelector(".fav").onclick = () => toggleFav(im.rel, !im.fav);
  if (sheet) c.style.marginBottom = "16px";
  return c;
}
function renderGallery() {
  const g = $("#gallery"); const list = images[tab] || [];
  const upscalable = (tab === "generated" || tab === "favorites" || tab === "top" || tab === "frontier");
  $("#refineBar").style.display = upscalable ? "flex" : "none";
  if (!list.length) {
    const msg = tab === "favorites"
      ? "no favorites yet — click the ☆ on any image to add it."
      : tab === "top"
      ? "nothing scored yet — click “Score all ▶” to rank by aesthetic."
      : tab === "frontier"
      ? "no analysis yet — click “🎯 Analyze” to compute novelty + resonance."
      : `no ${tab} artifacts yet — run something.`;
    g.innerHTML = `<div class="empty">${msg}</div>`; $("#galleryFoot").textContent = ""; return;
  }
  const sheet = (tab === "temperature" || tab === "sampler" || tab === "marginals");
  g.className = sheet ? "sheet" : "grid";
  const prev = $(".main").scrollTop;            // preserve scroll across re-render
  if (shown > list.length) shown = Math.max(PAGE, Math.ceil(list.length / PAGE) * PAGE);
  // gallery ordering (newest is the server default)
  let view = list;
  const sb = $("#sortBy") ? $("#sortBy").value : "new";
  if (sb === "score") view = [...list].sort((a,b) => (b.score ?? -1) - (a.score ?? -1));
  else if (sb === "dist") view = [...list].sort((a,b) => (b.dist ?? -1) - (a.dist ?? -1));
  else if (sb === "dist_asc") view = [...list].sort((a,b) => (a.dist ?? 1e9) - (b.dist ?? 1e9));
  else if (sb === "nov") view = [...list].sort((a,b) => (b.nov ?? -1) - (a.nov ?? -1));
  else if (sb === "res") view = [...list].sort((a,b) => (b.res ?? -1) - (a.res ?? -1));
  g.innerHTML = "";
  const count = Math.min(shown, view.length);
  for (let i = 0; i < count; i++) g.appendChild(makeCard(view[i], sheet));
  $("#galleryFoot").textContent = count < list.length
    ? `showing ${count} of ${list.length} — scroll for more`
    : `${list.length} image(s)`;
  $(".main").scrollTop = prev;
}
// Infinite scroll: when near the bottom of the main pane, reveal the next page.
function maybeLoadMore() {
  const m = $(".main"); const list = images[tab] || [];
  if (shown >= list.length) return;
  if (m.scrollTop + m.clientHeight >= m.scrollHeight - 600) {
    shown += PAGE; renderGallery();
  }
}
let pendingBreed = null;   // rel of the first-clicked breed parent
async function exploreImage(rel, btn) {
  const body = {
    src: rel, mode: "neighborhood",
    radius: $("#exRadius").value.trim() === "" ? 0.3 : Number($("#exRadius").value),
    n: $("#exN").value.trim() === "" ? 6 : Number($("#exN").value),
  };
  if (btn) { btn.disabled = true; }
  try {
    const r = await fetch("/api/explore", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) alert(j.detail || "explore failed");
    else { selectedJob = j.job_id; poll(); }
  } catch(e){}
  if (btn) setTimeout(() => { btn.disabled = false; }, 3000);
}
async function walkImage(rel, btn) {
  const body = {
    src: rel, mode: "walk", direction: "outward",
    step: $("#wkStep").value.trim() === "" ? 0.15 : Number($("#wkStep").value),
    n: $("#exN").value.trim() === "" ? 6 : Number($("#exN").value),
  };
  if (btn) btn.disabled = true;
  try {
    const r = await fetch("/api/explore", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) alert(j.detail || "walk failed");
    else { selectedJob = j.job_id; poll(); }
  } catch(e){}
  if (btn) setTimeout(() => { btn.disabled = false; }, 3000);
}
async function breedClick(rel) {
  if (pendingBreed === rel) { pendingBreed = null; renderGallery(); return; }  // cancel
  if (pendingBreed === null) {
    pendingBreed = rel;
    $("#exploreHint").innerHTML = `🧬 parent A = <b>${rel.split("/").pop()}</b> — click 🧬 on a second image (same backend) to breed, or click it again to cancel.`;
    renderGallery(); return;
  }
  const body = {
    src: pendingBreed, b: rel, mode: "breed",
    mutate: 0.15,
    n: $("#exN").value.trim() === "" ? 6 : Number($("#exN").value),
  };
  pendingBreed = null;
  $("#exploreHint").innerHTML = `<b>🧭 Explore</b> samples around an image's latent point (hill-climb on taste); <b>🧬 Breed</b>: click on two images to SLERP-cross them.`;
  try {
    const r = await fetch("/api/explore", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) alert(j.detail || "breed failed");
    else { selectedJob = j.job_id; poll(); }
  } catch(e){}
  renderGallery();
}
async function toggleFav(rel, on) {
  try {
    await fetch("/api/favorite", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({rel, on})});
    await refreshImages();
  } catch (e) {}
}
// The creative preset = the original "Recreate" instruction (re-renders with
// license to reinterpret); faithful = the script's built-in default (null).
const RECREATE_PROMPT = "Recreate this exact image at higher resolution with maximum fine detail and texture fidelity. Keep the composition, colors, style and every element identical.";
function refinePrompt() {
  const sel = $("#rfPromptSel") ? $("#rfPromptSel").value : "faithful";
  if (sel === "recreate") return RECREATE_PROMPT;
  if (sel === "custom") return (($("#rfPrompt") || {}).value || "").trim() || null;
  return null;   // faithful -> script default
}
async function refineImage(name, btn) {
  const body = {
    src: name,
    scale: parseFloat($("#rfScale").value),
    steps: $("#rfSteps").value.trim() === "" ? null : Number($("#rfSteps").value),
    strength: $("#rfStrength").value.trim() === "" ? 0.45 : Number($("#rfStrength").value),
    scheduler: $("#rfSched").value || null,
    tiled: $("#rfMode").value === "tiled",
    engine: $("#rfEngine").value,
    prompt: refinePrompt(),
  };
  if (btn) { btn.disabled = true; btn.textContent = "queued…"; }
  try {
    const r = await fetch("/api/refine", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) { if (btn){ btn.textContent="✗"; } alert(j.detail || "refine failed"); }
    else { selectedJob = j.job_id; if (btn){ btn.textContent="↑ #"+j.job_id; } poll(); }
  } catch (e) { if (btn) btn.textContent = "✗"; }
  setTimeout(() => { if (btn){ btn.disabled=false; btn.textContent="⤴ Upscale"; } }, 4000);
}
const PARAM_ORDER = ["kind","mode","parent","parent_b","distance","anchor_distance","radius","mutate",
  "direction","step","walk_frame","target_distance","elites","base_blend","dist",
  "backend","model","sampler","temperature","coherence",
  "components","comp_lo","equalize","truncation","steps","guidance","scheduler","neg_mode","height","width","init_image","init_mode","init_strength","ip_scale",
  "batch_seed","image_seed","index","refined_from","scale","strength","cond_reused","out_size","seed"];
async function openLight(src, rel, score){
  $("#lightimg").src = src;
  $("#lightbox").style.display = "flex";
  const box = $("#lightmeta"); box.innerHTML = "loading params…";
  const scoreLine = (score != null) ? `<span style="color:#7fae7f;margin-left:10px">aesthetic ${score.toFixed(2)}</span>` : "";
  if (!rel) { box.textContent = ""; return; }
  try {
    const m = await (await fetch("/api/meta?path=" + encodeURIComponent(rel))).json();
    const keys = Object.keys(m);
    if (!keys.length) { box.innerHTML = `<span style="color:#9aa0ad">${rel} — no params recorded (pre-dates param logging)</span>${scoreLine}`; return; }
    const ordered = PARAM_ORDER.filter(k => k in m && m[k] !== null);
    const cli = buildCli(m);
    const LINKED = new Set(["parent", "parent_b", "refined_from"]);
    const render = k => {
      let v = m[k];
      if (LINKED.has(k) && typeof v === "string")
        v = `<a href="#" style="color:#6fb3e0" onclick="openParent('${v}');return false;">${v}</a>`;
      return `<span style="display:inline-block;margin-right:14px"><span style="color:#9aa0ad">${k}</span> ${v}</span>`;
    };
    box.innerHTML = `<div style="color:#e0a13d;margin-bottom:4px">${rel}${scoreLine}</div>` +
      ordered.map(render).join("") +
      (cli ? `<div style="margin-top:6px;color:#7fae7f">${cli}</div>` : "");
  } catch(e){ box.textContent = ""; }
}
function openParent(name){
  const rel = "generated/" + name;
  openLight("/img?path=" + encodeURIComponent(rel), rel, null);
}
function buildCli(m){
  if (m.kind === "generate") {
    let s = `generate.py --backend ${m.backend} --sampler ${m.sampler} --temperature ${m.temperature} --seed ${m.batch_seed} --steps ${m.steps} --guidance ${m.guidance}`;
    if (m.scheduler && m.scheduler!=="default") s += ` --scheduler ${m.scheduler}`;
    if (m.coherence!=null) s += ` --coherence ${m.coherence}`;
    return s;
  }
  if (m.kind === "refine") return `refine.py --src <orig> --scale ${m.scale} --strength ${m.strength} --steps ${m.steps} --guidance ${m.guidance} --scheduler ${m.scheduler}`;
  if (m.kind === "explore") {
    let s = `explore.py --mode ${m.mode} --src ${m.parent} --seed ${m.batch_seed}`;
    if (m.mode === "breed") s += ` --b ${m.parent_b} --mutate ${m.mutate}`;
    else s += ` --radius ${m.radius}`;
    return s;
  }
  return "";
}
$("#lightbox").onclick = (e) => { if (e.target.id !== "lightmeta") $("#lightbox").style.display = "none"; };
$("#lightmeta").onclick = (e) => e.stopPropagation();

$("#rfPromptSel").onchange = () => {
  $("#rfPromptCustomWrap").style.display = $("#rfPromptSel").value === "custom" ? "" : "none";
};
$("#sortBy").onchange = () => { shown = PAGE; renderGallery(); };
$("#evolveBtn").onclick = async () => {
  if (!confirm("Refit a distribution branch around your ★ favorites and sample 8 images from it?")) return;
  $("#evolveBtn").disabled = true;
  try {
    const r = await fetch("/api/evolve", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({n: 8})});
    const j = await r.json();
    if (!r.ok) alert(j.detail || "evolve failed");
    else { selectedJob = j.job_id; poll(); }
  } catch(e){}
  setTimeout(() => { $("#evolveBtn").disabled = false; }, 4000);
};
async function refreshBand() {
  try {
    const b = await (await fetch("/api/tasteband")).json();
    const el = $("#bandHint"); if (!el) return;
    el.textContent = b.count
      ? `your ★ keepers: d≈${b.mean} (band ${b.p25}–${b.p75}, n=${b.count}) — try that as target`
      : "";
  } catch(e){}
}
$("#analyzeBtn").onclick = async () => {
  $("#analyzeBtn").disabled = true; $("#analyzeBtn").textContent = "analyzing…";
  try { const j = await (await fetch("/api/resonance", {method:"POST"})).json(); selectedJob = j.job_id; poll(); } catch(e){}
  setTimeout(()=>{ $("#analyzeBtn").disabled=false; $("#analyzeBtn").textContent="🎯 Analyze"; }, 4000);
};
$("#scoreBtn").onclick = async () => {
  $("#scoreBtn").disabled = true; $("#scoreBtn").textContent = "scoring…";
  try { const j = await (await fetch("/api/score", {method:"POST"})).json(); selectedJob = j.job_id; poll(); } catch(e){}
  setTimeout(()=>{ $("#scoreBtn").disabled=false; $("#scoreBtn").textContent="Score all ▶"; }, 4000);
};

// ---- boot ----
$(".main").addEventListener("scroll", maybeLoadMore, {passive: true});

(async () => {
  try { window._cfg = await (await fetch("/api/config")).json(); } catch(e){}
  populateInitFolders(); syncForm(); refreshImages(); poll(); refreshBand();
  setInterval(refreshBand, 20000);
  setInterval(poll, 1500);
  setInterval(() => { if (lastBusy) refreshImages(); }, 6000);
  // refresh init folders/counts so newly added folders show up without reload
  setInterval(async () => {
    try { window._cfg = await (await fetch("/api/config")).json(); populateInitFolders(); syncForm(); } catch(e){}
  }, 8000);
})();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("SA_HOST", "127.0.0.1")
    port = int(os.environ.get("SA_PORT", "8800"))
    print(f"[webui] python (jobs) = {PYTHON}")
    print(f"[webui] serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
