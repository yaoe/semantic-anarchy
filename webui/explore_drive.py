#!/usr/bin/env python3
"""Long, paced parameter-exploration driver for the Semantic Anarchy dashboard.

Submits a wide matrix of `generate` jobs (backend x sampler x temperature) to the
dashboard's queue ONE AT A TIME -- waiting for each to finish before submitting
the next. That keeps the single GPU safe (no concurrent model loads -> no OOM),
lets interactive jobs interleave between combos, and fills the gallery with
comparable, param-tagged images you can browse when you get back.

Every generated image gets a JSON sidecar (params) so the dashboard lightbox can
show exactly what made it. Stops when ``outputs/STOP_EXPLORE`` appears, after
``ROUNDS`` seed-rounds, or when the wall-clock budget elapses.

Run (backgrounded)::

    SA_BASE=http://100.74.77.125:8800 python webui/explore_drive.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("SA_BASE", "http://100.74.77.125:8800").rstrip("/")
REPO = Path(__file__).resolve().parent.parent
STOP = REPO / "outputs" / "STOP_EXPLORE"

ROUNDS = int(os.environ.get("SA_ROUNDS", "4"))            # seed rounds (seed = round idx)
MAX_HOURS = float(os.environ.get("SA_MAX_HOURS", "10"))
N_PER = int(os.environ.get("SA_N", "4"))                  # images per combo
STEPS = int(os.environ.get("SA_STEPS", "40"))

# backend -> sdxl model key (None for single-file backends)
BACKENDS = [("sd15", None), ("sd2", None), ("sdxl", "sdxl-base-1.0")]
SAMPLERS = ["diagonal", "pca", "blend"]
# Stay strictly below temperature 2 (user preference: >=2 drifts too far off-manifold).
TEMPS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
SCHEDULER = "ddim"


def _post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


def _job_status(job_id):
    for j in _get("/api/state")["jobs"]:
        if j["id"] == job_id:
            return j["status"]
    return None


def wait_for(job_id, poll=3.0):
    """Block until a job reaches a terminal state."""
    while True:
        st = _job_status(job_id)
        if st in ("done", "error", "cancelled"):
            return st
        time.sleep(poll)


def main():
    started = time.time()
    temps = [t for t in TEMPS if t < 2.0]          # hard cap: never >= 2
    combos = [(b, m, s, t) for (b, m) in BACKENDS for s in SAMPLERS for t in temps]
    print(f"[explore] BASE={BASE} rounds={ROUNDS} combos/round={len(combos)} "
          f"n={N_PER} steps={STEPS} sched={SCHEDULER}", flush=True)
    print(f"[explore] stop anytime: `touch {STOP}`", flush=True)

    done = 0
    seed = int(os.environ.get("SA_SEED_START", "1000"))   # fresh seed for EVERY job
    for rnd in range(ROUNDS):
        for (backend, model, sampler, temp) in combos:
            if STOP.exists():
                print("[explore] STOP_EXPLORE found -> exiting", flush=True)
                return
            if (time.time() - started) / 3600.0 > MAX_HOURS:
                print(f"[explore] hit {MAX_HOURS}h budget -> exiting", flush=True)
                return
            body = {
                "action": "generate", "backend": backend, "model": model,
                "sampler": sampler, "temperature": temp, "n": N_PER,
                "seed": seed, "steps": STEPS, "scheduler": SCHEDULER,
            }
            # Advance the seed by the batch size so no two images ever share one.
            seed += N_PER
            try:
                r = _post("/api/run", body)
            except Exception as e:
                print(f"[explore] submit failed ({e!r}); retrying in 10s", flush=True)
                time.sleep(10)
                continue
            jid = r["job_id"]
            done += 1
            print(f"[explore] #{done} round={rnd} -> job {jid}: "
                  f"{backend}/{sampler}/T={temp}/seed={body['seed']}", flush=True)
            status = wait_for(jid)
            if status == "error":
                print(f"[explore]   job {jid} ERRORED (see dashboard log)", flush=True)
            time.sleep(1.0)
    print(f"[explore] done -- {done} combos submitted over "
          f"{(time.time()-started)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
