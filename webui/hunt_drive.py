#!/usr/bin/env python3
"""The frontier hunter -- autonomous novelty/resonance-guided exploration.

The full generative-recommender loop, running unattended:

  1. GENERATE a batch on your territory (evolved ★ branch if one exists, else
     the base corpus; shell-pinned to your keeper band when known).
  2. Every few rounds, ANALYZE (CLIP-embed + novelty + retrain taste model).
  3. Read the 🎯 FRONTIER (Pareto front of novelty x resonance) and EXPLORE
     around the best not-yet-visited frontier images (they earn children).
  4. Repeat. You come back, star what resonates, the model sharpens.

Stops on ``outputs/STOP_HUNT`` or after SA_MAX_HOURS. One job at a time through
the dashboard queue, so interactive use interleaves freely.

    SA_BASE=http://100.74.77.125:8800 python webui/hunt_drive.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("SA_BASE", "http://100.74.77.125:8800").rstrip("/")
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"
STOP = OUT / "STOP_HUNT"
VISITED = OUT / "hunt_visited.json"

MAX_HOURS = float(os.environ.get("SA_MAX_HOURS", "8"))
N_PER = int(os.environ.get("SA_N", "4"))
ANALYZE_EVERY = int(os.environ.get("SA_ANALYZE_EVERY", "3"))
EXPLORE_PER_ROUND = int(os.environ.get("SA_EXPLORE_PER_ROUND", "2"))
BACKEND = os.environ.get("SA_BACKEND", "sdxl")


def _post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)


def wait_for(job_id, poll=4.0):
    while True:
        for j in _get("/api/state")["jobs"]:
            if j["id"] == job_id and j["status"] in ("done", "error", "cancelled"):
                return j["status"]
        time.sleep(poll)


def submit_and_wait(path, body, label):
    try:
        jid = _post(path, body)["job_id"]
    except Exception as e:
        print(f"[hunt] {label} submit failed ({e!r}); waiting 15s", flush=True)
        time.sleep(15)
        return None
    print(f"[hunt]   job {jid}: {label}", flush=True)
    return wait_for(jid)


def main():
    started = time.time()
    seed = int(os.environ.get("SA_SEED_START", "500000"))
    visited = set(json.loads(VISITED.read_text())) if VISITED.exists() else set()
    evolved = (OUT / ("dist_evolved.npz" if BACKEND == "sd15"
                      else f"dist_evolved_{BACKEND}.npz" if BACKEND == "sd2"
                      else "dist_evolved_sdxl__prompt_embeds.npz")).exists()
    print(f"[hunt] backend={BACKEND} dist={'evolved★' if evolved else 'base'} "
          f"n={N_PER}/round, analyze every {ANALYZE_EVERY}, "
          f"explore {EXPLORE_PER_ROUND} frontier imgs/round", flush=True)
    print(f"[hunt] stop anytime: touch {STOP}", flush=True)

    rnd = 0
    while True:
        if STOP.exists():
            print("[hunt] STOP_HUNT -> exiting", flush=True); return
        if (time.time() - started) / 3600 > MAX_HOURS:
            print("[hunt] time budget -> exiting", flush=True); return
        rnd += 1

        # keeper band -> shell target (recomputed every round as stars accrue)
        target = None
        try:
            band = _get("/api/tasteband")
            if band.get("count", 0) >= 5:
                target = band["mean"]
        except Exception:
            pass

        body = {"action": "generate", "backend": BACKEND,
                "model": "sdxl-base-1.0" if BACKEND == "sdxl" else None,
                "dist": "evolved" if evolved else "base",
                "sampler": "pca", "temperature": 1.7, "n": N_PER,
                "seed": seed, "steps": 30, "scheduler": "ddim",
                "target_distance": target}
        seed += N_PER
        submit_and_wait("/api/run", body,
                        f"round {rnd}: generate ({'d=' + str(target) if target else 'T=1.7'})")

        if rnd % ANALYZE_EVERY == 0:
            submit_and_wait("/api/resonance", {}, f"round {rnd}: analyze")
            try:
                frontier = _get("/api/images").get("frontier", [])
            except Exception:
                frontier = []
            fresh = [it for it in frontier if it["rel"] not in visited][:EXPLORE_PER_ROUND]
            for it in fresh:
                visited.add(it["rel"])
                VISITED.write_text(json.dumps(sorted(visited)))
                submit_and_wait("/api/explore",
                                {"src": it["rel"], "mode": "neighborhood",
                                 "radius": 0.25, "n": N_PER},
                                f"round {rnd}: explore frontier {it['rel'].split('/')[-1]}")
        time.sleep(1)


if __name__ == "__main__":
    main()
