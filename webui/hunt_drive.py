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
BACKENDS_ROT = [b.strip() for b in os.environ.get("SA_BACKENDS", "sdxl,sd15").split(",") if b.strip()]


def _post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get(path, tries=60, delay=10.0):
    """GET, surviving a dashboard restart underneath us (restart.sh, a reboot).

    An overnight run is worth more than any single round, so a refused
    connection waits the server out instead of killing the hunt.
    """
    for i in range(tries):
        try:
            with urllib.request.urlopen(BASE + path, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if i == 0:
                print(f"[hunt] dashboard unreachable ({e!r}); waiting for it "
                      f"to come back ...", flush=True)
            time.sleep(delay)
    raise SystemExit(f"[hunt] dashboard down for {tries * delay / 60:.0f} min -> exiting")


def wait_for(job_id, poll=4.0):
    while True:
        jobs = _get("/api/state")["jobs"]
        for j in jobs:
            if j["id"] == job_id and j["status"] in ("done", "error", "cancelled"):
                return j["status"]
        # The queue is in-memory: a server restart wipes it and ids start over,
        # so our job will never appear again. Detect that and move on rather
        # than polling forever for a job that no longer exists.
        if jobs and max(j["id"] for j in jobs) < job_id:
            print(f"[hunt]   job {job_id} lost to a server restart; continuing",
                  flush=True)
            return "lost"
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


def taste_bands():
    """Per-backend distance samples from the starred images' sidecars."""
    try:
        favs = json.loads((OUT / "favorites.json").read_text())
    except Exception:
        return {}
    per = {}
    for rel in favs:
        name = Path(rel).name
        for b in ("sdxl", "sd15", "sd2", "flux2", "krea2"):
            if f"anarchy_{b}_" in name:
                j = OUT / Path(rel).with_suffix(".json")
                if j.exists():
                    try:
                        d = json.loads(j.read_text()).get("distance")
                        if d:
                            per.setdefault(b, []).append(float(d))
                    except Exception:
                        pass
                break
    return per


def pick_strategy(backend, bands):
    """The diversity portfolio: the generator's job is VARIANCE (radial, lateral,
    fusion); selection happens later via novelty x resonance. Never pin all
    rounds to one shell -- that's how a whole night collapses onto d=2.72.
    """
    import random
    r = random.random()
    if r < 0.30:   # free-range radial spread
        return {"sampler": "pca",
                "temperature": round(random.uniform(1.4, 2.2), 2)}, "free"
    if r < 0.55:   # taste-band arm: a DISTRIBUTION, per backend, >=10 points, clipped
        ds = bands.get(backend, [])
        if len(ds) >= 10:
            import statistics
            mu, sd = statistics.mean(ds), max(0.35, statistics.pstdev(ds))
            t = round(min(2.4, max(1.3, random.gauss(mu, sd))), 2)
            return {"sampler": "pca", "temperature": 1.7,
                    "target_distance": t}, f"band(d={t})"
        return {"sampler": "pca",
                "temperature": round(random.uniform(1.4, 2.2), 2)}, "free*"
    if r < 0.80:   # lateral: weird minor axes, non-standard subjects
        return {"sampler": "pca", "comp_lo": random.choice([40, 80, 120, 160, 200]),
                "equalize": True,
                "temperature": round(random.uniform(1.05, 1.4), 2)}, "weird"
    # concept fusion
    return {"sampler": "hybrid",
            "temperature": round(random.uniform(0.9, 1.3), 2)}, "hybrid"


def main():
    started = time.time()
    seed = int(os.environ.get("SA_SEED_START", "500000"))
    visited = set(json.loads(VISITED.read_text())) if VISITED.exists() else set()

    def has_evolved(b):
        name = ("dist_evolved.npz" if b == "sd15"
                else "dist_evolved_sdxl__prompt_embeds.npz" if b == "sdxl"
                else f"dist_evolved_{b}.npz")
        return (OUT / name).exists()

    print(f"[hunt] backends={BACKENDS_ROT} n={N_PER}/round, analyze every "
          f"{ANALYZE_EVERY}, explore {EXPLORE_PER_ROUND} frontier imgs/round", flush=True)
    print(f"[hunt] stop anytime: touch {STOP}", flush=True)

    rnd = 0
    while True:
        if STOP.exists():
            print("[hunt] STOP_HUNT -> exiting", flush=True); return
        if (time.time() - started) / 3600 > MAX_HOURS:
            print("[hunt] time budget -> exiting", flush=True); return
        rnd += 1

        backend = BACKENDS_ROT[(rnd - 1) % len(BACKENDS_ROT)]
        knobs, tag = pick_strategy(backend, taste_bands())
        body = {"action": "generate", "backend": backend,
                "model": "sdxl-base-1.0" if backend == "sdxl" else None,
                # base corpus by default: the evolved branch is too narrow for
                # exploration (it's an exploitation tool). Opt in via SA_USE_EVOLVED=1.
                "dist": ("evolved" if os.environ.get("SA_USE_EVOLVED") == "1"
                         and has_evolved(backend) else "base"),
                "n": N_PER, "seed": seed, "steps": 30, "scheduler": "ddim",
                "min_distance": 1.0,   # never dip into the bland corpus core
                **knobs}
        seed += N_PER
        submit_and_wait("/api/run", body,
                        f"round {rnd}: generate {backend} [{tag}]")

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
