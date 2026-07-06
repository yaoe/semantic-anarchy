#!/usr/bin/env python3
"""Taste-guided generation driver -- exploit the user's *stars* to make more of
what they like, instead of a blind grid sweep.

Each iteration it RE-READS ``outputs/favorites.json`` + the starred images' param
sidecars, infers the knob distribution (which backend / sampler / temperature the
favorites cluster on), and samples the next job's parameters from that
distribution (with add-one smoothing so it still explores a little). Seeds change
every job. Every few jobs it kicks an aesthetic-scoring pass so the dashboard's
"Top rated" tab stays current. So as you star more, the sampler adapts.

Bias is exploitation-heavy by design; `EXPLORE` controls the exploration floor.

    SA_BASE=http://100.74.77.125:8800 python webui/guided_drive.py
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

BASE = os.environ.get("SA_BASE", "http://100.74.77.125:8800").rstrip("/")
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs"
STOP = OUT / "STOP_EXPLORE"

N_PER = int(os.environ.get("SA_N", "4"))
STEPS = int(os.environ.get("SA_STEPS", "40"))
INIT_ON = os.environ.get("SA_INIT", "1") != "0"          # use good init images if present
INIT_STRENGTH = float(os.environ.get("SA_INIT_STRENGTH", "0.7"))
INIT_MODE = os.environ.get("SA_INIT_MODE", "img2img")      # img2img | embedding
INIT_FOLDER = os.environ.get("SA_INIT_FOLDER", "__any__")  # subfolder name, or any
MAX_HOURS = float(os.environ.get("SA_MAX_HOURS", "10"))
SCORE_EVERY = int(os.environ.get("SA_SCORE_EVERY", "5"))
EXPLORE = float(os.environ.get("SA_EXPLORE", "0.15"))   # prob of a fully-random combo
TEMP_LO, TEMP_HI = 0.8, 1.9                              # honour the <2 cap

BACKENDS = ["sd15", "sd2", "sdxl", "flux2"]   # krea2: manual only (too slow for the loop)
SAMPLERS = ["diagonal", "pca", "blend", "hybrid"]
# Priors encode the "deck regime" that yields legible-but-surreal subjects:
#   SDXL base (strong prior renders sampled conditioning into real scenes; sd15
#   collapses to painterly mush), pca/hybrid (on-manifold -> legible, not washes).
# These ride on top of the star-derived counts so the loop still adapts as the
# user curates, but starts where the good stuff lives.
BACKEND_PRIOR = {"sdxl": 4, "flux2": 4, "sd2": 1, "sd15": 1}
SAMPLER_PRIOR = {"diagonal": 1, "blend": 1, "pca": 4, "hybrid": 3}
TEMP_FLOOR_EXPLOIT = 1.6   # pca needs real deviation to travel outside the hull
TEMP_CENTER = 1.9          # sit at the coherent travel-outside frontier (~1.7-2.2)
SDXL_MODEL = "sdxl-base-1.0"


def _post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.load(r)


def infer():
    """Read stars + their sidecars -> weighted knob distribution."""
    favs = json.loads((OUT / "favorites.json").read_text()) if (OUT / "favorites.json").exists() else []
    be = Counter()
    sa = Counter()
    temps = []
    for rel in favs:
        # backend is always recoverable from the filename prefix
        if "anarchy_sd2" in rel: be["sd2"] += 1
        elif "anarchy_sdxl" in rel: be["sdxl"] += 1
        elif "anarchy_sd15" in rel: be["sd15"] += 1
        j = OUT / (Path(rel).with_suffix(".json"))
        if j.exists():
            try:
                m = json.loads(j.read_text())
                if m.get("sampler"): sa[m["sampler"]] += 1
                if m.get("temperature") is not None: temps.append(float(m["temperature"]))
            except Exception:
                pass
    return be, sa, temps, len(favs)


def weighted(options, counts, prior=None):
    """smoothed weighted choice; `prior` per-option pseudo-counts (default 1)."""
    w = [counts.get(o, 0) + (prior.get(o, 1) if prior else 1) for o in options]
    return random.choices(options, weights=w, k=1)[0]


def sample_combo():
    be, sa, temps, nfav = infer()
    if random.random() < EXPLORE or nfav == 0:
        backend = random.choice(BACKENDS)
        sampler = random.choice(SAMPLERS)
        temp = round(random.uniform(TEMP_LO, TEMP_HI), 2)
        tag = "explore"
    else:
        # Backend is a quality-tier GOAL (deck look = SDXL base), not inferred
        # from the legacy sd15-heavy stars -> drive it by the prior only.
        backend = weighted(BACKENDS, {}, prior=BACKEND_PRIOR)
        sampler = weighted(SAMPLERS, sa, prior=SAMPLER_PRIOR)
        if backend == "krea2" and sampler == "pca":
            sampler = "blend"   # krea's 256-comp pca mine looks washed; blend restores variance
        # Center temperature in the interesting band; let the user's starred temps
        # nudge it, but keep enough deviation for a subject to crystallize.
        mu = TEMP_CENTER
        if len(temps) >= 2:
            import statistics
            mu = max(statistics.mean(temps), TEMP_FLOOR_EXPLOIT)
        temp = round(min(TEMP_HI, max(TEMP_FLOOR_EXPLOIT, random.gauss(mu, 0.18))), 2)
        tag = f"exploit(favs={nfav})"

    # Weird-axis mode: for pca, most of the time SKIP the dominant (=standard)
    # axes and ride the idiosyncratic minor ones (equalized) -> non-standard
    # subjects. Lower temp since equalize already amplifies. Mixed with plain
    # travel-outside so the gallery isn't monotone.
    comp_lo, equalize = 0, False
    if sampler == "pca" and random.random() < 0.7:
        comp_lo = random.choice([40, 60, 90, 120, 160, 200])
        equalize = True
        temp = round(random.uniform(1.05, 1.4), 2)
        tag += f"+weird(lo={comp_lo})"
    return backend, sampler, temp, tag, comp_lo, equalize


def wait_for(job_id, poll=3.0):
    while True:
        for j in _get("/api/state")["jobs"]:
            if j["id"] == job_id and j["status"] in ("done", "error", "cancelled"):
                return j["status"]
        time.sleep(poll)


def main():
    started = time.time()
    seed = int(os.environ.get("SA_SEED_START", "20000"))
    print(f"[guided] BASE={BASE} -- exploiting stars; explore-floor={EXPLORE}", flush=True)
    print(f"[guided] stop anytime: touch {STOP}", flush=True)
    done = 0
    while True:
        if STOP.exists():
            print("[guided] STOP_EXPLORE -> exiting", flush=True); return
        if (time.time() - started) / 3600 > MAX_HOURS:
            print("[guided] time budget hit -> exiting", flush=True); return
        backend, sampler, temp, tag, comp_lo, equalize = sample_combo()
        body = {"action": "generate", "backend": backend,
                "model": SDXL_MODEL if backend == "sdxl" else None,
                "sampler": sampler, "temperature": temp, "n": N_PER,
                "seed": seed, "steps": STEPS, "scheduler": "ddim",
                "comp_lo": comp_lo, "equalize": equalize,
                "init": INIT_ON, "init_mode": INIT_MODE, "init_folder": INIT_FOLDER,
                "init_strength": INIT_STRENGTH, "ip_scale": INIT_STRENGTH}
        seed += N_PER
        try:
            jid = _post("/api/run", body)["job_id"]
        except Exception as e:
            print(f"[guided] submit failed ({e!r}); retry 10s", flush=True); time.sleep(10); continue
        done += 1
        print(f"[guided] #{done} job {jid}: {backend}/{sampler}/T={temp} [{tag}]", flush=True)
        wait_for(jid)
        if done % SCORE_EVERY == 0:
            try:
                _post("/api/score", {})
                print(f"[guided]   queued aesthetic scoring pass", flush=True)
            except Exception:
                pass
        time.sleep(1.0)


if __name__ == "__main__":
    main()
