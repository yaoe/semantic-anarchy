"""The labels dataset and experiment identity — the referee of the exploration loop.

Torch-free, stdlib only (plus :mod:`.dist_paths` for the checkpoint slug), so the
dashboard, the report card and the tests can all share one definition of what a
label *is*.

Two artifacts live here:

**The dataset** — ``labels/labels.jsonl`` at the repo root, append-only, one JSON
object per line, **git-tracked**. Labels are the only output of this project that
cannot be regenerated: an image can always be re-rendered from its seed, but the
five seconds of a human eye deciding "that one is a 7" cannot. That is why it
does not live under the gitignored ``outputs/``. Nothing ever rewrites a line;
relabeling appends a fresh record and :func:`latest_by_rel` takes the last one.
Each record snapshots the knobs *at labeling time*, so it stays meaningful even
if the PNG and its sidecars are later deleted.

**The experiment manifest** — ``outputs/experiments/<id>.json``, written by every
image-producing script that was given ``--experiment``. argv + dist + ckpt + an
optional one-line hypothesis, so a batch is reconstructible months later. It is
regenerable-ish scaffolding (the labels carry the id independently), hence
``outputs/``.

The **fixed seed panel** (:data:`SEED_PANEL`) is the third piece of protocol:
comparative batches render against the same 16 noise draws so A/B pairs differ by
the idea alone. ``generate.py`` seeds image *i* with ``seed + i``, so the whole
panel is just ``--seed 1000 --n 16``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from .dist_paths import model_slug

#: Score range of one label. 0-9 because a single keypress must cover the scale.
SCORE_MIN, SCORE_MAX = 0, 9
#: A "keeper": the tail we optimise for. Keeper-rate = share of labels >= this.
KEEPER_MIN = 7

#: The fixed seed panel: comparative batches render against these 16 noise draws
#: so an A/B differs by the idea and nothing else. ``generate.py`` derives image
#: i's seed as ``batch_seed + i``, which makes the panel one flag pair:
#: ``--seed 1000 --n 16``.
SEED_PANEL_SEED = 1000
SEED_PANEL_N = 16
SEED_PANEL = tuple(range(SEED_PANEL_SEED, SEED_PANEL_SEED + SEED_PANEL_N))

#: Longest an experiment id may be, after slugging.
MAX_ID_LEN = 48

#: Sidecar fields copied verbatim into a label record. Everything a later
#: response-surface fit (E18) or per-knob report breakdown needs, and nothing
#: that would balloon the file. Missing keys are simply absent.
KNOB_KEYS = (
    "kind", "mode", "sampler", "temperature", "coherence", "components",
    "comp_lo", "equalize", "truncation", "steps", "guidance", "scheduler",
    "neg_mode", "target_distance", "min_distance", "height", "width",
    "radius", "mutate", "direction", "step", "base_blend", "elites",
    "init_mode", "init_strength", "ip_scale",
)

_append_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Where things live
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    """The repository root (this package's parent)."""
    return Path(__file__).resolve().parent.parent


def labels_file(root: Optional[Path] = None) -> Path:
    """The append-only dataset. ``SA_LABELS_FILE`` relocates it (tests, forks)."""
    env = os.environ.get("SA_LABELS_FILE")
    if env and root is None:
        return Path(os.path.expanduser(env))
    return (root or repo_root()) / "labels" / "labels.jsonl"


def experiments_dir(root: Optional[Path] = None) -> Path:
    """Where per-experiment manifests are written."""
    return (root or repo_root()) / "outputs" / "experiments"


# --------------------------------------------------------------------------- #
# Experiment identity
# --------------------------------------------------------------------------- #
def clean_experiment_id(raw: Optional[str]) -> Optional[str]:
    """Slug an experiment id, or ``None`` when there isn't one.

    Ids end up in filenames (the manifest), in argv and in the dataset, so they
    are reduced to ``[A-Za-z0-9._-]`` here once rather than validated in three
    places. ``"E07 · negatives"`` -> ``"E07-negatives"``.
    """
    if raw is None:
        return None
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw).strip()).strip("-._")
    return slug[:MAX_ID_LEN] or None


def manifest_path(exp_id: str, root: Optional[Path] = None) -> Path:
    return experiments_dir(root) / f"{clean_experiment_id(exp_id)}.json"


def write_manifest(exp_id: str, data: dict, root: Optional[Path] = None) -> Optional[Path]:
    """Record what a tagged batch actually was.

    Re-running the same id keeps the FIRST manifest's ``created`` and appends to
    its ``runs`` list — an experiment is usually a few batches, not one.
    """
    exp = clean_experiment_id(exp_id)
    if not exp:
        return None
    path = manifest_path(exp, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = read_manifest(exp, root) or {}
    runs = list(prev.get("runs", []))
    runs.append({"ts": time.time(), **data})
    doc = {
        "id": exp,
        "created": prev.get("created", time.time()),
        # The hypothesis is written once (by whoever first tagged the id) and
        # then carried; a later batch with no --hypothesis doesn't erase it.
        "hypothesis": data.get("hypothesis") or prev.get("hypothesis"),
        "runs": runs[-50:],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def read_manifest(exp_id: str, root: Optional[Path] = None) -> Optional[dict]:
    try:
        doc = json.loads(manifest_path(exp_id, root).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def list_manifests(root: Optional[Path] = None) -> list[dict]:
    """Every experiment manifest on disk, newest first."""
    d = experiments_dir(root)
    if not d.is_dir():
        return []
    out = []
    for p in d.glob("*.json"):
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("id"):
            out.append(doc)
    out.sort(key=lambda d: d.get("created", 0), reverse=True)
    return out


def used_seed_panel(seed: Optional[int], n: Optional[int]) -> bool:
    """Did this batch render against the fixed panel (seed 1000, n 16)?"""
    return seed == SEED_PANEL_SEED and (n or 0) >= SEED_PANEL_N


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
def clamp_score(score) -> int:
    """A label is an int in [0, 9]; anything else is a bad request upstream."""
    s = int(score)
    if not SCORE_MIN <= s <= SCORE_MAX:
        raise ValueError(f"score {score!r} outside {SCORE_MIN}-{SCORE_MAX}")
    return s


def make_record(rel: str, score, meta: Optional[dict] = None,
                ts: Optional[float] = None) -> dict:
    """One label, snapshotting everything the image's ``.json`` sidecar knows.

    The snapshot is the point: the report card, the knob-response surface and
    the score regressor all read the dataset, never the sidecars, so a label
    stays a complete data point after its PNG is wiped.
    """
    m = meta or {}
    rec = {
        "rel": str(rel),
        "ts": time.time() if ts is None else float(ts),
        "score": clamp_score(score),
        "experiment": clean_experiment_id(m.get("experiment")),
        "backend": m.get("backend"),
        "ckpt_slug": model_slug(m["model"]) if m.get("model") else None,
        "dist": m.get("dist"),
        "distance": m.get("distance"),
        "image_seed": m.get("image_seed"),
        "batch_seed": m.get("batch_seed"),
        "knobs": {k: m[k] for k in KNOB_KEYS if k in m and m[k] is not None},
    }
    return rec


def append_label(rec: dict, path: Optional[Path] = None) -> Path:
    """Append one record. Never rewrites, never reorders — history is the file."""
    p = path or labels_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False)
    with _append_lock:
        # Append mode + one write() per line: concurrent appends of a short line
        # to an O_APPEND handle don't interleave, so a second labeler (or the
        # dashboard plus a script) can't corrupt each other's rows.
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return p


def read_labels(path: Optional[Path] = None) -> list[dict]:
    """Every record, in file order. Malformed lines are skipped, not fatal."""
    p = path or labels_file()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and "rel" in rec and "score" in rec:
            out.append(rec)
    return out


def latest_by_rel(records: Iterable[dict]) -> dict[str, dict]:
    """Latest record per image — relabeling is allowed and the newest wins."""
    out: dict[str, dict] = {}
    for rec in records:
        out[rec["rel"]] = rec
    return out


def by_experiment(records: Iterable[dict]) -> dict[str, list[dict]]:
    """Group (already-deduped) records by experiment id; ``""`` = untagged."""
    out: dict[str, list[dict]] = {}
    for rec in records:
        out.setdefault(rec.get("experiment") or "", []).append(rec)
    return out


# --------------------------------------------------------------------------- #
# Summary statistics — the report card's headline row
# --------------------------------------------------------------------------- #
def percentile(values: list[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (``q`` in [0, 100]); ``None`` when empty."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)


def summarize(scores: Iterable) -> dict:
    """The numbers that decide whether a strategy earned another batch.

    Deliberately tail-weighted: ``keeper_rate`` and ``p90`` come first because
    forty 3s and ten 9s beats fifty 6s, and the mean cannot tell them apart.
    """
    xs = [float(s) for s in scores]
    hist = [0] * (SCORE_MAX - SCORE_MIN + 1)
    for s in xs:
        idx = int(round(s)) - SCORE_MIN
        if 0 <= idx < len(hist):
            hist[idx] += 1
    n = len(xs)
    return {
        "n": n,
        "keeper_rate": (sum(1 for s in xs if s >= KEEPER_MIN) / n) if n else None,
        "p90": percentile(xs, 90),
        "median": percentile(xs, 50),
        "mean": (sum(xs) / n) if n else None,
        "max": max(xs) if n else None,
        "hist": hist,
    }


def summarize_records(records: Iterable[dict]) -> dict:
    return summarize([r["score"] for r in records])
