#!/usr/bin/env python3
"""The report card: turn "I ran a batch" into "I measured a cell".

Reads the labels dataset (``labels/labels.jsonl``), each image's ``.json``
sidecar and — when they exist — the caches ``scripts/resonance.py`` already
writes, then emits ONE self-contained HTML page with a strip per experiment:

* the **tail** metrics first (keeper-rate %>=7, P90) because forty 3s and ten 9s
  beats fifty 6s and the mean cannot tell those apart,
* the label histogram,
* novelty (nearest-gallery-neighbour distance) and the share of near-duplicates
  *within* the batch,
* a per-knob breakdown for every knob the experiment actually swept,
* and thumbnails ranked by label, best first.

Torch-free: numpy for the duplicate check, PIL only to shrink thumbnails (and
the page still builds without it, just without pictures).

    python scripts/experiment_report.py                  # every labeled experiment
    python scripts/experiment_report.py E00-census E01-length
    python scripts/experiment_report.py --thumbs 60 --out report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import find_image
from semantic_anarchy.labels import (
    KEEPER_MIN, by_experiment, labels_file, latest_by_rel,
    list_manifests, read_labels, summarize_records,
)

OUTPUTS = Path("outputs")
#: Cosine similarity above which two CLIP embeddings are "the same image again".
#: 0.95 is well above the ~0.7 a pair of unrelated promptless samples scores and
#: below what a genuine variation pair reaches.
DUP_COS = 0.95
#: Knobs whose per-value breakdown is worth printing when the batch varied them.
BREAKDOWN_SKIP = {"kind", "height", "width", "steps", "scheduler"}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_clip() -> tuple[dict, np.ndarray | None]:
    """``outputs/clip_embeds.npz`` from resonance.py: rel -> row index, vectors."""
    f = OUTPUTS / "clip_embeds.npz"
    if not f.is_file():
        return {}, None
    try:
        d = np.load(f, allow_pickle=False)
        names = [str(n) for n in d["names"]]
        vecs = d["vecs"].astype(np.float32)
    except Exception:                                        # noqa: BLE001
        return {}, None
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
    return {n: i for i, n in enumerate(names)}, vecs


def duplicate_share(rels: list[str], index: dict, vecs) -> float | None:
    """Share of images in the batch that have a near-twin inside the same batch.

    A strategy that produces fifty renders of one idea is not exploring, however
    good those fifty are — this is the number that catches it.
    """
    if vecs is None:
        return None
    rows = [index[r] for r in rels if r in index]
    if len(rows) < 2:
        return None
    v = vecs[rows]
    sims = v @ v.T
    np.fill_diagonal(sims, -1.0)
    return float((sims.max(axis=1) > DUP_COS).mean())


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def thumb_uri(rel: str, width: int) -> str | None:
    """A downscaled JPEG data URI, so the page survives being moved or mailed."""
    # By stem, not by name: a label records the extension the image had when it
    # was scored, and the render may have been re-made in another format since.
    p = find_image(OUTPUTS / rel)
    if p is None:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        im = Image.open(p).convert("RGB")
        im.thumbnail((width, width * 2))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
    except Exception:                                        # noqa: BLE001
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fmt(x, nd=2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def histogram_svg(hist: list[int]) -> str:
    """The label distribution as a bare inline SVG — no plotting stack needed."""
    top = max(hist) or 1
    w, h, gap = 34, 96, 6
    bars = []
    for s, n in enumerate(hist):
        bh = round(h * n / top)
        x = s * (w + gap)
        tone = "#7fae7f" if s >= KEEPER_MIN else ("#9aa0ad" if s >= 4 else "#5a6070")
        bars.append(
            f'<rect x="{x}" y="{h - bh}" width="{w}" height="{bh}" fill="{tone}" rx="3"/>'
            f'<text x="{x + w / 2}" y="{h + 14}" text-anchor="middle" '
            f'font-size="11" fill="#9aa0ad">{s}</text>'
            + (f'<text x="{x + w / 2}" y="{h - bh - 4}" text-anchor="middle" '
               f'font-size="10" fill="#9aa0ad">{n}</text>' if n else "")
        )
    total_w = len(hist) * (w + gap)
    return (f'<svg class="hist" viewBox="0 -14 {total_w} {h + 20}" '
            f'width="{total_w}" height="{h + 34}">{"".join(bars)}</svg>')


def knob_breakdown(records: list[dict]) -> list[tuple[str, list[dict]]]:
    """Per-value stats for every knob the batch actually varied.

    A knob held constant carries no information about the labels, so it is
    dropped — what is left IS the swept variable (or a confounder worth seeing).
    """
    values: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        for k, v in (rec.get("knobs") or {}).items():
            if k in BREAKDOWN_SKIP:
                continue
            values[k][json.dumps(v, sort_keys=True)].append(rec)
    out = []
    for knob, groups in sorted(values.items()):
        if len(groups) < 2:
            continue
        rows = [{"value": json.loads(key), **summarize_records(recs)}
                for key, recs in groups.items()]
        rows.sort(key=lambda r: (r["keeper_rate"] or 0, r["mean"] or 0), reverse=True)
        out.append((knob, rows))
    return out


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#0d0e12;--panel:#16181f;--line:#2a2e3a;--ink:#e7e9ee;--dim:#9aa0ad;
--good:#7fae7f;--warn:#e0a13d;--acc:#e0533d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 ui-sans-serif,
system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:34px 22px 90px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:19px;margin:0 0 2px}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.8px;color:var(--dim);
margin:22px 0 8px}
.lede{color:var(--dim);margin:0 0 30px;max-width:74ch}
.exp{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:20px 22px;margin:0 0 26px}
.hyp{color:var(--dim);font-style:italic;margin:2px 0 0;max-width:80ch}
.badges{margin:6px 0 0;font-size:12px;color:var(--dim)}
.badge{border:1px solid var(--line);border-radius:20px;padding:1px 9px;margin-right:6px}
.metrics{display:flex;flex-wrap:wrap;gap:26px;margin:16px 0 0}
.metric b{display:block;font-size:26px;font-weight:600;line-height:1.15;
font-variant-numeric:tabular-nums}
.metric span{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--dim)}
.metric.head b{color:var(--good)}
.row{display:flex;flex-wrap:wrap;gap:34px;align-items:flex-start}
table{border-collapse:collapse;font-size:13px;margin:0 0 14px}
th,td{padding:4px 14px 4px 0;text-align:left;font-variant-numeric:tabular-nums}
th{color:var(--dim);font-weight:500;font-size:11px;text-transform:uppercase;
letter-spacing:.6px}
tr+tr td{border-top:1px solid var(--line)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
figure{margin:0}
figure img{width:100%;display:block;border-radius:7px;border:1px solid var(--line)}
figcaption{font-size:11px;color:var(--dim);margin-top:3px;font-variant-numeric:tabular-nums}
.sc{font-weight:600}
.sc.keep{color:var(--good)}.sc.mid{color:var(--dim)}.sc.low{color:#5a6070}
.note{color:var(--dim);font-size:12px}
code{font-family:ui-monospace,Menlo,monospace;font-size:.92em;color:var(--warn)}
"""


def score_class(s: float) -> str:
    return "keep" if s >= KEEPER_MIN else ("mid" if s >= 4 else "low")


def render_experiment(exp: str, records: list[dict], manifest: dict,
                      novelty: dict, clip_index: dict, clip_vecs,
                      thumbs: int, thumb_px: int) -> str:
    s = summarize_records(records)
    rels = [r["rel"] for r in records]
    novs = [novelty[r] for r in rels if r in novelty]
    dup = duplicate_share(rels, clip_index, clip_vecs)
    runs = manifest.get("runs") or []

    title = html.escape(exp or "(untagged)")
    parts = [f'<section class="exp"><h2>{title}</h2>']
    if manifest.get("hypothesis"):
        parts.append(f'<p class="hyp">“{html.escape(manifest["hypothesis"])}”</p>')

    badges = [f'<span class="badge">{len(records)} labeled</span>']
    if runs:
        badges.append(f'<span class="badge">{len(runs)} batch'
                      f'{"es" if len(runs) != 1 else ""}</span>')
    if any(r.get("seed_panel") for r in runs):
        badges.append('<span class="badge">⧉ seed panel</span>')
    backends = {r.get("backend") for r in records if r.get("backend")}
    ckpts = {r.get("ckpt_slug") for r in records if r.get("ckpt_slug")}
    for b in sorted(backends):
        badges.append(f'<span class="badge">{html.escape(str(b))}</span>')
    for c in sorted(ckpts):
        badges.append(f'<span class="badge">{html.escape(str(c))}</span>')
    parts.append(f'<p class="badges">{"".join(badges)}</p>')

    metrics = [
        ("head", "keeper-rate", pct(s["keeper_rate"]), f"labels ≥ {KEEPER_MIN}"),
        ("head", "P90", fmt(s["p90"], 1), "the tail"),
        ("", "median", fmt(s["median"], 1), ""),
        ("", "mean", fmt(s["mean"], 2), ""),
        ("", "best", fmt(s["max"], 0), ""),
    ]
    if novs:
        metrics.append(("", "novelty", fmt(float(np.median(novs)), 3), "median NN-dist"))
    if dup is not None:
        metrics.append(("", "near-dupes", pct(dup), f"cos > {DUP_COS}"))
    parts.append('<div class="metrics">' + "".join(
        f'<div class="metric {cls}"><b>{val}</b><span>{name}</span></div>'
        for cls, name, val, _hint in metrics) + "</div>")

    parts.append('<h3>Label distribution</h3>')
    parts.append(histogram_svg(s["hist"]))

    breakdown = knob_breakdown(records)
    if breakdown:
        parts.append('<h3>Per-knob breakdown — what the sweep actually did</h3>')
        parts.append('<div class="row">')
        for knob, rows in breakdown:
            body = "".join(
                f'<tr><td><code>{html.escape(str(r["value"]))}</code></td>'
                f'<td>{r["n"]}</td><td>{pct(r["keeper_rate"])}</td>'
                f'<td>{fmt(r["p90"], 1)}</td><td>{fmt(r["mean"], 2)}</td></tr>'
                for r in rows)
            parts.append(
                f'<table><tr><th>{html.escape(knob)}</th><th>n</th><th>keep</th>'
                f'<th>P90</th><th>mean</th></tr>{body}</table>')
        parts.append('</div>')
    else:
        parts.append('<p class="note">Nothing was swept — every labeled image in '
                     'this experiment carries the same knobs.</p>')

    ranked = sorted(records, key=lambda r: r["score"], reverse=True)[:thumbs]
    cells = []
    for rec in ranked:
        uri = thumb_uri(rec["rel"], thumb_px)
        if uri is None:
            continue
        seed = rec.get("image_seed")
        cap = (f'<span class="sc {score_class(rec["score"])}">{rec["score"]}</span>'
               f' · d {fmt(rec.get("distance"), 2)}'
               + (f' · seed {seed}' if seed is not None else ''))
        cells.append(f'<figure><img src="{uri}" alt="{html.escape(rec["rel"])}">'
                     f'<figcaption>{cap}</figcaption></figure>')
    if cells:
        parts.append(f'<h3>Ranked by label — best first ({len(cells)} of {len(records)})</h3>')
        parts.append(f'<div class="grid">{"".join(cells)}</div>')
    elif ranked:
        parts.append('<p class="note">No thumbnails: install Pillow, or the PNGs '
                     'have been wiped (the labels themselves survive).</p>')

    parts.append('</section>')
    return "".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*",
                    help="experiment ids (default: every id that has labels). "
                         "Use '' for the untagged pile.")
    ap.add_argument("--labels", type=Path, default=None,
                    help="labels dataset (default: labels/labels.jsonl)")
    ap.add_argument("--out", type=Path, default=Path("experiment_report.html"))
    ap.add_argument("--thumbs", type=int, default=36,
                    help="max thumbnails per experiment (ranked best first)")
    ap.add_argument("--thumb-px", type=int, default=240)
    args = ap.parse_args(argv)

    path = args.labels or labels_file()
    records = list(latest_by_rel(read_labels(path)).values())
    if not records:
        print(f"[report] no labels in {path} — label a batch in the 🏷 tab first")
        return 1

    groups = by_experiment(records)
    wanted = args.experiments or [e for e in groups if e] or [""]
    missing = [e for e in wanted if e not in groups]
    for e in missing:
        print(f"[report] no labels for experiment {e!r}")
    wanted = [e for e in wanted if e in groups]
    if not wanted:
        return 1

    novelty = load_json(OUTPUTS / "novelty.json")
    clip_index, clip_vecs = load_clip()
    manifests = {m["id"]: m for m in list_manifests()}
    if clip_vecs is None:
        print("[report] no outputs/clip_embeds.npz — skipping novelty/duplicate "
              "columns (run scripts/resonance.py to fill them in)")

    total = sum(len(groups[e]) for e in wanted)
    sections = []
    for exp in wanted:
        recs = groups[exp]
        print(f"[report] {exp or '(untagged)'}: {len(recs)} labels", flush=True)
        sections.append(render_experiment(
            exp, recs, manifests.get(exp, {}), novelty, clip_index, clip_vecs,
            args.thumbs, args.thumb_px))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Semantic Anarchy — experiment report cards</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Experiment report cards</h1>
<p class="lede">{total} labels across {len(wanted)} experiment(s), from
<code>{html.escape(str(path))}</code>. Read the <strong>tail</strong>, not the
mean: a strategy producing forty 3s and ten 9s beats one producing fifty 6s, and
only keeper-rate and P90 can tell them apart. The label is the ground truth here
— every number below is a compression of it, never a replacement.</p>
{"".join(sections)}
</div></body></html>
"""
    args.out.write_text(page, encoding="utf-8")
    size = args.out.stat().st_size / 1e6
    print(f"[report] wrote {args.out} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
