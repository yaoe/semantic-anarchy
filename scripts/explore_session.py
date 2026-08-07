#!/usr/bin/env python3
"""Unattended, time-budgeted exploration -- render lots, come back, skim a gallery.

Leave it running for ~an hour while you're away. It loads the SD model and each
corpus distribution *once*, then round-robins through a curated matrix of knob
settings (corpus x temperature x sampler), rendering a few fresh-seeded images
per combo and growing an HTML gallery you can open afterward. Each image's seed
is recorded (in the gallery caption and ``manifest.csv``) so you can reproduce a
favorite with ``scripts/generate.py --seed N``.

Efficiency: the 4GB model is loaded *once* and reused for every render. Time is
the budget -- it keeps doing passes (new seeds each time) until ``--minutes``
elapses, checking the clock between images. Ctrl-C or the deadline stops it
gracefully and finalizes the gallery + manifest (nothing rendered is lost).

    python scripts/explore_session.py --minutes 60 --device mps --ckpt MODEL.ckpt
"""

from __future__ import annotations

import argparse
import csv
import html
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import image_ext, save_image, unique_image_path
from semantic_anarchy.cli_args import (
    add_backend_args, resolve_gen_defaults, load_backend, dist_prefix,
    neg_dists_kwarg,
)


# A small curated set of samplers, paired with a short label fragment.
def _sampler_kwargs(name: str, coherence: float, components):
    """Per-sampler kwargs + a short label fragment for the combo label."""
    if name == "blend":
        return f"blend{coherence}", dict(sampler="blend", coherence=coherence)
    if name == "pca":
        frag = "pca" if components is None else f"pca_k{components}"
        return frag, dict(sampler="pca", components=components)
    if name == "hybrid":
        return "hybrid", dict(sampler="hybrid")
    return name, dict(sampler=name)  # diagonal or anything else


def _build_combos(dists: dict, temperatures, samplers, guidances,
                  coherence, components) -> list[dict]:
    """Cross corpora x temperature x sampler x guidance into a labeled combo list.

    ``dists`` maps a short corpus tag -> loaded EmbeddingDistribution. Samplers
    that need structure the corpus lacks are skipped gracefully: pca/blend need
    PCA, hybrid needs ``corpus_embeddings``.
    """
    combos = []
    for tag, named in dists.items():
        # `named` is a dict {tensor_name: EmbeddingDistribution}. Structure checks
        # use the first tensor (all tensors of one corpus are fit together).
        first = next(iter(named.values()))
        has_pca = first.pca_components is not None
        has_corpus = first.corpus_embeddings is not None
        for temp in temperatures:
            for sname in samplers:
                if sname in ("pca", "blend") and not has_pca:
                    print(f"[explore] skip {sname} for {tag!r} (no PCA)")
                    continue
                if sname == "hybrid" and not has_corpus:
                    print(f"[explore] skip hybrid for {tag!r} (no corpus_embeddings)")
                    continue
                frag, skw = _sampler_kwargs(sname, coherence, components)
                for guid in guidances:
                    combos.append({
                        "label": f"{tag}_T{temp}_{frag}_g{guid}",
                        "tag": tag,
                        "named": named,
                        "temperature": temp,
                        "sampler": skw["sampler"],
                        "coherence": skw.get("coherence"),
                        "components": skw.get("components"),
                        "guidance": guid,
                    })
    return combos


def _write_gallery(index_path: Path, runstamp: str, settings: str,
                   combos: list[dict], records: dict, started: float):
    """(Re)write index.html from the records so far -- safe to call repeatedly."""
    elapsed = time.time() - started
    total = sum(len(v) for v in records.values())
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Semantic Anarchy exploration {html.escape(runstamp)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}"
        "h1{font-size:20px}h2{font-size:15px;margin-top:28px;border-bottom:1px solid #333}"
        ".grid{display:flex;flex-wrap:wrap;gap:10px}.cell{text-align:center;font-size:11px;color:#aaa}"
        "img{display:block;border:1px solid #333;border-radius:4px}</style></head><body>",
        f"<h1>Semantic Anarchy exploration &mdash; {html.escape(runstamp)}</h1>",
        f"<p>{total} images &middot; elapsed {elapsed/60:.1f} min &middot; "
        f"{html.escape(settings)}</p>",
    ]
    for combo in combos:
        imgs = records.get(combo["label"], [])
        if not imgs:
            continue
        knobs = (f"corpus={combo['tag']} &middot; temperature={combo['temperature']} "
                 f"&middot; sampler={combo['sampler']} &middot; guidance={combo['guidance']}")
        if combo["coherence"] is not None:
            knobs += f" &middot; coherence={combo['coherence']}"
        if combo["components"] is not None:
            knobs += f" &middot; components={combo['components']}"
        parts.append(f"<h2>{html.escape(combo['label'])}</h2>")
        parts.append(f"<div style='font-size:12px;color:#888'>{knobs}</div>")
        parts.append("<div class='grid'>")
        for rel, seed in imgs:
            parts.append(
                f"<div class='cell'><img width=256 src='{html.escape(rel)}'>"
                f"<div>seed {seed}</div></div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    index_path.write_text("\n".join(parts))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--minutes", type=float, default=60.0,
                        help="wall-clock budget; stops promptly when exceeded")
    # Sweep axes (comma lists). Defaults target the validated "good zone":
    # the painterly art corpus, low guidance, and the on-manifold samplers.
    parser.add_argument("--dists", default="outputs/dist_art",
                        help="comma-separated corpus distribution path prefixes")
    parser.add_argument("--temperatures", default="0.9,1.1",
                        help="comma-separated temperatures to sweep")
    parser.add_argument("--samplers", default="hybrid,blend,pca",
                        help="comma-separated samplers (diagonal dropped by default)")
    parser.add_argument("--guidances", default="3.5,4.5,5.5",
                        help="comma-separated CFG guidance scales (low = painterly)")
    # NB: --coherence / --components / --steps / --ckpt / --device / --sampler
    # come from add_backend_args (shared with the other scripts).
    parser.add_argument("--imgs-per-combo", type=int, default=3,
                        help="images per combo per pass (new seeds each pass)")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="optional base seed; default is a fresh random seed per image")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/explore_session"))
    args = parser.parse_args(argv)
    resolve_gen_defaults(args)

    # ---- load the backend ONCE, then each corpus's named dists ----------
    print(f"[explore] loading {args.backend} model (once) ...")
    backend = load_backend(args)

    dists = {}
    for path in [p.strip() for p in args.dists.split(",") if p.strip()]:
        prefix = dist_prefix(args, path)
        tag = Path(path).name.replace("dist_", "").replace("dist", "base") or "base"
        try:
            named = backend.load_dists(prefix)
            dists[tag] = named
            first = next(iter(named.values()))
            print(f"[explore] loaded corpus {tag!r} <- {prefix} "
                  f"(pca={'yes' if first.pca_components is not None else 'no'})")
        except Exception as exc:
            print(f"[explore] WARNING: could not load {prefix}: {exc!r} (skipping)")
    if not dists:
        raise SystemExit("[explore] no usable distributions; nothing to do")

    temperatures = [float(x) for x in args.temperatures.split(",") if x.strip()]
    samplers = [x.strip() for x in args.samplers.split(",") if x.strip()]
    guidances = [float(x) for x in args.guidances.split(",") if x.strip()]
    combos = _build_combos(dists, temperatures, samplers, guidances,
                           args.coherence, args.components)
    if not combos:
        raise SystemExit("[explore] no usable combos (samplers vs corpus mismatch?)")
    print(f"[explore] {len(combos)} combos x {args.imgs_per_combo} imgs/pass, "
          f"budget {args.minutes:.0f} min")

    # ---- run folder + manifest + gallery scaffolding --------------------
    runstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.outdir / runstamp
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.csv"
    index_path = run_dir / "index.html"
    settings = (f"backend={args.backend}, model={args.ckpt or args.model or '(default)'}, "
                f"steps={args.steps}, imgs/combo={args.imgs_per_combo}")
    with manifest_path.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["path", "corpus", "temperature", "sampler", "coherence",
             "components", "guidance", "seed", "steps"]
        )
    records: dict[str, list] = {c["label"]: [] for c in combos}

    # ---- graceful stop on SIGINT / deadline -----------------------------
    stop = {"flag": False}

    def _handle_sigint(signum, frame):
        print("\n[explore] Ctrl-C -> finishing current image and finalizing ...")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    started = time.time()
    budget = args.minutes * 60.0
    base_rng = np.random.default_rng(args.seed)
    total = 0
    pass_i = 0

    try:
        while not stop["flag"] and (time.time() - started) < budget:
            pass_i += 1
            for combo in combos:
                if stop["flag"] or (time.time() - started) >= budget:
                    break
                for _ in range(args.imgs_per_combo):
                    if stop["flag"] or (time.time() - started) >= budget:
                        break
                    # Fresh per-image seed (reproducible later via --seed).
                    seed = int(base_rng.integers(0, 2**31 - 1))
                    skw = dict(sampler=combo["sampler"])
                    if combo["coherence"] is not None:
                        skw["coherence"] = combo["coherence"]
                    if combo["components"] is not None:
                        skw["components"] = combo["components"]
                    named_sample = backend.sample(
                        combo["named"], n=1, temperature=combo["temperature"],
                        rng=np.random.default_rng(seed), **skw,
                    )
                    gkw = dict(guidance=combo["guidance"], steps=args.steps, seed=seed,
                               height=args.height, width=args.width, neg_mode=args.neg_mode,
                               **neg_dists_kwarg(args, combo["named"]))
                    img = backend.generate(named_sample, **gkw)[0]

                    combo_dir = run_dir / combo["label"]
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    img_path = unique_image_path(combo_dir / f"img_seed{seed}{image_ext()}")
                    save_image(img, img_path)
                    rel = img_path.relative_to(run_dir).as_posix()
                    records[combo["label"]].append((rel, seed))
                    total += 1

                    with manifest_path.open("a", newline="") as f:
                        csv.writer(f).writerow([
                            rel, combo["tag"], combo["temperature"], combo["sampler"],
                            combo["coherence"], combo["components"], combo["guidance"],
                            seed, args.steps,
                        ])

                # Rewrite the gallery after each combo so an interrupted run is usable.
                _write_gallery(index_path, runstamp, settings, combos, records, started)
                elapsed = time.time() - started
                remaining = max(0.0, budget - elapsed)
                rate = total / elapsed if elapsed > 0 else 0.0
                print(f"[explore] pass {pass_i} | {combo['label']:<28} | "
                      f"imgs={total} | elapsed={elapsed/60:.1f}m "
                      f"remaining={remaining/60:.1f}m | {rate*60:.1f} img/min", flush=True)
    finally:
        _write_gallery(index_path, runstamp, settings, combos, records, started)

    elapsed = time.time() - started
    print(f"\n[explore] DONE: {total} images in {elapsed/60:.1f} min "
          f"({pass_i} pass(es) over {len(combos)} combos)")
    print(f"[explore] gallery  -> {index_path}")
    print(f"[explore] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
