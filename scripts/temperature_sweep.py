#!/usr/bin/env python3
"""Temperature sweep -- watch the typical center drift out toward anarchy.

For a fixed seed the underlying draw ``z`` is identical across all temperatures
(same rng seed), so the only thing changing down a column is how far temperature
pushes that same sample out:  ``embedding = mean + temperature * std * z`` (and,
for --sampler pca, per-principal-component sigma -> temps > 1 EXTRAPOLATE outside
the corpus hull). Rows = temperatures (left-labeled), columns = seeds. PIL only.

Works on EITHER backend via --backend; same command, only --backend (+ the right
--model/--ckpt) flips. Run::

    python scripts/temperature_sweep.py --backend sd15 --dist outputs/dist \
        --temps 0.5,1.0,1.5,2.0 --seeds 0,1,2 --steps 30 --guidance 7.5
    python scripts/temperature_sweep.py --backend sdxl --dist outputs/dist \
        --model ~/models/sdxl-base --temps 1,2,3,4 --steps 30 --guidance 7 \
        --sampler pca
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import unique_path
from semantic_anarchy.cli_args import (
    add_backend_args, resolve_gen_defaults, load_backend, dist_prefix,
)


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _strip(width, height, text):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(img).text((8, height // 2 - 6), text, fill="black")
    return img


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--dist", type=Path, default=Path("outputs/dist"))
    parser.add_argument("--temps", default="0.5,1.0,1.5,2.0,2.5",
                        help="comma temperatures (one row each)")
    parser.add_argument("--seeds", default="0,1,2",
                        help="comma seeds (one column each)")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/temperature_sweep.png"))
    args = parser.parse_args(argv)
    resolve_gen_defaults(args)

    from PIL import Image

    temps = _floats(args.temps)
    seeds = _ints(args.seeds)

    print(f"[sweep] loading {args.backend} model ...")
    backend = load_backend(args)
    dists = backend.load_dists(dist_prefix(args, str(args.dist)))
    print(f"[sweep] backend={args.backend} temps={temps} seeds={seeds} "
          f"({len(temps)}x{len(seeds)} images)")

    def render(temp, seed):
        rng = np.random.default_rng(seed)
        named = backend.sample(dists, n=1, temperature=temp, rng=rng,
                               sampler=args.sampler, coherence=args.coherence,
                               components=args.components, truncation=args.truncation)
        kw = dict(guidance=args.guidance, steps=args.steps, seed=seed,
                  height=args.height, width=args.width, neg_mode=args.neg_mode)
        if args.backend == "sdxl":
            kw["dists"] = dists
        return backend.generate(named, **kw)[0]

    grid = []
    cell_w = cell_h = None
    for ti, temp in enumerate(temps):
        row = []
        for si, seed in enumerate(seeds):
            img = render(temp, seed)
            if cell_w is None:
                cell_w, cell_h = img.size
            row.append(img)
            print(f"[sweep]   temp={temp:<4} seed={seed} done "
                  f"({ti*len(seeds)+si+1}/{len(temps)*len(seeds)})", flush=True)
        grid.append(row)

    label_w, header_h = 90, 28
    cols, rows = len(seeds), len(temps)
    sheet = Image.new("RGB", (label_w + cols*cell_w, header_h + rows*cell_h), "white")
    sheet.paste(_strip(label_w, header_h, "temp \\ seed"), (0, 0))
    for si, seed in enumerate(seeds):
        sheet.paste(_strip(cell_w, header_h, f"seed {seed}"), (label_w + si*cell_w, 0))
    for ti, temp in enumerate(temps):
        y = header_h + ti*cell_h
        sheet.paste(_strip(label_w, cell_h, f"T={temp}"), (0, y))
        for si in range(cols):
            sheet.paste(grid[ti][si], (label_w + si*cell_w, y))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = unique_path(args.out.with_name(f"{args.out.stem}_{args.backend}{args.out.suffix}"))
    sheet.save(out)
    print(f"[sweep] contact sheet ({rows} temps x {cols} seeds) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
