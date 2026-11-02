#!/usr/bin/env python3
"""Sampler sweep -- walk the anarchy <-> coherence axis.

One row per sampler at a fixed temperature; columns share a seed (a matched A/B/C
across samplers at one fixed draw):

* ``diagonal`` -- independent per-coordinate Gaussians (the deck's model).
* ``blend(lambda)`` -- interpolate the diagonal and PCA covariances.
* ``pca`` -- draw within the low-rank corpus subspace (on the manifold).

Works on EITHER backend via --backend (samples every named tensor with the row's
sampler). PIL contact sheet, no matplotlib. Run::

    python scripts/sampler_sweep.py --backend sd15 --dist outputs/dist \
        --seeds 0,1,2 --steps 30 --coherence 0.5
    python scripts/sampler_sweep.py --backend sdxl --dist outputs/dist \
        --model ~/models/sdxl-base --steps 30 --guidance 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import image_ext, save_image, unique_path
from semantic_anarchy.cli_args import (
    add_backend_args, resolve_gen_defaults, load_backend, dist_prefix,
    sampler_kwargs, resolve_lengths, neg_dists_kwarg,
)


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _tile(width, height, text):
    from PIL import Image, ImageDraw
    t = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(t).text((8, height // 2 - 6), text, fill="black")
    return t


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--dist", type=Path, default=Path("outputs/dist"))
    parser.add_argument("--seeds", default="0,1,2",
                        help="comma seeds (one column each)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    resolve_gen_defaults(args)

    from PIL import Image

    seeds = _ints(args.seeds)
    # Rows top (anarchy) -> bottom (coherence). Each is sampler kwargs.
    rows = [
        ("diagonal", dict(sampler="diagonal")),
        (f"blend l={args.coherence}", dict(sampler="blend", coherence=args.coherence)),
        ("pca", dict(sampler="pca")),
    ]

    print(f"[sweep] loading {args.backend} model ...")
    backend = load_backend(args)
    dists = backend.load_dists(dist_prefix(args, str(args.dist)))
    # PCA-requiring rows need pca_components in the fitted dists.
    if any(d.pca_components is None for d in dists.values()):
        raise SystemExit(
            "[sweep] a distribution has no PCA; re-mine so the npz carries "
            "pca_components (mine_distribution.py fits PCA by default).")
    print(f"[sweep] backend={args.backend} rows={[r[0] for r in rows]} seeds={seeds} "
          f"({len(rows)}x{len(seeds)} images)")

    resolve_lengths(args, backend, dists, 1, np.random.default_rng(0))

    def render(sampler_kw, seed):
        rng = np.random.default_rng(seed)
        lengths = resolve_lengths(args, backend, dists, 1, rng, quiet=True)
        # The row's own sampler/coherence override whatever the sidebar picked;
        # everything else (rho, empirical head, truncation, ...) is shared, so
        # every row is the same experiment with one thing changed.
        kwargs = {**sampler_kwargs(args, lengths), **sampler_kw}
        named = backend.sample(dists, n=1, temperature=args.temperature,
                               rng=rng, **kwargs)
        kw = dict(guidance=args.guidance, steps=args.steps, seed=seed,
                  height=args.height, width=args.width, neg_mode=args.neg_mode,
                  **neg_dists_kwarg(args, dists))
        return backend.generate(named, **kw)[0]

    grid, cw, ch = [], None, None
    for ri, (label, kw) in enumerate(rows):
        row_imgs = []
        for ci, seed in enumerate(seeds):
            img = render(kw, seed)
            if cw is None:
                cw, ch = img.size
            row_imgs.append(img)
            print(f"[sweep]   {label:<12} seed={seed} done "
                  f"({ri*len(seeds)+ci+1}/{len(rows)*len(seeds)})", flush=True)
        grid.append(row_imgs)

    label_w, header_h, ncol = 110, 28, len(seeds)
    sheet = Image.new("RGB", (label_w + ncol*cw, header_h + len(rows)*ch), "white")
    sheet.paste(_tile(label_w, header_h, "sampler\\seed"), (0, 0))
    for ci, seed in enumerate(seeds):
        sheet.paste(_tile(cw, header_h, f"seed {seed}"), (label_w + ci*cw, 0))
    for ri, (label, _) in enumerate(rows):
        y = header_h + ri*ch
        sheet.paste(_tile(label_w, ch, label), (0, y))
        for ci in range(ncol):
            sheet.paste(grid[ri][ci], (label_w + ci*cw, y))

    if args.out is None:      # default follows SA_IMAGE_FORMAT
        args.out = Path(f"outputs/sampler_sweep{image_ext()}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = unique_path(args.out.with_name(f"{args.out.stem}_{args.backend}{args.out.suffix}"))
    save_image(sheet, out)
    print(f"[sweep] contact sheet ({len(rows)} samplers x {ncol} seeds) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
