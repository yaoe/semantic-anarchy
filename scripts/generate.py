#!/usr/bin/env python3
"""Promptless generation -- sample conditioning tensor(s) and decode them.

Slide 9, made real for either backend: load the mined distribution(s), draw N
fresh conditioning tensors straight from them (same --sampler/--temperature for
every tensor), and feed them to the UNet via ``prompt_embeds`` (+ pooled for
sdxl) -- the text encoder is never touched. "There is no prompt."

Needs the FULL tier (torch + diffusers). Run::

    python scripts/generate.py --backend sd15 --dist outputs/dist --n 8
    python scripts/generate.py --backend sdxl --dist outputs/dist --n 8 \
        --model ~/models/sdxl-turbo
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--dist", type=Path, default=Path("outputs/dist"),
                        help="path prefix of saved distribution(s)")
    parser.add_argument("--n", type=int, default=8, help="how many images to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help=">1 = wilder/less typical, <1 = closer to the bland center")
    parser.add_argument("--seed", type=int, default=None,
                        help="random by default (fresh batch each run; the chosen "
                             "seed is printed); pass an int to reproduce a batch")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/generated"))
    args = parser.parse_args(argv)
    resolve_gen_defaults(args)

    print(f"[gen] loading {args.backend} model {args.ckpt or args.model or '(default)'} ...")
    backend = load_backend(args)

    prefix = dist_prefix(args, str(args.dist))
    dists = backend.load_dists(prefix)
    print(f"[gen] loaded distribution(s) from {prefix}: "
          f"{', '.join(f'{k}{tuple(d.feature_shape)}' for k, d in dists.items())}")

    # Random by default: draw a concrete seed from entropy and print it.
    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy) % (2**31)
        print(f"[gen] no --seed -> random seed {seed} (pass --seed {seed} to reproduce)")
    rng = np.random.default_rng(seed)

    named = backend.sample(
        dists, n=args.n, temperature=args.temperature, truncation=args.truncation,
        rng=rng, sampler=args.sampler, coherence=args.coherence,
        components=args.components,
    )
    extra = f", coherence={args.coherence}" if args.sampler == "blend" else ""
    shapes = ", ".join(f"{k}{np.asarray(v).shape[1:]}" for k, v in named.items())
    print(f"[gen] sampled {args.n} ({shapes}), sampler={args.sampler}{extra}, "
          f"temperature={args.temperature}, truncation={args.truncation}")

    print(f"[gen] decoding -> images (no text encoder; guidance={args.guidance}, "
          f"steps={args.steps}) ...")
    gen_kw = dict(guidance=args.guidance, steps=args.steps, seed=seed,
                  height=args.height, width=args.width, neg_mode=args.neg_mode)
    if args.backend == "sdxl":
        gen_kw["dists"] = dists  # needed for neg_mode=mean
    images = backend.generate(named, **gen_kw)

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, img in enumerate(images):
        path = unique_path(args.outdir / f"anarchy_{args.backend}_{seed}_{i:03d}.png")
        img.save(path)
        written.append(path.name)
        print(f"[gen] saved {path}")
    print(f"[gen] done -- {len(written)} promptless images in {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
