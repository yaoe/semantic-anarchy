#!/usr/bin/env python3
"""Local latent-space navigation around existing images (off-grid instruments).

Where generate.py throws independent darts at the distribution, this script
searches LOCALLY around images you already like -- the sidecar ``.npz`` written
at generation time holds each image's exact conditioning, i.e. its coordinates
in latent space. Two modes:

* ``neighborhood`` -- sample n small perturbations around --src at --radius
  (fraction of the corpus's own spread). Hill-climbing on taste: star the best
  child, explore around THAT.
* ``breed`` -- Picbreeder move: SLERP blends between --src and --b spread over
  the interpolation interval, plus a --mutate-radius perturbation per child.

Every child gets its own ``.npz`` (so it is itself explorable) and a ``.json``
recording parents, mode, radius and its RMS z-score DISTANCE from the corpus
center -- the "how far off-grid" gauge.

    python scripts/explore.py --mode neighborhood --src outputs/generated/anarchy_sdxl_X.png --radius 0.3
    python scripts/explore.py --mode breed --src A.png --b B.png --mutate 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import unique_path
from semantic_anarchy.cli_args import (
    add_backend_args, resolve_gen_defaults, load_backend, dist_prefix,
)


def _detect_backend(src: Path) -> str:
    name = src.name
    for b in ("flux2", "krea2", "sdxl", "sd2"):
        if f"anarchy_{b}_" in name:
            return b
    return "sd15"


def _load_anchor(png: Path) -> dict:
    """Load the conditioning sidecar written next to a generated image."""
    sidecar = png.with_suffix(".npz")
    if not sidecar.exists():
        raise SystemExit(
            f"[explore] no conditioning sidecar for {png.name} "
            f"(expected {sidecar.name}); only images generated after sidecar "
            f"support can be explored.")
    data = np.load(sidecar)
    return {k: data[k] for k in data.files}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--mode", default="neighborhood",
                        choices=["neighborhood", "breed", "walk"])
    parser.add_argument("--direction", default="outward",
                        choices=["outward", "random", "axis"],
                        help="walk: outward = straight toward the periphery")
    parser.add_argument("--axis", type=int, default=None,
                        help="walk --direction axis: which principal axis")
    parser.add_argument("--step", type=float, default=0.15,
                        help="walk: per-frame step (outward: fractional distance growth)")
    parser.add_argument("--src", type=Path, required=True,
                        help="anchor image (parent A for breed)")
    parser.add_argument("--b", type=Path, default=None,
                        help="second parent (breed mode)")
    parser.add_argument("--radius", type=float, default=0.3,
                        help="neighborhood: perturbation size as a fraction of corpus spread")
    parser.add_argument("--mutate", type=float, default=0.15,
                        help="breed: mutation radius added to each child")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/generated"))
    args = parser.parse_args(argv)

    if not args.src.exists():
        raise SystemExit(f"[explore] source not found: {args.src}")
    if args.mode == "breed" and (args.b is None or not args.b.exists()):
        raise SystemExit("[explore] breed mode needs --b <second parent image>")
    if not (args.model or args.ckpt):
        args.backend = _detect_backend(args.src)
    resolve_gen_defaults(args)

    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy) % (2**31)
    rng = np.random.default_rng(seed)

    print(f"[explore] mode={args.mode} backend={args.backend} src={args.src.name}"
          + (f" b={args.b.name}" if args.b else "")
          + {"neighborhood": f" radius={args.radius}",
             "walk": f" direction={args.direction} step={args.step}",
             "breed": f" mutate={args.mutate}"}[args.mode]
          + f" n={args.n} seed={seed}")

    print(f"[explore] loading {args.backend} model {args.ckpt or args.model or '(default)'} ...")
    backend = load_backend(args)
    if args.scheduler != "default":
        from semantic_anarchy.pipeline import set_scheduler
        set_scheduler(backend.model.pipe, args.scheduler)

    dists = backend.load_dists(dist_prefix(args, "outputs/dist"))
    anchor = _load_anchor(args.src)
    anchor_dist = backend.distance(dists, anchor)
    print(f"[explore] anchor distance from corpus center: {anchor_dist:.2f}")

    if args.mode == "neighborhood":
        named = backend.perturb(dists, anchor, n=args.n, radius=args.radius, rng=rng)
    elif args.mode == "walk":
        named = backend.walk(dists, anchor, steps=args.n, step=args.step,
                             mode=args.direction, rng=rng, axis=args.axis)
    else:
        parent_b = _load_anchor(args.b)
        named = backend.breed(dists, anchor, parent_b, n=args.n,
                              mutate=args.mutate, rng=rng)

    print(f"[explore] decoding {args.n} children (guidance={args.guidance}, "
          f"steps={args.steps}) ...")
    gen_kw = dict(guidance=args.guidance, steps=args.steps, seed=seed,
                  height=args.height, width=args.width, neg_mode=args.neg_mode)
    if args.backend == "sdxl":
        gen_kw["dists"] = dists
    images = backend.generate(named, **gen_kw)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = unique_path(args.outdir / f"anarchy_{args.backend}_{seed}_{i:03d}.png")
        img.save(path)
        one = {k: np.asarray(v)[i] for k, v in named.items()}
        np.savez(path.with_suffix(".npz"), **one)
        d = backend.distance(dists, one)
        meta = {
            "kind": "explore", "mode": args.mode, "backend": args.backend,
            "model": args.ckpt or args.model or "(default)",
            "parent": args.src.name,
            "parent_b": (args.b.name if args.b else None),
            "radius": (args.radius if args.mode == "neighborhood" else None),
            "mutate": (args.mutate if args.mode == "breed" else None),
            "direction": (args.direction if args.mode == "walk" else None),
            "step": (args.step if args.mode == "walk" else None),
            "walk_frame": (i + 1 if args.mode == "walk" else None),
            "steps": args.steps, "guidance": args.guidance,
            "scheduler": args.scheduler, "neg_mode": args.neg_mode,
            "batch_seed": seed, "image_seed": seed + i, "index": i,
            "distance": round(d, 3), "anchor_distance": round(anchor_dist, 3),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"[explore] saved {path}  (distance {d:.2f})")
    print(f"[explore] done -- {len(images)} children of {args.src.name} in {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
