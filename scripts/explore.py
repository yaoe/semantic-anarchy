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

    python scripts/explore.py --mode neighborhood --src outputs/generated/anarchy_sdxl_X.jpg --radius 0.3
    python scripts/explore.py --mode breed --src A.jpg --b B.jpg --mutate 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import image_ext, save_image, unique_image_path
from semantic_anarchy.cli_args import (
    add_backend_args, add_experiment_args, record_experiment, resolve_gen_defaults,
    load_backend, effective_negative, dist_prefix, neg_dists_kwarg,
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


def replay_source_params(args, src: Path) -> dict:
    """``--stay-local``: copy how ``src`` was rendered onto ``args``, and return
    its sidecar, for what can't be copied (noise seed, distribution)."""
    try:
        meta = json.loads(src.with_suffix(".json").read_text())
    except (OSError, ValueError):
        print(f"[explore] {src.name} has no readable params; rendering with the defaults")
        return {}
    for field in ("steps", "guidance", "height", "width", "scheduler", "neg_mode", "negative"):
        if meta.get(field) is not None:
            setattr(args, field, meta[field])
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--dist", type=Path, default=Path("outputs/dist"),
                        help="distribution base the radius/distance are measured against")
    parser.add_argument("--mode", default="neighborhood",
                        choices=["neighborhood", "breed", "walk"])
    parser.add_argument("--direction", default="outward",
                        choices=["outward", "random", "axis", "drift"],
                        help="walk: outward = straight toward the periphery; "
                             "drift = random start, heading turns every frame")
    parser.add_argument("--turn", type=float, default=0.35,
                        help="walk --direction drift: heading change per frame, radians")
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
    parser.add_argument("--stay-local", action="store_true",
                        help="render every child the way --src was rendered: its noise "
                             "seed, size, steps, guidance, scheduler and negative, so the "
                             "only thing that differs from the parent is the perturbation")
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
    source = replay_source_params(args, args.src) if args.stay_local else {}
    child_seed = source.get("image_seed")
    resolve_gen_defaults(args)

    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy) % (2**31)
    rng = np.random.default_rng(seed)
    experiment = record_experiment(args, seed=seed, n=args.n,
                                   dist=dist_prefix(args, str(args.dist)))

    print(f"[explore] mode={args.mode} backend={args.backend} src={args.src.name}"
          + (f" b={args.b.name}" if args.b else "")
          + {"neighborhood": f" radius={args.radius}",
             "walk": f" direction={args.direction} step={args.step}"
                     + (f" turn={args.turn}" if args.direction == "drift" else ""),
             "breed": f" mutate={args.mutate}"}[args.mode]
          + f" n={args.n} seed={seed}")

    print(f"[explore] loading {args.backend} model {args.ckpt or args.model or '(default)'} ...")
    backend = load_backend(args)
    if args.scheduler != "default":
        backend.set_scheduler(args.scheduler)

    dists = backend.load_dists(dist_prefix(args, str(args.dist)))
    anchor = _load_anchor(args.src)
    anchor_dist = backend.distance(dists, anchor)
    print(f"[explore] anchor distance from corpus center: {anchor_dist:.2f}")

    if args.mode == "neighborhood":
        named = backend.perturb(dists, anchor, n=args.n, radius=args.radius, rng=rng)
    elif args.mode == "walk":
        named = backend.walk(dists, anchor, steps=args.n, step=args.step,
                             mode=args.direction, rng=rng, axis=args.axis, turn=args.turn)
    else:
        parent_b = _load_anchor(args.b)
        named = backend.breed(dists, anchor, parent_b, n=args.n,
                              mutate=args.mutate, rng=rng)

    print(f"[explore] decoding {args.n} children (guidance={args.guidance}, "
          f"steps={args.steps}) ...")
    # --neg-mode mean pushes from a distribution's mean, so the parent's own.
    neg_dists = dists
    if source.get("dist") and args.neg_mode == "mean":
        try:
            neg_dists = backend.load_dists(dist_prefix(args, str(source["dist"])))
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"[explore] can't read {source['dist']} ({exc}); "
                  "negative from the selected distribution")
    gen_kw = dict(guidance=args.guidance, steps=args.steps, seed=seed,
                  height=args.height, width=args.width, neg_mode=args.neg_mode,
                  **neg_dists_kwarg(args, neg_dists))
    if child_seed is None:
        images = backend.generate(named, **gen_kw)
    else:
        # generate() seeds image i with seed + i, so one call per child.
        gen_kw["seed"] = child_seed
        images = [backend.generate({k: np.asarray(v)[i:i + 1] for k, v in named.items()},
                                   **gen_kw)[0] for i in range(args.n)]

    args.outdir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = unique_image_path(
            args.outdir / f"anarchy_{args.backend}_{seed}_{i:03d}{image_ext()}")
        save_image(img, path)
        one = {k: np.asarray(v)[i] for k, v in named.items()}
        np.savez(path.with_suffix(".npz"), **one)
        d = backend.distance(dists, one)
        meta = {
            "kind": "explore", "experiment": experiment,
            "mode": args.mode, "backend": args.backend,
            "executor": backend.executor,
            "model": args.ckpt or args.model or "(default)",
            "dist": str(args.dist),
            "parent": args.src.name,
            "parent_b": (args.b.name if args.b else None),
            "radius": (args.radius if args.mode == "neighborhood" else None),
            "mutate": (args.mutate if args.mode == "breed" else None),
            "direction": (args.direction if args.mode == "walk" else None),
            "step": (args.step if args.mode == "walk" else None),
            "turn": (args.turn if args.mode == "walk" and args.direction == "drift" else None),
            "walk_frame": (i + 1 if args.mode == "walk" else None),
            "steps": args.steps, "guidance": args.guidance,
            "scheduler": args.scheduler, "neg_mode": args.neg_mode,
            "negative": effective_negative(backend, args.neg_mode),
            "batch_seed": seed,
            "image_seed": seed + i if child_seed is None else child_seed, "index": i,
            "distance": round(d, 3), "anchor_distance": round(anchor_dist, 3),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"[explore] saved {path}  (distance {d:.2f})")
    print(f"[explore] done -- {len(images)} children of {args.src.name} in {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
