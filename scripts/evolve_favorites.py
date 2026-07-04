#!/usr/bin/env python3
"""Evolve a personal distribution branch from your STARRED images and sample it.

The Picbreeder/Eden "evolutionary branch" move: collect the latent coordinates
(.npz sidecars) of every favorited image for one backend, refit the distribution
around that elite pool (mean -> your taste center; spread -> elite spread blended
with the base corpus so the branch doesn't collapse), graft the corpus PCA axes
back on so all samplers still work, then sample fresh images FROM YOUR TASTE.

The evolved distributions are saved backend-namespaced under
``outputs/dist_evolved*`` so later runs can point --dist at them directly.

    python scripts/evolve_favorites.py --backend sdxl --n 8 --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import unique_path
from semantic_anarchy.cli_args import (
    add_backend_args, resolve_gen_defaults, load_backend, dist_prefix,
)


def _favorite_anchors(backend_name: str) -> list[dict]:
    """Latent coordinates of every starred image of this backend (with sidecars)."""
    favs_file = Path("outputs/favorites.json")
    if not favs_file.exists():
        raise SystemExit("[evolve] no favorites.json -- star some images first")
    anchors = []
    for rel in json.loads(favs_file.read_text()):
        if f"anarchy_{backend_name}_" not in Path(rel).name:
            continue
        npz = Path("outputs") / Path(rel).with_suffix(".npz")
        if npz.exists():
            data = np.load(npz)
            anchors.append({k: data[k] for k in data.files})
    return anchors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="spread around your taste center (evolved std units)")
    parser.add_argument("--sampler-override", default="diagonal",
                        choices=["diagonal", "pca", "blend"],
                        help="diagonal = tightest to the elite spread (default)")
    parser.add_argument("--base-blend", type=float, default=0.25,
                        help="how much base-corpus spread to blend into the branch")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/generated"))
    args = parser.parse_args(argv)
    resolve_gen_defaults(args)

    anchors = _favorite_anchors(args.backend)
    if len(anchors) < 3:
        raise SystemExit(f"[evolve] only {len(anchors)} starred {args.backend} images "
                         f"with sidecars -- star at least 3 to evolve a branch")
    print(f"[evolve] elite pool: {len(anchors)} starred {args.backend} images")

    print(f"[evolve] loading {args.backend} model {args.ckpt or args.model or '(default)'} ...")
    backend = load_backend(args)
    if args.scheduler != "default":
        from semantic_anarchy.pipeline import set_scheduler
        set_scheduler(backend.model.pipe, args.scheduler)
    dists = backend.load_dists(dist_prefix(args, "outputs/dist"))

    # Refit each named tensor around the elites; keep the corpus PCA axes so the
    # pca/blend samplers (and neighborhood/walk on children) still work.
    evolved = {}
    for k, d in dists.items():
        elites = np.stack([a[k] for a in anchors])
        ev = d.refit_from_elites(elites, base_blend=args.base_blend)
        evolved[k] = replace(ev, pca_components=d.pca_components,
                             pca_std=d.pca_std,
                             corpus_embeddings=d.corpus_embeddings)
        print(f"[evolve]   {k}: taste-center distance from corpus center = "
              f"{d.distance(evolved[k].mean):.2f}")
    written = backend.save_dists(evolved, dist_prefix(args, "outputs/dist_evolved"))
    print(f"[evolve] branch saved -> {', '.join(map(str, written))}")

    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy) % (2**31)
    rng = np.random.default_rng(seed)
    named = backend.sample(evolved, n=args.n, temperature=args.temperature,
                           rng=rng, sampler=args.sampler_override)

    print(f"[evolve] decoding {args.n} from your branch (guidance={args.guidance}, "
          f"steps={args.steps}, seed={seed}) ...")
    gen_kw = dict(guidance=args.guidance, steps=args.steps, seed=seed,
                  neg_mode=args.neg_mode)
    if args.backend == "sdxl":
        gen_kw["dists"] = evolved
    images = backend.generate(named, **gen_kw)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = unique_path(args.outdir / f"anarchy_{args.backend}_{seed}_{i:03d}.png")
        img.save(path)
        one = {k: np.asarray(v)[i] for k, v in named.items()}
        np.savez(path.with_suffix(".npz"), **one)
        meta = {
            "kind": "evolve", "backend": args.backend,
            "model": args.ckpt or args.model or "(default)",
            "elites": len(anchors), "base_blend": args.base_blend,
            "sampler": args.sampler_override, "temperature": args.temperature,
            "steps": args.steps, "guidance": args.guidance,
            "scheduler": args.scheduler, "neg_mode": args.neg_mode,
            "batch_seed": seed, "image_seed": seed + i, "index": i,
            "distance": round(backend.distance(dists, one), 3),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"[evolve] saved {path}")
    print(f"[evolve] done -- {len(images)} images from your evolutionary branch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
