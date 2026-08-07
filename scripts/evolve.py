#!/usr/bin/env python3
"""Grow an evolutionary branch -- sample, score by taste, refit, repeat.

Slides 16 & 44: drive the distribution toward a chosen aesthetic with CEM. With
``--scorer aesthetic`` (or ``human``) each generation renders its population and
scores the images; the branch distribution and a contact sheet are saved.

For a torch-free smoke test of the search itself, use ``--scorer random`` with
``--no-sd`` -- it scores raw embeddings analytically and never loads SD.

    python scripts/evolve.py --dist outputs/dist --scorer aesthetic --generations 8
    python scripts/evolve.py --dist outputs/dist --scorer random --no-sd   # dry run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy import (
    EmbeddingDistribution,
    evolve_distribution,
    get_scorer,
)
from semantic_anarchy.evolve import make_image_scorer
from semantic_anarchy.io_utils import image_ext, save_image


def _contact_sheet(images, out_path: Path, cols: int = 4):
    """Tile PIL images into a single contact sheet."""
    from PIL import Image

    if not images:
        return None
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, img in enumerate(images):
        sheet.paste(img, ((i % cols) * w, (i // cols) * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(sheet, out_path)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("outputs/dist"))
    parser.add_argument("--scorer", default="aesthetic",
                        choices=["aesthetic", "human", "random"])
    parser.add_argument("--no-sd", action="store_true",
                        help="don't render images; score embeddings analytically (dry run)")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--truncation", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=Path, default=Path("weights/sac+logos+ava1-l14-linearMSE.pth"))
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/branch"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)

    dist = EmbeddingDistribution.load(args.dist)
    print(f"[evolve] base distribution {args.dist} -> {dist.summary()}")
    rng = np.random.default_rng(args.seed)

    if args.no_sd:
        # Pure-numpy smoke test: reward proximity to a hidden target embedding.
        target = dist.sample(1, temperature=1.5, rng=np.random.default_rng(7))[0]

        def scorer(pop: np.ndarray) -> np.ndarray:
            diff = pop.reshape(pop.shape[0], -1) - target.reshape(-1)[None]
            return (-np.sqrt((diff ** 2).mean(axis=1))).astype(np.float32)

        print("[evolve] --no-sd: scoring embeddings analytically (no torch)")
    else:
        from semantic_anarchy.pipeline import SDModel

        print(f"[evolve] loading SD model {args.model_id} ...")
        sd = SDModel.load(model_id=args.model_id, device=args.device)
        image_scorer = get_scorer(args.scorer, **(
            {"weights_path": args.weights} if args.scorer == "aesthetic" else {}
        ))
        scorer = make_image_scorer(sd, image_scorer, num_inference_steps=args.steps)
        print(f"[evolve] scorer = {image_scorer.name}")

    evolved, history = evolve_distribution(
        dist, scorer,
        generations=args.generations, pop_size=args.pop_size,
        elite_frac=args.elite_frac, temperature=args.temperature,
        truncation=args.truncation, rng=rng, verbose=True,
    )

    evolved.save(args.out)
    print(f"[evolve] branch distribution saved -> {args.out}.npz")
    print(f"[evolve] elite-mean score {history.elite_mean_score[0]:+.3f} -> "
          f"{history.elite_mean_score[-1]:+.3f} "
          f"({'improved' if history.improved() else 'flat'})")

    if not args.no_sd:
        final = evolved.sample(args.pop_size, rng=rng)
        images = sd.generate_from_embeddings(final, num_inference_steps=args.steps)
        sheet = _contact_sheet(images, args.outdir / f"branch_contact_sheet{image_ext()}")
        print(f"[evolve] contact sheet -> {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
