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
    parser.add_argument("--target-distance", type=float, default=None,
                        help="shell sampling: pin every sample's distance gauge to this "
                             "value (draw ON the ring where your keepers live); "
                             "direction still comes from the sampler.")
    parser.add_argument("--seed", type=int, default=None,
                        help="random by default (fresh batch each run; the chosen "
                             "seed is printed); pass an int to reproduce a batch")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/generated"))
    parser.add_argument("--init-dir", type=Path, default=None,
                        help="folder of 'good init images' -- each output starts img2img "
                             "from a RANDOM one (entropy injection) instead of pure noise.")
    parser.add_argument("--init-strength", type=float, default=0.7,
                        help="img2img denoise from the init (0.6-0.8 keeps its colors/shapes).")
    parser.add_argument("--init-mode", default="img2img", choices=["img2img", "embedding"],
                        help="img2img = start from the init's latent (structure); "
                             "embedding = IP-Adapter CLIP image embedding (content/style).")
    parser.add_argument("--ip-scale", type=float, default=0.7,
                        help="IP-Adapter image-embedding strength (embedding mode; ~0.6-0.9).")
    args = parser.parse_args(argv)
    resolve_gen_defaults(args)

    print(f"[gen] loading {args.backend} model {args.ckpt or args.model or '(default)'} ...")
    backend = load_backend(args)
    if args.scheduler != "default":
        from semantic_anarchy.pipeline import set_scheduler
        set_scheduler(backend.model.pipe, args.scheduler)
        print(f"[gen] scheduler -> {args.scheduler}")

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
        components=args.components, comp_lo=args.comp_lo, equalize=args.equalize,
    )
    if args.target_distance is not None:
        named = backend.retarget(dists, named, args.target_distance)
        print(f"[gen] shell-retargeted all samples to distance {args.target_distance}")
    extra = f", coherence={args.coherence}" if args.sampler == "blend" else ""
    shapes = ", ".join(f"{k}{np.asarray(v).shape[1:]}" for k, v in named.items())
    print(f"[gen] sampled {args.n} ({shapes}), sampler={args.sampler}{extra}, "
          f"temperature={args.temperature}, truncation={args.truncation}")

    # Optional init-image entropy injection: pick a random good init per image.
    init_images, init_names = None, None
    if args.init_dir is not None:
        from PIL import Image
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        # Recursive: pointing at the init_images root picks from ANY subfolder;
        # pointing at one subfolder picks only within it.
        pool = sorted(p for p in args.init_dir.rglob("*") if p.suffix.lower() in exts)
        if not pool:
            print(f"[gen] WARNING: --init-dir {args.init_dir} has no images; using pure noise.")
        else:
            idx = rng.integers(0, len(pool), size=args.n)
            chosen = [pool[k] for k in idx]
            init_images = [Image.open(p) for p in chosen]
            # record path relative to the chosen init dir (shows subfolder if any)
            init_names = [str(p.relative_to(args.init_dir)) for p in chosen]
            knob = (f"strength={args.init_strength}" if args.init_mode == "img2img"
                    else f"ip_scale={args.ip_scale}")
            print(f"[gen] init-image injection [{args.init_mode}] from {len(pool)} imgs under "
                  f"{args.init_dir.name}/ ({knob}): {', '.join(init_names)}")

    print(f"[gen] decoding -> images (no text encoder; guidance={args.guidance}, "
          f"steps={args.steps}) ...")
    gen_kw = dict(guidance=args.guidance, steps=args.steps, seed=seed,
                  height=args.height, width=args.width, neg_mode=args.neg_mode)
    if args.backend == "sdxl":
        gen_kw["dists"] = dists  # needed for neg_mode=mean
    if init_images is not None:
        if args.init_mode == "embedding":
            gen_kw["ip_images"] = init_images
            gen_kw["ip_scale"] = args.ip_scale
        else:
            gen_kw["init_images"] = init_images
            gen_kw["init_strength"] = args.init_strength
    images = backend.generate(named, **gen_kw)

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, img in enumerate(images):
        path = unique_path(args.outdir / f"anarchy_{args.backend}_{seed}_{i:03d}.png")
        img.save(path)
        # Save this image's conditioning as a sidecar so a later upscale/refine
        # pass can reuse the SAME latents (coherent hires-fix) instead of an
        # empty-prompt refine. One npz per image, named to match the PNG stem.
        sidecar = path.with_suffix(".npz")
        np.savez(sidecar, **{k: np.asarray(v)[i] for k, v in named.items()})
        # And a human-readable param sidecar so the UI can show "what made this".
        import json
        meta = {
            "kind": "generate", "backend": args.backend,
            "model": args.ckpt or args.model or "(default)",
            "sampler": args.sampler, "temperature": args.temperature,
            "coherence": args.coherence if args.sampler == "blend" else None,
            "components": args.components, "comp_lo": args.comp_lo,
            "equalize": args.equalize, "truncation": args.truncation,
            "steps": args.steps, "guidance": args.guidance,
            "scheduler": args.scheduler, "neg_mode": args.neg_mode,
            "height": args.height, "width": args.width,
            "batch_seed": seed, "image_seed": seed + i, "index": i,
            "dist": str(args.dist),
            "distance": round(backend.distance(
                dists, {k: np.asarray(v)[i] for k, v in named.items()}), 3),
            "target_distance": args.target_distance,
            "init_image": (init_names[i] if init_names else None),
            "init_mode": (args.init_mode if init_images is not None else None),
            "init_strength": (args.init_strength if init_images is not None and args.init_mode == "img2img" else None),
            "ip_scale": (args.ip_scale if init_images is not None and args.init_mode == "embedding" else None),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        written.append(path.name)
        print(f"[gen] saved {path}")
    print(f"[gen] done -- {len(written)} promptless images in {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
