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
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import (IMAGE_EXTS, image_ext, save_image,
                                       unique_image_path)
from semantic_anarchy.cli_args import (
    add_backend_args, add_experiment_args, record_experiment, resolve_gen_defaults,
    load_backend, effective_negative, dist_prefix, sampler_kwargs, resolve_lengths,
    warn_sampler_args,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    add_experiment_args(parser)
    parser.add_argument("--dist", type=Path, default=Path("outputs/dist"),
                        help="path prefix of saved distribution(s)")
    parser.add_argument("--n", type=int, default=8, help="how many images to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help=">1 = wilder/less typical, <1 = closer to the bland center")
    parser.add_argument("--min-distance", type=float, default=None,
                        help="floor: rescale any sample whose distance gauge falls "
                             "below this up onto it (avoid the bland corpus core).")
    parser.add_argument("--target-distance", type=float, default=None,
                        help="shell sampling: pin every sample's distance gauge to this "
                             "value (draw ON the ring where your keepers live); "
                             "direction still comes from the sampler.")
    parser.add_argument("--radius-band", action="store_true",
                        help="give each sample its OWN target distance, bootstrapped "
                             "from the corpus's radius distribution, instead of one "
                             "shell for the batch. The corpus is a band (sd15 spread "
                             "0.031); every sampler is a ~9x tighter spike.")
    parser.add_argument("--radius-scale", type=float, default=1.0,
                        help="shift the whole --radius-band outward (or in). 1.0 = "
                             "the corpus's own band, >1 = the same shape further out.")
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

    # Experiment identity: recorded once the seed is concrete, so the manifest
    # says which noise this batch actually ran against (and whether that was the
    # fixed seed panel).
    experiment = record_experiment(args, seed=seed, dist=str(prefix))
    if experiment:
        print(f"[gen] experiment {experiment} -> outputs/experiments/{experiment}.json")

    warn_sampler_args(args, dists)
    lengths = resolve_lengths(args, backend, dists, args.n, rng)
    named = backend.sample(dists, n=args.n, temperature=args.temperature, rng=rng,
                           **sampler_kwargs(args, lengths))

    # Radius handling, most specific first: an explicit shell wins over the band.
    #
    # Every one of these pins the radius, and `retarget` rescales a sample by
    # target/distance(sample) -- so the temperature factor divides straight back
    # out and the result is bit-identical at any T. Saying so is the difference
    # between "my temperature does nothing" and knowing why. The exception is
    # `hybrid`, where temperature weights the jitter added to a SLERP of two real
    # corpus embeddings rather than scaling a deviation, so it survives the pin.
    pinned = args.target_distance is not None or args.radius_band
    if pinned and args.temperature != 1.0 and args.sampler != "hybrid":
        print(f"[gen] WARNING: --temperature {args.temperature} has no effect here "
              f"-- the radius is pinned, which cancels it exactly. Drop the pin to "
              f"use temperature, or move the temperature into the target itself.")
    if pinned and args.sampler == "split" and args.temp_on != args.temp_off:
        print(f"[gen] note: radius pinned, so only the RATIO "
              f"{args.temp_on}:{args.temp_off} of --temp-on/--temp-off changes "
              f"anything; their common scale cancels.")

    radii = None
    if args.target_distance is not None:
        if args.radius_band:
            print("[gen] WARNING: --target-distance pins one shell, so "
                  "--radius-band is ignored.")
        named = backend.retarget(dists, named, args.target_distance)
        print(f"[gen] shell-retargeted all samples to distance {args.target_distance}")
    elif args.radius_band:
        radii = backend.sample_radii(dists, args.n, rng=rng, scale=args.radius_scale)
        if radii is None:
            print("[gen] WARNING: this distribution has no corpus radius band; "
                  "--radius-band ignored. Re-mine to enable it.")
        else:
            named = backend.retarget(dists, named, radii)
            print(f"[gen] radius band x{args.radius_scale}: targets "
                  f"{radii.min():.3f}-{radii.max():.3f} (mean {radii.mean():.3f})")
    if args.min_distance is not None:
        # The floor runs LAST, so a floor at or above the shell quietly replaces
        # it -- every sample ends up on the floor and the shell never shows.
        if (args.target_distance is not None
                and args.min_distance >= args.target_distance):
            print(f"[gen] WARNING: --min-distance {args.min_distance} is at or "
                  f"above the --target-distance {args.target_distance} shell and "
                  f"is applied after it, so every sample lands on the floor "
                  f"instead of the shell.")
        named = backend.floor_distance(dists, named, args.min_distance)
        print(f"[gen] distance floor enforced: d >= {args.min_distance}")
    extra = f", coherence={args.coherence}" if args.sampler == "blend" else ""
    if args.sampler == "split":
        extra = f", temp_on={args.temp_on}, temp_off={args.temp_off}"
    if args.rho:
        extra += f", rho={args.rho}"
    shapes = ", ".join(f"{k}{np.asarray(v).shape[1:]}" for k, v in named.items())
    print(f"[gen] sampled {args.n} ({shapes}), sampler={args.sampler}{extra}, "
          f"temperature={args.temperature}, truncation={args.truncation}")

    # Optional init-image entropy injection: pick a random good init per image.
    init_images, init_names = None, None
    if args.init_dir is not None:
        from PIL import Image
        exts = IMAGE_EXTS + (".bmp",)
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

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(i, img):
        """Write image ``i`` and its sidecars the moment it finishes rendering.

        The batch is rendered one image at a time, so writing here rather than
        after the whole run is what makes the dashboard gallery fill in live --
        and what lets a cancelled batch keep everything already rendered.
        """
        path = unique_image_path(
            args.outdir / f"anarchy_{args.backend}_{seed}_{i:03d}{image_ext()}")
        one = {k: np.asarray(v)[i] for k, v in named.items()}
        # Both sidecars land BEFORE the image: the gallery scan keys on the
        # image files and then reads them, so an image that appeared first
        # would show up stripped of its params/distance for one poll cycle.
        #
        # The .npz is this image's conditioning -- what makes it explorable
        # later (a coherent hires-fix reuses these exact latents instead of an
        # empty-prompt refine). One per image, named to match the image stem.
        np.savez(path.with_suffix(".npz"), **one)
        # The .json is human-readable: "what made this".
        meta = {
            "kind": "generate", "experiment": experiment, "backend": args.backend,
            "model": args.ckpt or args.model or "(default)",
            "sampler": args.sampler, "temperature": args.temperature,
            "coherence": args.coherence if args.sampler == "blend" else None,
            "components": args.components, "comp_lo": args.comp_lo,
            "equalize": args.equalize, "truncation": args.truncation,
            # The corpus-autopsy knobs. Recorded per image (not per batch) because
            # `length` varies WITHIN a batch in corpus mode, and a label is only
            # as useful as the knob vector it can be regressed against.
            "rho": args.rho or None,
            "length_mode": None if args.length_mode == "off" else args.length_mode,
            "length": (int(lengths[i]) if lengths is not None else None),
            "empirical_head": args.empirical_head or None,
            "temp_on": args.temp_on if args.sampler == "split" else None,
            "temp_off": args.temp_off if args.sampler == "split" else None,
            "radius_band": (round(float(radii[i]), 4) if radii is not None else None),
            "radius_scale": args.radius_scale if radii is not None else None,
            "steps": args.steps, "guidance": args.guidance,
            "scheduler": args.scheduler, "neg_mode": args.neg_mode,
            "negative": effective_negative(backend, args.neg_mode),
            "height": args.height, "width": args.width,
            "batch_seed": seed, "image_seed": seed + i, "index": i,
            "dist": str(args.dist),
            "distance": round(backend.distance(dists, one), 3),
            "target_distance": args.target_distance,
            "min_distance": args.min_distance,
            "init_image": (init_names[i] if init_names else None),
            "init_mode": (args.init_mode if init_images is not None else None),
            "init_strength": (args.init_strength if init_images is not None and args.init_mode == "img2img" else None),
            "ip_scale": (args.ip_scale if init_images is not None and args.init_mode == "embedding" else None),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        save_image(img, path)
        written.append(path.name)
        print(f"[gen] saved {path}  ({len(written)}/{args.n})")

    gen_kw = dict(guidance=args.guidance, steps=args.steps, seed=seed,
                  height=args.height, width=args.width, neg_mode=args.neg_mode,
                  on_image=save)
    if args.backend in ("sd15", "sd2", "sdxl"):
        gen_kw["dists"] = dists  # needed for neg_mode=mean
    if init_images is not None:
        if args.init_mode == "embedding":
            gen_kw["ip_images"] = init_images
            gen_kw["ip_scale"] = args.ip_scale
        else:
            gen_kw["init_images"] = init_images
            gen_kw["init_strength"] = args.init_strength
    backend.generate(named, **gen_kw)

    print(f"[gen] done -- {len(written)} promptless images in {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
