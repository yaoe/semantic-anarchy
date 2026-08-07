#!/usr/bin/env python3
"""Same-latent hires fix: enlarge an output, then re-denoise it with its OWN conditioning.

The faithful upscaler. Where ``refine.py`` is a general-purpose img2img pass (its
own steps, its own scheduler, optionally tiled, optionally prompt-free) and
``refine_flux.py`` hands the image to a different model entirely, this one
reproduces the *original generation* at a higher resolution:

1. Resample the source to ``width*f x height*f``, snapped to a multiple of 16 so
   the VAE/UNet grid stays aligned (``--interp``: lanczos / bicubic / nearest).
2. Feed that back as the init latent of the SAME diffusion model that made it.
3. Denoise the LAST ``--denoise`` fraction of the ORIGINAL schedule -- the same
   step count, guidance, scheduler and seed recorded in the image's sidecar, with
   the exact conditioning vector from its ``.npz``. Nothing new is invented; the
   model simply re-renders the picture it already drew, with more pixels to do it in.

Chains: upscaling an upscale walks ``refined_from`` back to the ancestor that
still owns the conditioning, so 2x then 2x again keeps using the true latents.

    python scripts/upscale.py --src outputs/generated/anarchy_sd15_7_000.jpg \
        --factor 2.0 --denoise 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.cli_args import add_backend_args, load_backend
from semantic_anarchy.io_utils import image_ext, save_image, unique_image_path
from semantic_anarchy.upscale import (
    LATENT_MULTIPLE, clamp_denoise, conditioning_source, denoise_steps, target_size,
)

#: Backends whose conditioning this pass can replay (the SD-family img2img pipes).
SUPPORTED = ("sd15", "sd2", "sdxl")

#: Fallbacks for sidecars that predate a field (or evolve branches that record none).
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = {"sd15": 7.5, "sd2": 9.0, "sdxl": 7.0}


def _backend_of(name: str, meta: dict) -> str:
    """The backend that made the image: its own sidecar first, filename second."""
    b = meta.get("backend")
    if b in SUPPORTED:
        return b
    for cand in SUPPORTED:
        if f"anarchy_{cand}_" in name:
            return cand
    return "sd15"


def _load_cond(npz, backend: str):
    """The exact conditioning tensors that produced the original image."""
    import numpy as np

    data = np.load(npz)
    if backend == "sdxl":
        if "prompt_embeds" not in data.files or "pooled" not in data.files:
            raise SystemExit(
                f"[upscale] {Path(npz).name} has {list(data.files)}, not an sdxl sidecar")
        return {"prompt_embeds": data["prompt_embeds"], "pooled": data["pooled"]}
    if "embeds" not in data.files:
        raise SystemExit(
            f"[upscale] {Path(npz).name} has {list(data.files)}, not an {backend} sidecar")
    return data["embeds"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)   # --backend/--model/--ckpt/--device/--steps/--guidance/...
    parser.add_argument("--src", type=Path, required=True, help="image to upscale")
    parser.add_argument("--factor", type=float, default=2.0,
                        help="upscale factor (target = source * factor, snapped to 16px)")
    parser.add_argument("--denoise", type=float, default=0.3,
                        help="fraction of the ORIGINAL schedule to re-run (0.3 = last 30%%)")
    parser.add_argument("--interp", default="lanczos",
                        choices=("lanczos", "bicubic", "bilinear", "nearest"),
                        help="resampling filter for the enlarge")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the original's image_seed for the added noise")
    parser.add_argument("--max-side", type=int, default=3072,
                        help="cap the long side (VRAM guard); 0 disables")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/generated"))
    args = parser.parse_args(argv)

    from PIL import Image

    src = args.src
    if not src.is_file():
        raise SystemExit(f"[upscale] source not found: {src}")

    # The conditioning (and the original's render settings) may live one or more
    # `refined_from` hops back -- upscales carry no .npz of their own.
    try:
        origin, ometa = conditioning_source(src)
    except FileNotFoundError as exc:
        raise SystemExit(f"[upscale] {exc}")

    backend_name = args.backend if (args.model or args.ckpt) else _backend_of(origin.name, ometa)
    if backend_name not in SUPPORTED:
        raise SystemExit(
            f"[upscale] {backend_name} conditioning can't be replayed here "
            f"(supported: {', '.join(SUPPORTED)}); use the FLUX engine instead")
    args.backend = backend_name
    cond = _load_cond(origin.with_suffix(".npz"), backend_name)

    # Replay the ORIGINAL render settings unless the caller overrode them.
    steps = args.steps if args.steps is not None else int(ometa.get("steps") or DEFAULT_STEPS)
    guidance = (args.guidance if args.guidance is not None
                else float(ometa.get("guidance")
                           if ometa.get("guidance") is not None
                           else DEFAULT_GUIDANCE.get(backend_name, 7.0)))
    scheduler = (args.scheduler if args.scheduler != "default"
                 else (ometa.get("scheduler") or "default"))
    seed = args.seed if args.seed is not None else ometa.get("image_seed")

    denoise = clamp_denoise(steps, args.denoise)
    eff = denoise_steps(steps, denoise)

    img = Image.open(src)
    w, h = img.size
    tw, th = target_size(w, h, args.factor, LATENT_MULTIPLE,
                         args.max_side if args.max_side > 0 else None)
    if args.max_side > 0 and max(w * args.factor, h * args.factor) > args.max_side:
        print(f"[upscale] capped to --max-side {args.max_side}")

    print(f"[upscale] backend={backend_name} src={src.name} "
          f"cond={origin.name}{' (traced)' if origin != src else ''}")
    print(f"[upscale] {w}x{h} -> {tw}x{th} (x{args.factor}, {args.interp}, /{LATENT_MULTIPLE}) "
          f"denoise={denoise} -> last {eff}/{steps} steps, guidance={guidance}, "
          f"scheduler={scheduler}, seed={seed}", flush=True)
    print(f"[upscale] loading {backend_name} model "
          f"{args.ckpt or args.model or '(default)'} ...", flush=True)
    backend = load_backend(args)

    out_img = backend.model.refine_image(
        img, size=(tw, th), interp=args.interp, num_inference_steps=steps,
        strength=denoise, guidance_scale=guidance, seed=seed, cond=cond,
        scheduler=scheduler)

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"hires{str(args.factor).replace('.', 'p')}"
    out = unique_image_path(args.outdir / f"{src.stem}_{tag}{image_ext()}")
    save_image(out_img, out)
    out.with_suffix(".json").write_text(json.dumps({
        "kind": "refine", "engine": "hires", "backend": backend_name,
        "refined_from": src.name, "cond_from": origin.name,
        "factor": args.factor, "denoise": denoise, "interp": args.interp,
        "steps": steps, "denoise_steps": eff, "guidance": guidance,
        "scheduler": scheduler, "seed": seed,
        "cond_reused": True, "scale": args.factor, "strength": denoise,
        "src_size": [w, h], "out_size": list(out_img.size),
    }, indent=2))
    print(f"[upscale] saved {out}  ({out_img.size[0]}x{out_img.size[1]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
