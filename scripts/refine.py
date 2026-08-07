#!/usr/bin/env python3
"""Upscale + add denoising steps to an existing output image (promptless img2img).

A "hires fix" for Semantic Anarchy outputs: Lanczos-upscale the source PNG to
``--scale x`` then run an img2img pass for ``--steps`` steps (of which
``--strength`` fraction actually denoise), conditioned on the EMPTY prompt so no
linguistics enter. The backend is auto-detected from the filename
(``anarchy_sd15_*`` -> sd15, ``anarchy_sdxl_*`` -> sdxl) unless ``--backend`` is
given. Run::

    python scripts/refine.py --src outputs/generated/anarchy_sdxl_7_000.jpg \
        --scale 1.5 --steps 40 --strength 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import image_ext, save_image, unique_image_path
from semantic_anarchy.cli_args import add_backend_args, load_backend


def _detect_backend(src: Path, override: str | None) -> str:
    if override:
        return override
    name = src.name
    # flux/krea images get SD-refined by the strongest SD backend (sdxl)
    if any(f"anarchy_{b}" in name for b in ("sdxl", "flux2", "krea2")):
        return "sdxl"
    if "anarchy_sd2" in name:
        return "sd2"
    return "sd15"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)  # --backend/--model/--ckpt/--device/--steps/--guidance/...
    parser.add_argument("--src", type=Path, required=True, help="image to refine")
    parser.add_argument("--scale", type=float, default=1.5, help="upscale factor")
    parser.add_argument("--strength", type=float, default=0.3,
                        help="img2img denoise strength (0=keep, 1=reinvent); ~0.35 is the sweet spot")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/generated"))
    parser.add_argument("--tiled", action="store_true",
                        help="tiled native-resolution detail pass (adds real detail, not just zoom)")
    parser.add_argument("--tile", type=int, default=None,
                        help="tile size px (default: backend native 512/768/1024)")
    parser.add_argument("--overlap", type=int, default=128, help="tile overlap px (feathered)")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    # Refine-appropriate CFG per backend. NOT BACKEND_DEFAULTS -- sdxl's default
    # there is turbo's guidance=0.0 (no CFG), which would disable CFG on the
    # refine. base SDXL wants real CFG to reinforce detail.
    REFINE_GUIDANCE = {"sd15": 7.5, "sd2": 9.0, "sdxl": 7.0}

    src = args.src
    if not src.exists():
        raise SystemExit(f"[refine] source not found: {src}")
    # If a concrete model/ckpt was given, trust --backend; otherwise auto-detect
    # the backend from the filename so plain `--src ...` just works.
    if not (args.model or args.ckpt):
        args.backend = _detect_backend(src, None)

    # Reuse the ORIGINAL conditioning (the "same latents" pass) when the
    # generation sidecar exists -- a coherent hires-fix rather than a generic
    # empty-prompt refine. With conditioning present we run real CFG.
    cond = None
    sidecar = src.with_suffix(".npz")
    if sidecar.exists():
        data = np.load(sidecar)
        # Only reuse conditioning if the sidecar's tensors match the refining
        # backend (a flux2 image refined via sdxl falls back to empty-prompt).
        if args.backend == "sdxl" and "prompt_embeds" in data.files:
            cond = {"prompt_embeds": data["prompt_embeds"], "pooled": data["pooled"]}
        elif args.backend != "sdxl" and "embeds" in data.files:
            cond = data["embeds"]

    steps = args.steps if args.steps is not None else 50
    if args.guidance is not None:
        guidance = args.guidance
    elif cond is not None:
        guidance = REFINE_GUIDANCE.get(args.backend, 7.0)       # CFG reinforces the content
    else:
        guidance = 1.0                                          # no sidecar -> plain refine
    scheduler = args.scheduler if args.scheduler != "default" else "ddim"

    print(f"[refine] backend={args.backend} src={src.name} scale={args.scale} steps={steps} "
          f"strength={args.strength} guidance={guidance} scheduler={scheduler} "
          f"cond={'reused (same latents)' if cond is not None else 'empty-prompt (no sidecar)'}")
    print(f"[refine] loading {args.backend} model {args.ckpt or args.model or '(default)'} ...")
    backend = load_backend(args)

    img = Image.open(src)
    if args.tiled:
        from semantic_anarchy.pipeline import tiled_upscale
        tile = args.tile or {"sd15": 512, "sd2": 768, "sdxl": 1024}.get(args.backend, 768)
        print(f"[refine] {img.size[0]}x{img.size[1]} -> TILED x{args.scale} "
              f"(tile={tile} overlap={args.overlap}, native-res detail) ...")

        def _tile_fn(t):
            # Denoise one native-res tile in place (scale=1.0 -> no resize).
            return backend.model.refine_image(
                t, scale=1.0, num_inference_steps=steps, strength=args.strength,
                guidance_scale=guidance, seed=args.seed, cond=cond, scheduler=scheduler)

        out_img = tiled_upscale(img, args.scale, tile, args.overlap, _tile_fn)
    else:
        print(f"[refine] {img.size[0]}x{img.size[1]} -> upscaling x{args.scale} + denoising ...")
        out_img = backend.model.refine_image(
            img, scale=args.scale, num_inference_steps=steps, strength=args.strength,
            guidance_scale=guidance, seed=args.seed, cond=cond, scheduler=scheduler)

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"up{str(args.scale).replace('.', 'p')}"
    out = unique_image_path(args.outdir / f"{src.stem}_{tag}{image_ext()}")
    save_image(out_img, out)
    import json
    out.with_suffix(".json").write_text(json.dumps({
        "kind": "refine", "backend": args.backend, "refined_from": src.name,
        "mode": "tiled" if args.tiled else "single", "tile": args.tile,
        "overlap": args.overlap if args.tiled else None,
        "scale": args.scale, "strength": args.strength, "steps": steps,
        "guidance": guidance, "scheduler": scheduler,
        "cond_reused": cond is not None, "seed": args.seed,
        "out_size": list(out_img.size),
    }, indent=2))
    print(f"[refine] saved {out}  ({out_img.size[0]}x{out_img.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
