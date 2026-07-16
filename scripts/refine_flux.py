#!/usr/bin/env python3
"""FLUX.2-klein refine/upscale -- reference-conditioned regeneration.

Instead of SD img2img (noise the latent, denoise), klein takes the source image
as a NATIVE reference (kontext-style conditioning) and regenerates it at the
target resolution following an enhancement instruction. A far stronger image
prior than SDXL: it rebuilds coherent fine detail rather than sharpening pixels.

Runs in the flux venv (.venv-flux). Model via --model / SA_FLUX2_MODEL
(black-forest-labs/FLUX.2-klein-4B default; 9B slots in once the HF token is set).

    .venv-flux/bin/python scripts/refine_flux.py --src outputs/generated/X.png --scale 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import unique_path

DEFAULT_PROMPT = (
    "Faithful upscaling task: output the SAME image at a higher resolution. "
    "Preserve the exact composition, framing, every subject and object in its "
    "exact position, the color palette, lighting, and the original's own "
    "medium, materials and surface texture. Do not add, remove, move, restyle "
    "or reinterpret anything. Do NOT add painterly brushstrokes, oil-paint, "
    "watercolor, canvas grain, or any hand-painted look that is not already "
    "there; if the original is a photograph, render or digital image, keep it "
    "exactly that. Only render the existing image crisper, with finer, native "
    "detail true to its original medium.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="enhancement instruction given with the reference image")
    ap.add_argument("--model", default=os.environ.get(
        "SA_FLUX_REFINE_MODEL", "black-forest-labs/FLUX.2-klein-4B"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-side", type=int, default=2048,
                    help="cap the long side (VRAM guard)")
    ap.add_argument("--outdir", type=Path, default=Path("outputs/generated"))
    args = ap.parse_args(argv)

    if not args.src.exists():
        raise SystemExit(f"[flux-refine] source not found: {args.src}")

    import torch
    from PIL import Image
    from diffusers import Flux2KleinPipeline

    img = Image.open(args.src).convert("RGB")
    w, h = img.size
    s = args.scale
    if max(w, h) * s > args.max_side:
        s = args.max_side / max(w, h)
    tw, th = int(round(w * s / 16) * 16), int(round(h * s / 16) * 16)
    print(f"[flux-refine] {args.model}  {w}x{h} -> {tw}x{th} "
          f"(steps={args.steps}, guidance={args.guidance})", flush=True)

    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "little")  # record it: reproducible upscales
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    out = pipe(prompt=args.prompt, image=[img], height=th, width=tw,
               num_inference_steps=args.steps, guidance_scale=args.guidance,
               generator=gen).images[0]

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = f"flux{str(args.scale).replace('.', 'p')}"
    dest = unique_path(args.outdir / f"{args.src.stem}_{tag}.png")
    out.save(dest)
    dest.with_suffix(".json").write_text(json.dumps({
        "kind": "refine", "engine": "flux2-klein", "model": args.model,
        "refined_from": args.src.name, "scale": args.scale,
        "prompt": args.prompt,
        "steps": args.steps, "guidance": args.guidance,
        "seed": args.seed, "out_size": list(out.size),
    }, indent=2))
    print(f"[flux-refine] saved {dest}  ({out.size[0]}x{out.size[1]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
