#!/usr/bin/env python3
"""Latent morph film -- travel between selected images through conditioning space.

For each pair of consecutive keyframes (images with .npz conditioning
sidecars), SLERP both ingredients that made them -- the conditioning tensor
AND the initial noise latent (reconstructed exactly from the recorded
image_seed) -- and re-render every in-between frame deterministically (DDIM).
Frame 0 / frame N are therefore EXACTLY the selected images; everything
between is new territory that has never existed.

Optional ``--refine flux``: run every base frame through FLUX.2-klein
reference-conditioned upscaling with ONE fixed seed and ONE fixed faithful
prompt for the whole film, so the reinterpretation stays temporally stable.

    .venv-flux/bin/python scripts/morph_film.py --name pilot \
        --images generated/a.png generated/b.png --frames-per 24 --fps 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("outputs")

FAITHFUL_PROMPT = (
    "Faithful upscaling task: output the SAME image at a higher resolution. "
    "Preserve the exact composition, framing, every subject and object in its "
    "exact position, the color palette, lighting, and the original's own "
    "medium, materials and surface texture. Do not add, remove, move, restyle "
    "or reinterpret anything. Do NOT add painterly brushstrokes, oil-paint, "
    "watercolor, canvas grain, or any hand-painted look that is not already "
    "there; if the original is a photograph, render or digital image, keep it "
    "exactly that. Only render the existing image crisper, with finer, native "
    "detail true to its original medium.")


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interp (direction on the great circle, magnitude linear)."""
    fa, fb = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na < 1e-8 or nb < 1e-8:
        return ((1 - t) * a + t * b)
    ua, ub = fa / na, fb / nb
    dot = float(np.clip(ua @ ub, -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-4:
        direction = (1 - t) * ua + t * ub
    else:
        so = np.sin(omega)
        direction = (np.sin((1 - t) * omega) / so) * ua \
            + (np.sin(t * omega) / so) * ub
    mag = (1 - t) * na + t * nb
    return (mag * direction).reshape(a.shape).astype(np.float32)


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def load_keyframe(rel: str):
    png = OUT / rel
    meta = json.loads(png.with_suffix(".json").read_text())
    npz = np.load(png.with_suffix(".npz"))
    if "embeds" not in npz:
        raise SystemExit(f"[film] {rel}: not an sd15 sidecar (only sd15 supported for now)")
    return {"rel": rel, "meta": meta, "emb": np.asarray(npz["embeds"], np.float32)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True,
                    help="ordered keyframes (paths relative to outputs/); "
                         "flux upscales are auto-redirected to their originals")
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames-per", type=int, default=24,
                    help="frames per transition (incl. easing)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--loop", action="store_true",
                    help="also travel last -> first, so the film loops")
    ap.add_argument("--refine", default="flux", choices=["flux", "none"])
    ap.add_argument("--scale", type=float, default=1.5,
                    help="flux refine scale (512 -> 768 at 1.5)")
    ap.add_argument("--noise-window", type=float, default=1.0,
                    help="fraction of each transition during which the NOISE "
                         "travels (centered). 1.0 = alongside conditioning; "
                         "0.4 = composition locked at both ends, drifting only "
                         "through the middle 40%% of the transition")
    ap.add_argument("--fixed-noise", action="store_true",
                    help="use keyframe 1's initial latent for EVERY frame: pure "
                         "conditioning travel, no noise morph (composition holds "
                         "still; endpoints beyond the first won't match exactly)")
    ap.add_argument("--film-seed", type=int, default=42,
                    help="ONE noise seed for every flux frame (stability)")
    ap.add_argument("--flux-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--resume", action="store_true",
                    help="skip frames whose files already exist (crash recovery)")
    args = ap.parse_args(argv)

    # resolve refines -> originals (the conditioning lives on the original)
    keys = []
    for rel in args.images:
        p = OUT / rel
        for _ in range(5):
            if p.with_suffix(".npz").is_file():
                break
            m = json.loads(p.with_suffix(".json").read_text()) \
                if p.with_suffix(".json").is_file() else {}
            if not m.get("refined_from"):
                break
            p = p.parent / m["refined_from"]
        keys.append(load_keyframe(str(p.relative_to(OUT))))
    if args.loop:
        keys.append(keys[0])
    if len(keys) < 2:
        raise SystemExit("[film] need at least 2 keyframes")

    m0 = keys[0]["meta"]
    ckpt = m0.get("model")
    steps, guidance = m0.get("steps", 30), m0.get("guidance", 7.5)
    h, w = m0.get("height") or 512, m0.get("width") or 512
    print(f"[film] {len(keys)} keyframes, {args.frames_per}/transition, "
          f"{h}x{w}, steps={steps}, guidance={guidance}", flush=True)

    film_dir = OUT / "films" / args.name
    base_dir = film_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import StableDiffusionPipeline

    print(f"[film] loading sd15 {ckpt} ...", flush=True)
    if ckpt and ckpt.endswith((".safetensors", ".ckpt")):
        pipe = StableDiffusionPipeline.from_single_file(
            ckpt, torch_dtype=torch.float16, safety_checker=None)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            ckpt or "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    from semantic_anarchy.pipeline import set_scheduler
    set_scheduler(pipe, m0.get("scheduler", "ddim"))

    def init_latent(seed: int) -> torch.Tensor:
        """EXACT reconstruction of the latent generate() drew for image_seed."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn((1, 4, h // 8, w // 8), generator=g,
                           device="cuda", dtype=torch.float16)

    lats = [init_latent(k["meta"]["image_seed"]).float().cpu().numpy()
            for k in keys]

    # frame schedule: eased t per transition; endpoints render exactly once
    frames = []
    for s in range(len(keys) - 1):
        n = args.frames_per
        for i in range(n):
            t = smoothstep(i / n)
            frames.append((s, t))
    frames.append((len(keys) - 2, 1.0))

    print(f"[film] rendering {len(frames)} base frames ...", flush=True)
    for fi, (s, t) in enumerate(frames):
        if args.resume and (base_dir / f"frame_{fi:04d}.png").is_file():
            continue
        emb = slerp(keys[s]["emb"], keys[s + 1]["emb"], t)
        if args.fixed_noise:
            lat = lats[0]
        else:
            wn = max(1e-6, min(1.0, args.noise_window))
            tn = smoothstep(min(1.0, max(0.0, (t - (0.5 - wn / 2)) / wn)))
            lat = slerp(lats[s], lats[s + 1], tn)
        img = pipe(prompt_embeds=torch.tensor(emb[None]).half().cuda(),
                   latents=torch.tensor(lat).half().cuda(),
                   num_inference_steps=steps, guidance_scale=guidance,
                   height=h, width=w).images[0]
        img.save(base_dir / f"frame_{fi:04d}.png")
        if fi % 5 == 0 or fi == len(frames) - 1:
            print(f"[film]   base {fi + 1}/{len(frames)}", flush=True)

    del pipe
    torch.cuda.empty_cache()

    render_dir = base_dir
    if args.refine == "flux":
        from diffusers import Flux2KleinPipeline
        from PIL import Image
        flux_dir = film_dir / "flux"
        flux_dir.mkdir(exist_ok=True)
        tw = int(round(w * args.scale / 16) * 16)
        th = int(round(h * args.scale / 16) * 16)
        print(f"[film] flux pass {args.flux_model} -> {tw}x{th}, "
              f"seed={args.film_seed} (locked) ...", flush=True)
        fpipe = Flux2KleinPipeline.from_pretrained(
            args.flux_model, torch_dtype=torch.bfloat16)
        try:
            fpipe.enable_model_cpu_offload()
        except Exception:
            fpipe = fpipe.to("cuda")
        fpipe.set_progress_bar_config(disable=True)
        for fi in range(len(frames)):
            if args.resume and (flux_dir / f"frame_{fi:04d}.png").is_file():
                continue
            src = Image.open(base_dir / f"frame_{fi:04d}.png").convert("RGB")
            gen = torch.Generator(device="cpu").manual_seed(args.film_seed)
            out = fpipe(prompt=FAITHFUL_PROMPT, image=[src],
                        height=th, width=tw, num_inference_steps=28,
                        guidance_scale=4.0, generator=gen).images[0]
            out.save(flux_dir / f"frame_{fi:04d}.png")
            print(f"[film]   flux {fi + 1}/{len(frames)}", flush=True)
        render_dir = flux_dir

    mp4 = film_dir / f"{args.name}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(args.fps),
         "-i", str(render_dir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)],
        check=True, capture_output=True)
    (film_dir / "film.json").write_text(json.dumps({
        "keyframes": [k["rel"] for k in keys], "frames": len(frames),
        "fps": args.fps, "refine": args.refine, "film_seed": args.film_seed,
        "steps": steps, "guidance": guidance, "loop": args.loop,
    }, indent=2))
    print(f"[film] saved {mp4}", flush=True)
    print(f"[film] watch: /img?path=films/{args.name}/{args.name}.mp4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
