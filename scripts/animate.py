#!/usr/bin/env python3
"""Animate an image with AnimateDiff -- promptlessly, from its OWN coordinates.

The other kind of video this project can make. ``morph_film.py`` travels
*between* images; this one stays put and lets a single point in conditioning
space move.

The reason it belongs here rather than being a bolt-on: AnimateDiff is a motion
module attached to the **same SD1.5 UNet** everything else uses, and
``AnimateDiffPipeline`` accepts ``prompt_embeds``. So the ``(77, 768)`` tensor in
an image's ``.npz`` sidecar -- conditioning that was sampled from a distribution
and was never typeable -- drives the video directly. No text encoder is involved
at any point. What you get is not "a description of this picture, animated": it
is that exact off-grid coordinate, in motion.

Because the motion module perturbs the UNet, frame 0 is NOT pixel-identical to
the source still; it is the same *world*, moving. ``--strength`` (img2img over
the whole clip) trades motion for fidelity to the original pixels when you want
the frame itself pinned.

**16 frames is a hard ceiling, not a default.** The motion module was trained on
a 16-frame window and collapses into coloured scribble past it (measured, not
assumed). diffusers' FreeNoise solves that by sliding the window -- but it
demands a *text* prompt (``_check_inputs_free_noise`` rejects ``prompt_embeds``
outright), so it is unreachable from here without patching diffusers, which this
project doesn't do. For longer-*looking* output, drop ``--fps`` or raise it back
up with ``--smooth-fps``, which interpolates motion rather than inventing it.

    python scripts/animate.py --src outputs/generated/anarchy_sd15_600040_000.jpg
    python scripts/animate.py --src ...jpg --fps 8 --smooth-fps 24 --motion-lora zoom-in
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import find_image, image_ext, save_image  # noqa: E402
from semantic_anarchy.upscale import conditioning_source  # noqa: E402
from semantic_anarchy.video import encode_mp4, smooth_mp4  # noqa: E402

OUT = Path("outputs")
FILMS = OUT / "films"

#: The motion module. v1-5-3 is the last SD1.5 one and the most stable.
DEFAULT_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-3"
#: Motion LoRAs — camera moves, applied on top of the adapter.
MOTION_LORAS = {
    "none": None,
    "zoom-in": "guoyww/animatediff-motion-lora-zoom-in",
    "zoom-out": "guoyww/animatediff-motion-lora-zoom-out",
    "pan-left": "guoyww/animatediff-motion-lora-pan-left",
    "pan-right": "guoyww/animatediff-motion-lora-pan-right",
    "tilt-up": "guoyww/animatediff-motion-lora-tilt-up",
    "tilt-down": "guoyww/animatediff-motion-lora-tilt-down",
    "roll-cw": "guoyww/animatediff-motion-lora-rolling-clockwise",
    "roll-ccw": "guoyww/animatediff-motion-lora-rolling-anticlockwise",
}
#: The window the motion module was trained on. Asking for more does not error --
#: it renders, and the tail is noise. Enforced rather than warned about.
TRAIN_FRAMES = 16


def fit_render_size(w: int, h: int, max_side: int, mod: int = 8) -> tuple[int, int]:
    """The source's own aspect, capped and snapped to the VAE's multiple-of-8."""
    scale = min(1.0, max_side / max(w, h))
    return (max(mod, int(w * scale) // mod * mod),
            max(mod, int(h * scale) // mod * mod))


def load_conditioning(src: Path) -> tuple[Path, dict, np.ndarray]:
    """The image's own conditioning, following refines back to who owns it."""
    origin, meta = conditioning_source(src)
    npz = np.load(origin.with_suffix(".npz"))
    if "embeds" not in npz:
        raise SystemExit(
            f"[animate] {origin.name} carries {list(npz.files)}, not sd15 `embeds` — "
            "AnimateDiff rides the SD1.5 UNet, so only sd15/sd2 sources can drive it.")
    return origin, meta, np.asarray(npz["embeds"], np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="image to animate (a flux/hires upscale is followed back "
                         "to the ancestor that owns the conditioning)")
    ap.add_argument("--name", default=None, help="output name (default: the source stem)")
    ap.add_argument("--frames", type=int, default=TRAIN_FRAMES,
                    help=f"clip length, max {TRAIN_FRAMES} (the trained window)")
    ap.add_argument("--fps", type=int, default=8, help="8 is AnimateDiff's native rate")
    ap.add_argument("--smooth-fps", type=int, default=0,
                    help="also write <name>_smooth.mp4 at this fps, with motion-"
                         "compensated interpolation (ffmpeg minterpolate). 24 turns "
                         "a choppy 2s clip into a smooth one without new frames")
    ap.add_argument("--steps", type=int, default=None, help="default: the source's own")
    ap.add_argument("--guidance", type=float, default=None, help="default: the source's own")
    ap.add_argument("--seed", type=int, default=None,
                    help="omit for a FRESH RANDOM seed each run (recorded in "
                         "film.json, so any clip stays reproducible); pass one to "
                         "re-render a clip you liked, or to A/B a knob honestly")
    ap.add_argument("--seed-from-source", action="store_true",
                    help="start in the same noise the still was drawn from "
                         "(the source's image_seed) instead of a random one")
    ap.add_argument("--strength", type=float, default=None,
                    help="pin the source PIXELS: img2img the clip from the still at "
                         "this denoise strength (0.5-0.8). THIS is how you animate a "
                         "flux-enhanced upscale -- the enhanced pixels are the starting "
                         "point, its own sd15 conditioning supplies the meaning")
    ap.add_argument("--max-side", type=int, default=768,
                    help="cap for --strength renders; sd15 loses coherence far past "
                         "768 even with the motion module holding it together")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--prompt", default=None,
                    help="drive the motion with TEXT instead of the image's own "
                         "conditioning. The temporal layers were trained on text, "
                         "and off-grid conditioning gives them little to enact -- "
                         "which is why a promptless clip can come out barely moving")
    ap.add_argument("--prompt-from", default="none", choices=["none", "pez", "native"],
                    help="use the prompt PEZ already discovered for this image "
                         "(sidecar inverted_prompt / native_prompt) -- derived from "
                         "the picture rather than typed at it")
    ap.add_argument("--free-init", action="store_true",
                    help="FreeInit: re-run sampling, each pass reusing the previous "
                         "clip's low-frequency structure. THE fix for frames that "
                         "boil instead of moving. Costs --free-init-iters x the time")
    ap.add_argument("--free-init-iters", type=int, default=3)
    ap.add_argument("--motion-lora", default="none", choices=sorted(MOTION_LORAS),
                    help="camera move layered on top of the motion module")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the PNG/JPEG frames beside the mp4 (they are the "
                         "re-encodable master; the mp4 is ~30x smaller)")
    args = ap.parse_args(argv)

    src = find_image(OUT / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    if src is None or not src.is_file():
        raise SystemExit(f"[animate] source not found: {args.src}")
    origin, meta, emb = load_conditioning(src)

    steps = args.steps or meta.get("steps") or 30
    guidance = args.guidance if args.guidance is not None else meta.get("guidance", 7.5)
    if args.seed is not None:
        seed = args.seed
    elif args.seed_from_source and meta.get("image_seed") is not None:
        seed = int(meta["image_seed"])
    else:
        # Random per run. Pinning to the source's image_seed made every clip of a
        # given image identical, which looks exactly like nothing having changed.
        seed = int.from_bytes(os.urandom(4), "little")
    if args.strength:
        # Render at the SOURCE's true resolution, not the conditioning ancestor's.
        # A flux upscale is 768+ while the sd15 original was 512 and its sidecar
        # often records no size at all -- taking the ancestor's would quietly
        # shrink the enhanced image back down, throwing away the whole point.
        from PIL import Image as _Image
        sw, sh = _Image.open(src).size
        w, h = fit_render_size(sw, sh, args.max_side)
    else:
        h = args.height or meta.get("height") or 512
        w = args.width or meta.get("width") or 512
    w = args.width or w
    h = args.height or h
    ckpt = meta.get("model")
    name = args.name or f"{src.stem}_anim"
    if args.frames > TRAIN_FRAMES:
        raise SystemExit(
            f"[animate] --frames {args.frames} exceeds the motion module's trained "
            f"{TRAIN_FRAMES}-frame window; past it the clip degenerates into noise. "
            f"Use --frames {TRAIN_FRAMES} and --smooth-fps to lengthen it smoothly.")

    # A discovered prompt lives on whichever sidecar the inversion was run
    # against -- the upscale's or the ancestor's.
    prompt = args.prompt
    if not prompt and args.prompt_from != "none":
        key = "inverted_prompt" if args.prompt_from == "pez" else "native_prompt"
        own = {}
        j = src.with_suffix(".json")
        if j.is_file():
            try:
                own = json.loads(j.read_text())
            except ValueError:
                own = {}
        prompt = own.get(key) or meta.get(key)
        if not prompt:
            raise SystemExit(
                f"[animate] {src.name} has no {key} — run the 🔤 inversion on it "
                "first, or pass --prompt")

    print(f"[animate] {src.name}", flush=True)
    if origin != src:
        print(f"[animate] conditioning from {origin.name} (followed the refine chain)",
              flush=True)
    if prompt:
        print(f'[animate] text-driven: "{prompt}"', flush=True)
    print(f"[animate] {args.frames} frames @ {args.fps}fps · {w}x{h} · steps={steps} "
          f"· guidance={guidance} · seed={seed}"
          + (f" · strength={args.strength}" if args.strength else "")
          + (f" · {args.motion_lora}" if args.motion_lora != "none" else ""), flush=True)

    import torch
    from diffusers import (AnimateDiffPipeline, AnimateDiffVideoToVideoPipeline,
                           DDIMScheduler, MotionAdapter, StableDiffusionPipeline)
    from PIL import Image

    from semantic_anarchy.pipeline import SD15_NEGATIVE_PROMPT

    t0 = time.time()
    adapter = MotionAdapter.from_pretrained(args.adapter, torch_dtype=torch.float16)
    if ckpt and str(ckpt).endswith((".safetensors", ".ckpt")):
        sd = StableDiffusionPipeline.from_single_file(
            ckpt, torch_dtype=torch.float16, safety_checker=None)
    else:
        sd = StableDiffusionPipeline.from_pretrained(
            ckpt or "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=torch.float16, safety_checker=None)

    parts = dict(vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer,
                 unet=sd.unet, motion_adapter=adapter, scheduler=sd.scheduler)
    pipe = (AnimateDiffVideoToVideoPipeline(**parts) if args.strength
            else AnimateDiffPipeline(**parts))
    # AnimateDiff was trained against LINEAR betas with linspace spacing; the
    # stock sd15 scheduler is scaled_linear and produces mush with the motion
    # module. This is the single most load-bearing line in the file.
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear", timestep_spacing="linspace",
        clip_sample=False, steps_offset=1)
    if MOTION_LORAS[args.motion_lora]:
        pipe.load_lora_weights(MOTION_LORAS[args.motion_lora], adapter_name="motion")
        pipe.set_adapters(["motion"], adapter_weights=[args.lora_scale])
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_vae_slicing()          # the VAE decodes every frame
    except Exception:                                          # noqa: BLE001
        pass
    print(f"[animate] loaded in {time.time() - t0:.1f}s", flush=True)

    # The CFG branch: the same house negative every sd15 render pushes away from,
    # so the clip is negated exactly like the still it came from.
    with torch.no_grad():
        ids = pipe.tokenizer(SD15_NEGATIVE_PROMPT, padding="max_length", max_length=77,
                             truncation=True, return_tensors="pt").input_ids.to("cuda")
        neg = pipe.text_encoder(ids)[0].half()

    kw = dict(
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    if prompt:
        # Text replaces the sampled conditioning entirely: the pipeline encodes
        # both sides itself, so the house negative goes in as text too.
        kw["prompt"] = prompt
        kw["negative_prompt"] = SD15_NEGATIVE_PROMPT
    else:
        kw["prompt_embeds"] = torch.tensor(emb[None]).half().cuda()
        kw["negative_prompt_embeds"] = neg

    # FreeInit re-noises to the FULL noise level on every pass. In --strength
    # mode the schedule is truncated to strength x steps, so there are not enough
    # steps left to denoise back down and the clip lands in confetti (measured:
    # clean at 0.6 without it, destroyed with it). It belongs to the full
    # schedule only, i.e. conditioning mode.
    if args.free_init and args.strength:
        print("[animate] FreeInit skipped: it needs the full denoise schedule and "
              "--strength truncates it. Use conditioning mode for FreeInit.",
              flush=True)
    elif args.free_init:
        pipe.enable_free_init(num_iters=args.free_init_iters, method="butterworth")
        print(f"[animate] FreeInit on ({args.free_init_iters} iters) — "
              f"~{args.free_init_iters}x slower, far steadier", flush=True)
    t0 = time.time()
    if args.strength:
        # v2v takes a VIDEO: the still, repeated. Denoising every frame from the
        # same picture is what pins the pixels while the motion module pulls
        # them apart -- the classic AnimateDiff img2img trick.
        still = Image.open(src).convert("RGB").resize((w, h))
        # height/width are NOT inferred from the input frames: left unset the
        # v2v pipeline falls back to the UNet's 512 default and silently
        # downsizes a 768 flux upscale back to sd15's native size.
        out = pipe(video=[still] * args.frames, strength=args.strength,
                   height=h, width=w, **kw)
    else:
        out = pipe(num_frames=args.frames, height=h, width=w, **kw)
    frames = out.frames[0]
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[animate] {len(frames)} frames in {time.time() - t0:.1f}s (peak {peak:.1f}GB)",
          flush=True)

    film_dir = FILMS / name
    frame_dir = film_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    ext = image_ext()
    for i, f in enumerate(frames):
        save_image(f, frame_dir / f"frame_{i:04d}{ext}")

    mp4 = film_dir / f"{name}.mp4"
    encode_mp4(str(frame_dir / f"frame_%04d{ext}"), mp4, args.fps, args.crf, "animate")
    smooth = None
    if args.smooth_fps and args.smooth_fps > args.fps:
        smooth = smooth_mp4(mp4, args.smooth_fps, args.crf)
        print(f"[animate] saved {smooth} ({args.fps} -> {args.smooth_fps}fps)", flush=True)
    (film_dir / "film.json").write_text(json.dumps({
        "kind": "animate", "engine": "animatediff", "adapter": args.adapter,
        "source": str(src.relative_to(OUT)) if OUT in src.parents else str(src),
        "conditioning_from": (str(origin.relative_to(OUT))
                              if OUT in origin.parents else str(origin)),
        "backend": meta.get("backend", "sd15"), "model": ckpt,
        "frames": len(frames), "fps": args.fps,
        "steps": steps, "guidance": guidance, "seed": seed,
        "strength": args.strength, "motion_lora": args.motion_lora,
        "prompt": prompt, "prompt_from": args.prompt_from if prompt else None,
        "free_init": (args.free_init_iters
                      if args.free_init and not args.strength else None),
        "smooth_fps": args.smooth_fps or None,
        "lora_scale": args.lora_scale if args.motion_lora != "none" else None,
        "height": h, "width": w, "created": time.time(),
    }, indent=2))

    if not args.keep_frames:
        for f in frame_dir.iterdir():
            f.unlink()
        frame_dir.rmdir()

    print(f"[animate] saved {mp4}", flush=True)
    print(f"[animate] watch: /img?path=films/{name}/{name}.mp4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
