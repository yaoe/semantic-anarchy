#!/usr/bin/env python3
"""Latent travel film -- interpolate through selected images and render a video.

For each pair of consecutive keyframes (images with ``.npz`` conditioning
sidecars), interpolate both ingredients that made them -- the conditioning
tensor(s) AND the initial noise latent (reconstructed exactly from the recorded
``image_seed``) -- and re-render every in-between frame deterministically.
Frame 0 / frame N are therefore EXACTLY the selected images; everything between
is new territory that has never existed. The frames are muxed to an x264 mp4
that the dashboard can play inline.

The trajectory math (slerp/lerp, easing, the frame schedule, the noise window)
lives in :mod:`semantic_anarchy.travel` and is torch-free / unit-tested.

Optional ``--refine flux``: run every base frame through FLUX.2-klein
reference-conditioned upscaling with ONE fixed seed and ONE fixed faithful
prompt for the whole film, so the reinterpretation stays temporally stable.

    python scripts/morph_film.py --name pilot \
        --images generated/a.jpg generated/b.jpg \
        --frames-per 24 --fps 16 --interp slerp --refine none
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import (  # noqa: E402
    IMAGE_EXTS, image_ext, save_image,
)
from semantic_anarchy.travel import (  # noqa: E402
    EASINGS, INTERPOLATIONS, frame_plan, interpolate, loop_keys, noise_t,
)

OUT = Path("outputs")


def frame_ext(d: Path) -> str:
    """Which extension this frame directory is already using.

    Frames are written in the configured image format (JPEG by default -- a
    2000-frame film is where PNG really costs), but ``--resume`` on a run
    started under a different format must keep writing what ffmpeg will later
    glob as one numbered sequence, so an existing frame wins over the default.
    """
    for e in IMAGE_EXTS:
        if next(d.glob(f"frame_*{e}"), None) is not None:
            return e
    return image_ext()

#: Which named conditioning tensors each backend's sidecar carries. The flow
#: backends (flux2/krea2) pack latents differently and have no reconstructible
#: init noise here, so films stay on the SD family for now.
FILM_TENSORS = {
    "sd15": ("embeds",),
    "sd2": ("embeds",),
    "sdxl": ("prompt_embeds", "pooled"),
}

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


def pick_h264_encoder(crf: int) -> tuple[str, list[str]]:
    """Find an ffmpeg that can actually write H.264, and the flags to do it.

    ``libx264`` is the target, but a conda ffmpeg early on PATH is often built
    ``--disable-gpl`` and has no x264 at all -- so probe every ffmpeg we can
    reach and fall back to another H.264 encoder (openh264 / NVENC) rather than
    dying after a full render. ``SA_FFMPEG`` pins a specific binary.
    """
    import os

    cands = [os.environ.get("SA_FFMPEG"), shutil.which("ffmpeg"),
             "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    seen, fallback = set(), None
    for c in cands:
        if not c or c in seen or not Path(c).is_file():
            continue
        seen.add(c)
        try:
            enc = subprocess.run([c, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception:                                      # noqa: BLE001
            continue
        if "libx264" in enc:
            return c, ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf)]
        if fallback is None and "libopenh264" in enc:
            # No CRF mode in openh264; ~12 Mbit keeps 512-1024px frames clean.
            fallback = (c, ["-c:v", "libopenh264", "-b:v", "12M"])
        if fallback is None and "h264_nvenc" in enc:
            fallback = (c, ["-c:v", "h264_nvenc", "-cq", str(crf)])
    if fallback:
        print(f"[film] no libx264 build found; encoding H.264 with "
              f"{fallback[1][1]} ({fallback[0]})", flush=True)
        return fallback
    raise SystemExit(
        "[film] no ffmpeg with an H.264 encoder found (tried "
        f"{', '.join(sorted(seen)) or 'nothing on PATH'}). Install ffmpeg with "
        "libx264, or point SA_FFMPEG at one that has it.")


def detect_backend(png: Path, meta: dict) -> str:
    """Backend of a generated image -- from its sidecar, else its filename."""
    b = meta.get("backend")
    if b:
        return b
    for name in ("flux2", "krea2", "sdxl", "sd2"):
        if f"anarchy_{name}_" in png.name:
            return name
    return "sd15"


def resolve_original(png: Path) -> Path:
    """Follow ``refined_from`` links back to the image that owns the .npz.

    Upscales/refines have no conditioning of their own, so picking one as a
    keyframe means travelling through the image it came from.
    """
    for _ in range(5):
        if png.with_suffix(".npz").is_file():
            return png
        j = png.with_suffix(".json")
        parent = None
        if j.is_file():
            try:
                parent = json.loads(j.read_text()).get("refined_from")
            except Exception:
                parent = None
        if not parent:
            break
        cand = png.parent / parent
        if not cand.is_file():
            break
        png = cand
    raise SystemExit(f"[film] {png.name}: no conditioning sidecar (.npz) and no "
                     f"traceable original -- it can't be a keyframe")


def keyframe_size(png: Path, meta: dict, backend: str) -> tuple[int, int]:
    """``(height, width)`` this keyframe was actually rendered at.

    The PNG itself is ground truth — older sidecars (evolve branches) recorded
    no height/width at all, and those default to the backend's own resolution
    rather than a hardcoded 512.
    """
    try:
        from PIL import Image
        with Image.open(png) as im:
            return (im.height, im.width)
    except Exception:                                          # noqa: BLE001
        from semantic_anarchy.backend import BACKEND_DEFAULTS
        d = BACKEND_DEFAULTS[backend]
        return (meta.get("height") or d["height"], meta.get("width") or d["width"])


def load_keyframe(png: Path, meta: dict, backend: str) -> dict:
    """Load one keyframe's conditioning + params."""
    npz = np.load(png.with_suffix(".npz"))
    want = FILM_TENSORS[backend]
    missing = [k for k in want if k not in npz]
    if missing:
        raise SystemExit(f"[film] {png.name}: sidecar is missing {missing} "
                         f"(expected a {backend} sidecar with {list(want)})")
    return {
        "rel": str(png.relative_to(OUT)) if OUT in png.parents else png.name,
        "meta": meta,
        "size": keyframe_size(png, meta, backend),
        "cond": {k: np.asarray(npz[k], np.float32) for k in want},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", nargs="+", required=True,
                    help="ordered keyframes (paths relative to outputs/); "
                         "flux upscales are auto-redirected to their originals")
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames-per", type=int, default=24,
                    help="frames rendered per transition (excl. the final frame)")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--interp", default="slerp", choices=list(INTERPOLATIONS),
                    help="slerp = great-circle blend, stays on the endpoints' "
                         "shell (default); lerp = straight line, dips through "
                         "the washed-out interior")
    ap.add_argument("--easing", default="smooth", choices=list(EASINGS),
                    help="pacing within each transition. smooth/smoother rest "
                         "on the keyframes; linear = constant speed")
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
    ap.add_argument("--steps", type=int, default=None,
                    help="override the keyframes' own denoising steps")
    ap.add_argument("--guidance", type=float, default=None,
                    help="override the keyframes' own CFG scale")
    ap.add_argument("--scheduler", default=None,
                    choices=["default", "ddim", "euler", "euler_a", "dpm"],
                    help="default: whatever the first keyframe was made with")
    ap.add_argument("--height", type=int, default=None,
                    help="film resolution (default: keyframe 1's own). Keyframes "
                         "of a different size are re-rendered at this one and "
                         "won't be reproduced exactly.")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better)")
    ap.add_argument("--flux-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--resume", action="store_true",
                    help="skip frames whose files already exist (crash recovery)")
    args = ap.parse_args(argv)

    # Resolve the encoder BEFORE the model load, not after 200 rendered frames.
    ffmpeg, codec_args = pick_h264_encoder(args.crf)

    # Resolve refines -> originals (the conditioning lives on the original).
    pngs = [resolve_original(OUT / rel) for rel in args.images]
    metas = [json.loads(p.with_suffix(".json").read_text())
             if p.with_suffix(".json").is_file() else {} for p in pngs]
    backends = {detect_backend(p, m) for p, m in zip(pngs, metas)}
    if len(backends) > 1:
        raise SystemExit(f"[film] keyframes mix backends {sorted(backends)} -- "
                         f"their conditioning tensors aren't compatible")
    backend = backends.pop()
    if backend not in FILM_TENSORS:
        raise SystemExit(f"[film] backend {backend!r} can't be filmed yet "
                         f"(supported: {', '.join(sorted(FILM_TENSORS))})")

    keys = [load_keyframe(p, m, backend) for p, m in zip(pngs, metas)]
    if len(keys) < 2:
        raise SystemExit("[film] need at least 2 keyframes")
    if args.loop:
        keys = loop_keys(keys)

    m0 = keys[0]["meta"]
    ckpt_or_model = m0.get("model")
    steps = args.steps if args.steps is not None else m0.get("steps", 30)
    guidance = args.guidance if args.guidance is not None else m0.get("guidance", 7.5)
    scheduler = args.scheduler or m0.get("scheduler") or "ddim"

    # A video has ONE resolution. Default to keyframe 1's; --height/--width
    # override. Any keyframe of a different size gets re-rendered at the film's
    # size from a differently-shaped noise draw -- so it will NOT come back as
    # itself, which is the one promise this tool makes. Say so, loudly.
    h = args.height or keys[0]["size"][0]
    w = args.width or keys[0]["size"][1]
    if h % 8 or w % 8:
        raise SystemExit(f"[film] resolution must be a multiple of 8 (got {w}x{h})")
    odd = [k for k in keys if k["size"] != (h, w)]
    if odd:
        print(f"[film] WARNING: {len(odd)} of {len(keys)} keyframes were made at "
              f"a different resolution than this film's {w}x{h}. They are "
              f"re-rendered at {w}x{h} from a differently-shaped noise draw, so "
              f"those keyframes will NOT be reproduced exactly -- the film just "
              f"passes near them:", flush=True)
        for k in odd:
            print(f"[film]     {k['rel']}  ({k['size'][1]}x{k['size'][0]})", flush=True)
        print(f"[film]   fix: drop them, or pass --width/--height to film at "
              f"their size instead.", flush=True)

    started = [k for k in keys if k["meta"].get("init_image")]
    if started:
        print(f"[film] NOTE: {len(started)} keyframe(s) were generated from an "
              f"init image; their exact pixels can't be reproduced from the "
              f"seed alone, so those endpoints will be close, not identical.",
              flush=True)

    plan = frame_plan(len(keys), args.frames_per, args.easing)
    print(f"[film] {args.name}: {len(keys)} keyframes ({backend}), "
          f"{args.frames_per}/transition -> {len(plan)} frames @ {args.fps}fps "
          f"= {len(plan) / args.fps:.1f}s", flush=True)
    print(f"[film] interp={args.interp} easing={args.easing} "
          f"noise={'fixed' if args.fixed_noise else f'window {args.noise_window}'} "
          f"{h}x{w} steps={steps} guidance={guidance} scheduler={scheduler}",
          flush=True)

    film_dir = OUT / "films" / args.name
    base_dir = film_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from semantic_anarchy.backend import make_backend
    from semantic_anarchy.pipeline import set_scheduler

    model_id = ckpt = None
    if ckpt_or_model and ckpt_or_model != "(default)":
        if ckpt_or_model.endswith((".safetensors", ".ckpt")):
            ckpt = ckpt_or_model
        else:
            model_id = ckpt_or_model
    print(f"[film] loading {backend} {ckpt_or_model or '(default)'} ...", flush=True)
    be = make_backend(backend, model_id=model_id, ckpt=ckpt)
    pipe = be.model.pipe
    set_scheduler(pipe, scheduler)
    device = be.model.device
    dtype = next(pipe.unet.parameters()).dtype

    def init_latent(seed) -> "torch.Tensor":
        """EXACT reconstruction of the latent the generator drew for image_seed.

        Mirrors diffusers' ``randn_tensor``: sd15/sd2 draw on the compute device,
        sdxl draws on the CPU generator and moves -- match or the noise differs.
        """
        f = pipe.vae_scale_factor
        shape = (1, pipe.unet.config.in_channels, h // f, w // f)
        gen_device = "cpu" if backend == "sdxl" else device
        g = torch.Generator(device=gen_device).manual_seed(int(seed))
        return torch.randn(shape, generator=g, device=gen_device,
                           dtype=dtype).to(device)

    seeds = [k["meta"].get("image_seed") for k in keys]
    if any(s is None for s in seeds):
        print("[film] WARNING: some keyframes have no image_seed; those "
              "endpoints fall back to seed 0 and won't match exactly.", flush=True)
    lats = [init_latent(s if s is not None else 0).float().cpu().numpy()
            for s in seeds]

    negatives = (_sdxl_negatives(backend, m0, guidance, device, dtype)
                 if backend == "sdxl"
                 else _sd_negatives(be.model, m0, guidance, device, dtype))

    def render(cond: dict, latent: np.ndarray):
        kwargs = dict(latents=torch.from_numpy(latent).to(device, dtype),
                      num_inference_steps=steps, guidance_scale=guidance,
                      height=h, width=w)
        if backend == "sdxl":
            kwargs["prompt_embeds"] = _t(cond["prompt_embeds"], 3, device, dtype)
            kwargs["pooled_prompt_embeds"] = _t(cond["pooled"], 2, device, dtype)
            kwargs.update(negatives)
        else:
            kwargs["prompt_embeds"] = _t(cond["embeds"], 3, device, dtype)
            kwargs.update(negatives)
        return pipe(**kwargs).images[0]

    print(f"[film] rendering {len(plan)} base frames ...", flush=True)
    t0 = time.time()
    bext = frame_ext(base_dir)
    for fi, (s, t) in enumerate(plan):
        if args.resume and (base_dir / f"frame_{fi:04d}{bext}").is_file():
            continue
        cond = {k: interpolate(keys[s]["cond"][k], keys[s + 1]["cond"][k],
                               t, args.interp)
                for k in FILM_TENSORS[backend]}
        if args.fixed_noise:
            lat = lats[0]
        else:
            # The noise ALWAYS slerps regardless of --interp: lerping two
            # gaussian latents shrinks their variance, and an under-noised
            # start renders flat, washed-out middles.
            tn = noise_t(t, args.noise_window, args.easing)
            lat = interpolate(lats[s], lats[s + 1], tn, "slerp")
        save_image(render(cond, lat), base_dir / f"frame_{fi:04d}{bext}")
        done, left = fi + 1, len(plan) - fi - 1
        eta = (time.time() - t0) / done * left
        print(f"[film]   base {done}/{len(plan)}  (eta {eta / 60:.1f}m)", flush=True)

    del be, pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    render_dir = base_dir
    if args.refine == "flux":
        render_dir = _flux_pass(args, film_dir, base_dir, len(plan), h, w)

    mp4 = film_dir / f"{args.name}.mp4"
    cmd = [ffmpeg, "-y", "-framerate", str(args.fps),
           "-i", str(render_dir / f"frame_%04d{frame_ext(render_dir)}"),
           # yuv420p needs even dimensions; refine scales can land on odd ones.
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           *codec_args, "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(mp4)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("\n".join(proc.stderr.strip().splitlines()[-15:]), flush=True)
        raise SystemExit(f"[film] ffmpeg failed (rc={proc.returncode})")
    (film_dir / "film.json").write_text(json.dumps({
        "keyframes": [k["rel"] for k in keys], "frames": len(plan),
        "fps": args.fps, "frames_per": args.frames_per, "backend": backend,
        "interp": args.interp, "easing": args.easing, "loop": args.loop,
        "noise_window": (None if args.fixed_noise else args.noise_window),
        "fixed_noise": args.fixed_noise, "refine": args.refine,
        "film_seed": args.film_seed, "steps": steps, "guidance": guidance,
        "scheduler": scheduler, "height": h, "width": w,
        "duration": round(len(plan) / args.fps, 2),
        # Which keyframes could NOT be reproduced exactly at this size.
        "offsize_keyframes": [k["rel"] for k in odd],
    }, indent=2))
    print(f"[film] saved {mp4}  ({len(plan)} frames, "
          f"{len(plan) / args.fps:.1f}s)", flush=True)
    print(f"[film] watch: /img?path=films/{args.name}/{args.name}.mp4", flush=True)
    return 0


def _t(a, want: int, device, dtype):
    """numpy -> torch, padded with leading batch dims up to ``want``."""
    import torch
    a = np.asarray(a, dtype=np.float32)
    while a.ndim < want:
        a = a[None]
    return torch.from_numpy(a).to(device, dtype)


def _sd_negatives(model, meta: dict, guidance: float, device, dtype) -> dict:
    """CFG negative for sd15/sd2, matching how the keyframes themselves were made.

    generate.py defaults sd15 to ``neg_mode=text`` — the house negative prompt —
    so a film that left it out would render every frame against a *different*
    negative than its own endpoints. Passing nothing here would make diffusers
    silently encode the empty prompt instead.
    """
    if guidance <= 1.0:
        return {}
    mode = meta.get("neg_mode") or "text"
    if mode == "empty":
        return {"negative_prompt_embeds": _t(model.uncond_embedding(), 3, device, dtype)}
    if mode != "text":     # mean/zeros are sdxl-only knobs; fall back to the default
        return {}
    return {"negative_prompt_embeds": _t(model.negative_embedding(), 3, device, dtype)}


def _sdxl_negatives(backend: str, meta: dict, guidance: float, device, dtype) -> dict:
    """CFG negatives for sdxl, matching how the keyframes themselves were made.

    generate.py defaults sdxl to ``neg_mode=mean`` (push away from the corpus
    average), so reproducing a keyframe exactly means reusing that negative.
    Falls back to the pipeline's own empty-prompt negative if the distribution
    isn't on disk.
    """
    if guidance <= 1.0:
        return {}
    mode = meta.get("neg_mode") or "mean"
    if mode == "empty":
        return {}
    try:
        from semantic_anarchy.backend import dist_backend
        prefix = f"{meta.get('dist') or 'outputs/dist'}_sdxl"
        dists = dist_backend("sdxl").load_dists(prefix)
        neg = ({k: np.zeros_like(d.mean) for k, d in dists.items()}
               if mode == "zeros" else {k: d.mean for k, d in dists.items()})
        return {
            "negative_prompt_embeds": _t(neg["prompt_embeds"], 3, device, dtype),
            "negative_pooled_prompt_embeds": _t(neg["pooled"], 2, device, dtype),
        }
    except Exception as exc:                                   # noqa: BLE001
        print(f"[film] WARNING: neg_mode={mode} unavailable ({exc!r}); "
              f"using the empty-prompt negative instead", flush=True)
        return {}


def _flux_pass(args, film_dir: Path, base_dir: Path, n_frames: int,
               h: int, w: int) -> Path:
    """Reference-conditioned FLUX.2-klein upscale of every base frame.

    ONE seed and ONE prompt for the whole film so the reinterpretation doesn't
    flicker between frames.
    """
    import torch
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
    fext, bext = frame_ext(flux_dir), frame_ext(base_dir)
    for fi in range(n_frames):
        if args.resume and (flux_dir / f"frame_{fi:04d}{fext}").is_file():
            continue
        src = Image.open(base_dir / f"frame_{fi:04d}{bext}").convert("RGB")
        gen = torch.Generator(device="cpu").manual_seed(args.film_seed)
        out = fpipe(prompt=FAITHFUL_PROMPT, image=[src],
                    height=th, width=tw, num_inference_steps=28,
                    guidance_scale=4.0, generator=gen).images[0]
        save_image(out, flux_dir / f"frame_{fi:04d}{fext}")
        print(f"[film]   flux {fi + 1}/{n_frames}", flush=True)
    return flux_dir


if __name__ == "__main__":
    raise SystemExit(main())
