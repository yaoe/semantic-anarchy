#!/usr/bin/env python3
"""Animate an image with Wan 2.1 I2V-14B -- a real video model, no prompt.

The heavyweight sibling of ``animate.py``. AnimateDiff rides the SD1.5 UNet and
therefore eats this project's own ``(77, 768)`` conditioning directly, but it is
capped at a 16-frame window and inherits SD1.5's grasp of motion. Wan is a
purpose-built video transformer: 81 frames at 16fps (~5s), real camera and
subject motion, temporal coherence SD1.5 cannot reach.

**Promptless by default.** Wan I2V conditions on the image itself (VAE-encoded
first frame + CLIP vision), with text as a secondary steer -- so an empty prompt
is a legitimate mode, not a degenerate one: the picture is the instruction.
``--prompt`` exists for comparison, and is the one place a human sentence can
enter. (The thesis-true version -- mining a UMT5 conditioning distribution and
sampling it the way ``mine_distribution.py`` does for sd15 -- is a later step;
``WanImageToVideoPipeline`` takes ``prompt_embeds``, so the door is open.)

Runs in **.venv-flux** (diffusers >= 0.35), like flux2/krea2.

    .venv-flux/bin/python scripts/animate_wan.py --src generated/X.jpg
    .venv-flux/bin/python scripts/animate_wan.py --src X.jpg --frames 49 --quant nf4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set before torch is imported (and before the dashboard, which just inherits its
# own environment, gets a say): the quantized transformer plus a temporal VAE
# fragment the allocator badly, and an expandable heap is the difference between
# decoding the clip and OOMing on the last step.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from semantic_anarchy.io_utils import find_image, image_ext, save_image  # noqa: E402
from semantic_anarchy.video import encode_mp4, smooth_mp4  # noqa: E402

OUT = Path("outputs")
FILMS = OUT / "films"

DEFAULT_MODEL = os.environ.get("SA_WAN_MODEL", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
#: Wan 2.1 renders at 16fps; 81 frames is the ~5s clip it was trained for.
NATIVE_FPS = 16
#: The VAE compresses time 4:1, so a frame count must be 4k+1 to land whole.
FRAME_MULTIPLE = 4
#: Wan's own reference sizing (the official 2.2 Space): sides rounded to 16 and
#: clamped to [480, 832], with a square image going to 640 rather than being
#: stretched toward the 480x832 landscape box.
WAN_MULTIPLE, WAN_MIN_DIM, WAN_MAX_DIM, WAN_SQUARE = 16, 480, 832, 640
#: The official Wan negative prompt, verbatim. It is Chinese because the model
#: was trained that way -- an English paraphrase is not the same vector, and
#: "static / still" is the part that actually buys motion.
WAN_NEGATIVE = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, "
    "整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, "
    "画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, "
    "静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走")
#: A MOTION-only instruction: it says how to move, never what the thing is, so
#: the picture remains the only description of content. Pass --prompt "" for
#: strictly nothing.
WAN_MOTION_PROMPT = "make this image come alive, cinematic motion, smooth animation"
#: The Lightning (lightx2v) step-distillation LoRA, exactly as Wan's own Space
#: loads it: the SAME file into both experts, then fused at DIFFERENT scales --
#: 3.0 into the high-noise transformer, 1.0 into the low-noise one.
LIGHTNING_REPO = "Kijai/WanVideo_comfy"
LIGHTNING_FILE = ("Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16"
                  ".safetensors")
LIGHTNING_SCALES = {"transformer": 3.0, "transformer_2": 1.0}
#: What the distillation buys: 6 steps instead of 30.
LIGHTNING_STEPS = 6


def apply_lightning(pipe, quantized: bool) -> str:
    """Load the step-distillation LoRA into both experts of the MoE.

    Fusing is the reference path, but it merges LoRA deltas into the base
    weights -- impossible against NF4-quantized params. When quantized we keep
    the adapters live instead and fold the reference's per-expert fuse scales
    into the adapter weights, which is the same arithmetic applied at runtime
    rather than baked in.
    """
    pipe.load_lora_weights(LIGHTNING_REPO, weight_name=LIGHTNING_FILE,
                           adapter_name="lightx2v")
    pipe.load_lora_weights(LIGHTNING_REPO, weight_name=LIGHTNING_FILE,
                           adapter_name="lightx2v_2", load_into_transformer_2=True)
    if not quantized:
        pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])
        pipe.fuse_lora(adapter_names=["lightx2v"],
                       lora_scale=LIGHTNING_SCALES["transformer"],
                       components=["transformer"])
        pipe.fuse_lora(adapter_names=["lightx2v_2"],
                       lora_scale=LIGHTNING_SCALES["transformer_2"],
                       components=["transformer_2"])
        pipe.unload_lora_weights()
        return "fused"
    pipe.set_adapters(["lightx2v", "lightx2v_2"],
                      adapter_weights=[LIGHTNING_SCALES["transformer"],
                                       LIGHTNING_SCALES["transformer_2"]])
    return "adapters (unfused: NF4 base)"


def quantized_components(model_id: str) -> list[str]:
    """Which submodules to quantize.

    Wan 2.2's A14B is a mixture of experts: ``transformer`` handles the noisy
    half of the schedule and ``transformer_2`` the clean half, switched at
    ``boundary_ratio``. BOTH are 14B and both live on the card, so quantizing
    only the first leaves 28GB of bf16 weights to find room for.
    """
    parts = ["transformer", "text_encoder"]
    if "A14B" in model_id:
        parts.insert(1, "transformer_2")
    return parts


def frame_count(n: int) -> int:
    """Snap to the 4k+1 grid the temporal VAE needs (81, 77, 49, 33 ...)."""
    return max(FRAME_MULTIPLE + 1,
               ((n - 1) // FRAME_MULTIPLE) * FRAME_MULTIPLE + 1)


def fit_dims(w: int, h: int) -> tuple[int, int]:
    """Wan's reference sizing: aspect preserved, 16-aligned, clamped, square->640."""
    if w == h:
        return WAN_SQUARE, WAN_SQUARE
    scale = min(WAN_MAX_DIM / max(w, h), 1.0)
    tw, th = w * scale, h * scale
    if min(tw, th) < WAN_MIN_DIM:                  # too thin after the cap: widen
        up = WAN_MIN_DIM / min(tw, th)
        tw, th = min(tw * up, WAN_MAX_DIM), min(th * up, WAN_MAX_DIM)
    out = []
    for v in (tw, th):
        v = round(v / WAN_MULTIPLE) * WAN_MULTIPLE
        out.append(int(max(WAN_MIN_DIM, min(WAN_MAX_DIM, v))))
    return out[0], out[1]


def load_fp8_experts(model_id: str):
    """Both MoE experts at fp8, quantized AS THEY LOAD.

    This is the Space's precision (Float8DynamicActivationFloat8Weight) on a
    24GB card, which needs two tricks it doesn't:

    * **Quantize during load.** Each expert is ~13.3B params = 26.6GB in bf16,
      more than the card holds and most of this box's 30GB of RAM. Passing a
      TorchAoConfig to ``from_pretrained`` converts shard by shard, so the full
      bf16 model is never materialised -- peak is one shard, not the model.
    * **One expert resident.** ``WanImageToVideoPipeline`` picks
      ``transformer`` above ``boundary_timestep`` and ``transformer_2`` below,
      so they never run in the same step. Expert 1 starts on the GPU and expert
      2 in host RAM; cpu-offload swaps them at the boundary.
    """
    import torch
    from diffusers import WanTransformer3DModel, TorchAoConfig
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

    cfg = TorchAoConfig(Float8DynamicActivationFloat8WeightConfig())
    out = []
    for sub in ("transformer", "transformer_2"):
        t0 = time.time()
        out.append(WanTransformer3DModel.from_pretrained(
            model_id, subfolder=sub, torch_dtype=torch.bfloat16,
            quantization_config=cfg))
        print(f"[wan]   {sub} fp8 in {time.time() - t0:.0f}s", flush=True)
    return out


def quant_kwargs(mode: str, free_gb: float, model_id: str = "") -> dict:
    """How to squeeze a 14B transformer + an 11GB text encoder onto this card.

    bf16 needs ~28GB for the transformer alone, and ``enable_model_cpu_offload``
    does not save us: this box has 30GB of system RAM, so the weights have
    nowhere to be staged either. Quantized-on-GPU is the only shape that fits,
    which is the same conclusion ``_flow_pipeline_kwargs`` reached for klein-9B.
    """
    import torch
    from diffusers.quantizers import PipelineQuantizationConfig

    if mode == "auto":
        # Measured on this box: int8 puts ~14GB of transformer + ~5.5GB of
        # UMT5-XXL + an fp32 VAE + CLIP vision on the card and OOMs during load
        # with 23.5GB free. It needs real headroom, not a card that merely fits
        # the weights; nf4 is ~11GB all-in and leaves room to actually denoise.
        mode = "8bit" if free_gb >= 32 else "nf4"
    if mode == "off":
        return {"torch_dtype": torch.bfloat16, "_quant": "off"}
    parts = quantized_components(model_id)
    if mode == "8bit":
        cfg = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=parts)
    else:
        cfg = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=parts)
    return {"torch_dtype": torch.bfloat16, "quantization_config": cfg, "_quant": mode}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="image to animate (any backend, any size)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--prompt", default=WAN_MOTION_PROMPT,
                    help="MOTION only, never content -- the image stays the sole "
                         'description of what the thing is. --prompt "" for none')
    ap.add_argument("--negative", default=WAN_NEGATIVE,
                    help="the official Wan negative (Chinese, as trained)")
    ap.add_argument("--frames", type=int, default=81, help="snapped to 4k+1")
    ap.add_argument("--fps", type=int, default=NATIVE_FPS)
    ap.add_argument("--smooth-fps", type=int, default=0,
                    help="also write <name>_smooth.mp4 at this fps (minterpolate)")
    ap.add_argument("--steps", type=int, default=None,
                    help=f"default: {LIGHTNING_STEPS} with Lightning, 30 without")
    ap.add_argument("--lightning", dest="lightning", action="store_true", default=True,
                    help="step-distillation LoRA (default ON for A14B): 6 steps "
                         "instead of 30, ~5x faster")
    ap.add_argument("--no-lightning", dest="lightning", action="store_false")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="Wan's own reference uses 1.0, i.e. CFG OFF. Higher only "
                         "makes sense with a CONTENT prompt to extrapolate toward; "
                         "with a motion-only prompt it overpowers the image and "
                         "the first frame stops resembling the source")
    ap.add_argument("--guidance-2", type=float, default=None,
                    help="A14B's second expert (default: same as --guidance)")
    ap.add_argument("--seed", type=int, default=None,
                    help="omit for a FRESH RANDOM seed each run (recorded in "
                         "film.json); pass one to reproduce a clip")
    ap.add_argument("--size", type=int, default=None,
                    help="force the render size (16-aligned, clamped to Wan's "
                         f"{WAN_MIN_DIM}-{WAN_MAX_DIM} band). Matching the source's "
                         "own size removes the resample, so frame 1 is a VAE "
                         "round-trip of your picture and nothing else")
    ap.add_argument("--quant", default="auto",
                    choices=["auto", "fp8", "8bit", "nf4", "off"],
                    help="fp8 matches the reference implementation's precision "
                         "(8 bits/weight vs nf4's 4) at the cost of swapping the "
                         "inactive expert out to host RAM")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args(argv)

    src = find_image(OUT / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    if src is None or not src.is_file():
        raise SystemExit(f"[wan] source not found: {args.src}")
    meta = {}
    j = src.with_suffix(".json")
    if j.is_file():
        try:
            meta = json.loads(j.read_text())
        except ValueError:
            meta = {}
    name = args.name or f"{src.stem}_wan"
    frames = frame_count(args.frames)
    # The LoRA is trained for the 14B I2V experts; nothing else has them.
    lightning = args.lightning and "A14B" in args.model
    steps = args.steps if args.steps is not None else (
        LIGHTNING_STEPS if lightning else 30)
    # NOT the source's image_seed: an upscale's sidecar carries no image_seed at
    # all, so that fallback quietly resolved to 0 for every clip made from one --
    # same source, same seed, byte-identical video every time.
    seed = (args.seed if args.seed is not None
            else int.from_bytes(os.urandom(4), "little"))

    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video  # noqa: F401  (kept: parity w/ docs)

    free_gb = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0
    kw = quant_kwargs(args.quant, free_gb, args.model) if args.quant != "fp8" else {"_quant": "fp8"}
    quant = kw.pop("_quant")
    print(f"[wan] {args.model}", flush=True)
    print(f"[wan] {free_gb:.1f}GB VRAM free -> quantization: {quant}"
          + (f" on {', '.join(quantized_components(args.model))}" if quant != "off" else ""),
          flush=True)
    if quant == "off" and free_gb < 30:
        print("[wan] WARNING: bf16 needs ~28GB for the transformer alone; "
              "expect an OOM. Use --quant 8bit or nf4.", flush=True)

    t0 = time.time()
    # The VAE stays in fp32: Wan's temporal VAE is the one component that
    # visibly degrades in half precision (banding in flat gradients).
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder="vae",
                                           torch_dtype=torch.float32)
    if quant == "fp8":
        t1, t2 = load_fp8_experts(args.model)
        pipe = WanImageToVideoPipeline.from_pretrained(
            args.model, vae=vae, transformer=t1, transformer_2=t2,
            torch_dtype=torch.bfloat16)
        # Only the expert the current timestep needs is on the card.
        pipe.enable_model_cpu_offload()
    else:
        pipe = WanImageToVideoPipeline.from_pretrained(args.model, vae=vae, **kw)
        if quant == "off":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    # Denoising fits comfortably; the DECODE is what OOMs. Wan's VAE is temporal,
    # so it reconstructs every frame of the clip in one pass -- 49 frames of
    # fp32 at 624^2 wants more headroom than the quantized transformer ever did.
    # Tiling + slicing bring it back into range with no quality cost worth the
    # bytes (the blend regions overlap).
    for meth in ("enable_tiling", "enable_slicing"):
        try:
            getattr(pipe.vae, meth)()
        except Exception as exc:                               # noqa: BLE001
            print(f"[wan] vae.{meth} unavailable ({exc!r})", flush=True)
    print(f"[wan] loaded in {time.time() - t0:.1f}s", flush=True)

    img = Image.open(src).convert("RGB")
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    if args.size:
        n = int(max(WAN_MIN_DIM, min(WAN_MAX_DIM,
                    round(args.size / WAN_MULTIPLE) * WAN_MULTIPLE)))
        w, h = (n, n) if img.width == img.height else fit_dims(img.width, img.height)
    else:
        w, h = fit_dims(img.width, img.height)
    img = img.resize((w, h))
    print(f"[wan] {src.name}  {img.width}x{img.height} · {frames} frames @ {args.fps}fps "
          f"(~{frames / args.fps:.1f}s) · steps={steps} · guidance={args.guidance} "
          f"· seed={seed}", flush=True)
    if not args.prompt:
        print("[wan] promptless: the image is the only instruction", flush=True)

    # Encode once, then DROP the text encoder before denoising. UMT5-XXL is
    # ~3GB even at NF4 and is dead weight from here on -- which is the difference
    # between fitting and not on A14B, where two 14B experts are already resident.
    # Passing embeds rather than text is also the promptless shape: an empty
    # prompt is encoded once and never mentioned again.
    with torch.no_grad():
        pe, npe = pipe.encode_prompt(
            prompt=args.prompt or "", negative_prompt=args.negative,
            do_classifier_free_guidance=args.guidance > 1.0, device="cuda")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[wan] text encoder freed ({torch.cuda.memory_allocated() / 2**30:.1f}GB "
          f"resident before denoising)", flush=True)

    # Lightning goes on LAST, once UMT5 has vacated: the adapters add bf16 LoRA
    # weights on top of two resident 14B experts, and loading them alongside the
    # text encoder is the difference between 23.5GB and an OOM.
    if lightning:
        t1 = time.time()
        mode = apply_lightning(pipe, quant != "off")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[wan] Lightning LoRA: {mode} — "
              f"{LIGHTNING_SCALES['transformer']}x / "
              f"{LIGHTNING_SCALES['transformer_2']}x, {time.time() - t1:.1f}s "
              f"({torch.cuda.memory_allocated() / 2**30:.1f}GB resident)", flush=True)

    t0 = time.time()
    # output_type defaults to "np" on Wan (AnimateDiff hands back PIL), and
    # io_utils.save_image wants an image, not an (F, H, W, C) float array.
    gkw = {"guidance_scale": args.guidance}
    if "A14B" in args.model:
        # The MoE takes one guidance per expert; the reference sets both.
        gkw["guidance_scale_2"] = (args.guidance if args.guidance_2 is None
                                   else args.guidance_2)
    out = pipe(image=img, prompt_embeds=pe, negative_prompt_embeds=npe,
               height=h, width=w, num_frames=frames,
               num_inference_steps=steps, **gkw,
               output_type="pil",
               generator=torch.Generator("cuda").manual_seed(seed))
    result = out.frames[0]
    print(f"[wan] {len(result)} frames in {time.time() - t0:.1f}s "
          f"(peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB)", flush=True)

    film_dir = FILMS / name
    frame_dir = film_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    ext = image_ext()
    for i, f in enumerate(result):
        if not hasattr(f, "mode"):        # belt and braces if a future diffusers
            import numpy as _np           # ignores output_type and returns arrays
            f = Image.fromarray((_np.asarray(f) * 255).round().astype("uint8"))
        save_image(f, frame_dir / f"frame_{i:04d}{ext}")

    mp4 = film_dir / f"{name}.mp4"
    encode_mp4(str(frame_dir / f"frame_%04d{ext}"), mp4, args.fps, args.crf, "wan")
    if args.smooth_fps and args.smooth_fps > args.fps:
        s = smooth_mp4(mp4, args.smooth_fps, args.crf, "wan")
        print(f"[wan] saved {s} ({args.fps} -> {args.smooth_fps}fps)", flush=True)
    (film_dir / "film.json").write_text(json.dumps({
        "kind": "animate", "engine": "wan-i2v", "model": args.model,
        "source": str(src.relative_to(OUT)) if OUT in src.parents else str(src),
        "prompt": args.prompt, "negative": args.negative, "promptless": not args.prompt,
        "frames": len(result), "fps": args.fps, "steps": steps,
        "guidance": args.guidance, "seed": seed, "quant": quant,
        "steps_actual": steps, "lightning": lightning,
        "height": h, "width": w, "smooth_fps": args.smooth_fps or None,
        "created": time.time(),
    }, indent=2))

    if not args.keep_frames:
        for f in frame_dir.iterdir():
            f.unlink()
        frame_dir.rmdir()

    print(f"[wan] saved {mp4}", flush=True)
    print(f"[wan] watch: /img?path=films/{name}/{name}.mp4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
