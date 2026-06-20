"""Shared argparse wiring so every script speaks the SAME --backend language.

Both backends carry an identical knob set; only --ckpt (sd15 single-file) vs
--model (sdxl repo/dir) and the per-family defaults differ. These helpers keep
that in one place.
"""

from __future__ import annotations

import argparse

from .backend import BACKEND_DEFAULTS, make_backend


def add_backend_args(p: argparse.ArgumentParser, *, with_steps=True) -> None:
    """Add --backend + model selection + shared sampler/generation knobs."""
    p.add_argument("--backend", default="sd15", choices=["sd15", "sdxl"],
                   help="Which model runs the identical drift method.")
    p.add_argument("--model", default=None,
                   help="HF id or diffusers dir. sd15 default "
                        "runwayml/stable-diffusion-v1-5; sdxl default "
                        "stabilityai/sdxl-turbo. (--ckpt overrides for single-file.)")
    p.add_argument("--ckpt", default=None,
                   help="single-file .ckpt/.safetensors (from_single_file).")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto if unset)")

    # sampler / drift knobs (forwarded verbatim to EmbeddingDistribution.sample)
    p.add_argument("--sampler", default="diagonal",
                   choices=["diagonal", "pca", "blend", "hybrid"],
                   help="anarchy<->coherence axis. diagonal=independent coords; "
                        "pca=on the corpus manifold (temp>1 extrapolates OUTSIDE "
                        "the hull); blend=interpolate; hybrid=SLERP two real concepts.")
    p.add_argument("--coherence", type=float, default=0.5,
                   help="blend lambda in [0,1]: 1=pure pca, 0=pure diagonal.")
    p.add_argument("--components", type=int, default=None,
                   help="pca/blend: use only the top N principal axes (default all).")
    p.add_argument("--truncation", type=float, default=None,
                   help="resample coords beyond this many sigma (typical-set trick).")

    # generation knobs (per-backend defaults filled by resolve_gen_defaults)
    if with_steps:
        p.add_argument("--steps", type=int, default=None,
                       help="denoising steps (default: sd15=30, sdxl-turbo=1).")
    p.add_argument("--guidance", type=float, default=None,
                   help="CFG scale (default: sd15=7.5, sdxl-turbo=0). >1 enables CFG.")
    p.add_argument("--neg-mode", default=None, choices=["mean", "empty", "zeros"],
                   help="(sdxl CFG) negative conditioning. mean=corpus mean (push "
                        "away from average toward the sample). Default mean (sdxl) "
                        "/ empty (sd15).")


def resolve_gen_defaults(args) -> None:
    """Fill steps/guidance/neg_mode from the backend family if left unset."""
    d = BACKEND_DEFAULTS[args.backend]
    if getattr(args, "steps", None) is None:
        args.steps = d["steps"]
    if getattr(args, "guidance", None) is None:
        args.guidance = d["guidance"]
    if getattr(args, "neg_mode", None) is None:
        args.neg_mode = "mean" if args.backend == "sdxl" else "empty"
    if getattr(args, "height", None) in (None, 0):
        args.height = d["height"]
    if getattr(args, "width", None) in (None, 0):
        args.width = d["width"]


def load_backend(args):
    """Construct the backend from parsed args (touches torch/diffusers)."""
    return make_backend(args.backend, model_id=args.model, ckpt=args.ckpt,
                        device=args.device)


def dist_prefix(args, base: str) -> str:
    """Backend-namespaced distribution prefix so sd15/sdxl never clash.

    sd15 keeps the bare ``<base>`` (original layout); sdxl appends ``_sdxl``.
    """
    return base if args.backend == "sd15" else f"{base}_sdxl"
