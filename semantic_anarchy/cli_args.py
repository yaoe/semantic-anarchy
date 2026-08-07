"""Shared argparse wiring so every script speaks the SAME --backend language.

Both backends carry an identical knob set; only --ckpt (sd15 single-file) vs
--model (sdxl repo/dir) and the per-family defaults differ. These helpers keep
that in one place.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .backend import BACKEND_DEFAULTS, make_backend
from .dist_paths import backend_prefix


def add_backend_args(p: argparse.ArgumentParser, *, with_steps=True) -> None:
    """Add --backend + model selection + shared sampler/generation knobs."""
    p.add_argument("--backend", default="sd15",
                   choices=["sd15", "sd2", "sdxl", "flux2", "krea2"],
                   help="Which model runs the identical drift method.")
    p.add_argument("--model", default=None,
                   help="HF id or diffusers dir. sd15 default "
                        "runwayml/stable-diffusion-v1-5; sd2 default "
                        "stabilityai/stable-diffusion-2-1; sdxl default "
                        "stabilityai/sdxl-turbo. (--ckpt overrides for single-file.)")
    p.add_argument("--ckpt", default=None,
                   help="single-file .ckpt/.safetensors (from_single_file).")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto if unset)")
    p.add_argument("--scheduler", default="default",
                   choices=["default", "ddim", "euler", "euler_a", "dpm"],
                   help="diffusion sampler. ddim = smooth classic sampler used for "
                        "the nicer high-step renders.")

    # sampler / drift knobs (forwarded verbatim to EmbeddingDistribution.sample)
    p.add_argument("--sampler", default="diagonal",
                   choices=["diagonal", "pca", "blend", "hybrid", "split"],
                   help="anarchy<->coherence axis. diagonal=independent coords; "
                        "pca=on the corpus manifold (temp>1 extrapolates OUTSIDE "
                        "the hull); blend=interpolate; hybrid=SLERP two real "
                        "concepts; split=diagonal with separate on/off-manifold "
                        "temperatures (--temp-on/--temp-off).")
    p.add_argument("--coherence", type=float, default=0.5,
                   help="blend lambda in [0,1]: 1=pure pca, 0=pure diagonal.")
    p.add_argument("--components", type=int, default=None,
                   help="pca/blend: use N principal axes starting at --comp-lo (default all).")
    p.add_argument("--comp-lo", type=int, default=0,
                   help="pca/blend: first principal axis (0=dominant/generic; raise to "
                        "skip the tasteful axes and ride weird mid/minor ones).")
    p.add_argument("--equalize", action="store_true",
                   help="pca/blend: express every selected axis at equal (RMS) strength "
                        "so minor 'weird' axes actually register.")
    p.add_argument("--truncation", type=float, default=None,
                   help="resample coords beyond this many sigma (typical-set "
                        "trick). OFF by default and meant to stay that way -- the "
                        "real corpus has 5-8 sigma events, so clipping at 2-3 "
                        "removes behaviour that is actually there.")

    # ---- the corpus-autopsy corrections (all default to the old behaviour) ---
    p.add_argument("--rho", type=float, default=0.0,
                   help="row coherence in [0,1] for the diagonal draw: how much "
                        "of each deviation is SHARED across the 77 token "
                        "positions. 0 (default) = the historical static, ~0.65 = "
                        "corpus-like, 1 = one deviation smeared through the whole "
                        "sentence. Marginals stay exact either way.")
    p.add_argument("--length-mode", default="off",
                   choices=["off", "corpus", "fixed"],
                   help="length-conditional sampling (needs a fit mined with the "
                        "length split). corpus=draw each sample's content length "
                        "from the corpus histogram; fixed=pin every sample to "
                        "--length. Off = the single pooled Gaussian, which peaks "
                        "in the gap between the content and padding lobes.")
    p.add_argument("--length", type=int, default=None, metavar="N",
                   help="content length in tokens for --length-mode fixed "
                        "('sample me a 60-token image'). Passing it implies "
                        "--length-mode fixed. Prompt length was PC1/PC2 in "
                        "disguise, so this is the corpus's biggest semantic dial.")
    p.add_argument("--empirical-head", type=int, default=0, metavar="K",
                   help="pca/blend: draw the first K principal coefficients from "
                        "the corpus's own CDF instead of N(0,1). PC1 is bimodal, "
                        "so the Gaussian's densest mass lands in an empty gap. "
                        "K=2 fixes the two length-correlated axes.")
    p.add_argument("--temp-on", type=float, default=1.0,
                   help="--sampler split: temperature of the ON-manifold (PCA "
                        "subspace) half of the deviation.")
    p.add_argument("--temp-off", type=float, default=1.0,
                   help="--sampler split: temperature of the OFF-manifold "
                        "(orthogonal) half of the deviation.")

    # generation knobs (per-backend defaults filled by resolve_gen_defaults)
    if with_steps:
        p.add_argument("--steps", type=int, default=None,
                       help="denoising steps (default: sd15=30, sdxl-turbo=1).")
    p.add_argument("--guidance", type=float, default=None,
                   help="CFG scale (default: sd15=7.5, sdxl-turbo=0). >1 enables CFG.")
    p.add_argument("--neg-mode", default=None,
                   choices=["text", "mean", "empty", "zeros"],
                   help="CFG negative conditioning. text=the house SD1.5 negative "
                        "prompt, mean=corpus mean (push away from average toward "
                        "the sample), empty=empty prompt, zeros=zero tensor. "
                        "Default text (sd15) / mean (sdxl) / empty (others).")
    p.add_argument("--negative", default=None, metavar="TEXT",
                   help="the negative prompt text itself (sd15/sd2, --neg-mode "
                        "text). Default: the house SD1.5 negative, "
                        "pipeline.SD15_NEGATIVE_PROMPT. Pass '' for none.")


def add_experiment_args(p: argparse.ArgumentParser) -> None:
    """Add ``--experiment`` / ``--hypothesis`` to an image-producing script.

    Only the scripts that write a per-image ``.json`` sidecar take these — the
    id has to reach the sidecar to reach a label, and a contact sheet has no
    sidecar to put it in.
    """
    p.add_argument("--experiment", default=None, metavar="ID",
                   help="tag every image of this batch with an experiment id "
                        "(e.g. E01-length). Lands in each .json sidecar, gets "
                        "echoed into every label, and writes a manifest to "
                        "outputs/experiments/<id>.json.")
    p.add_argument("--hypothesis", default=None, metavar="TEXT",
                   help="one falsifiable sentence about what this batch should "
                        "show; stored in the experiment manifest.")


def record_experiment(args, **extra):
    """Slug ``--experiment``, write/extend its manifest, return the id (or None).

    Called *after* the batch seed is resolved so the manifest records the seed
    the run actually used, not the ``None`` that meant "draw one".
    """
    from .labels import clean_experiment_id, used_seed_panel, write_manifest

    exp = clean_experiment_id(getattr(args, "experiment", None))
    if not exp:
        return None
    data = {
        "argv": sys.argv[1:],
        "backend": getattr(args, "backend", None),
        "model": getattr(args, "ckpt", None) or getattr(args, "model", None),
        "dist": str(getattr(args, "dist", "") or "") or None,
        "hypothesis": getattr(args, "hypothesis", None),
        "seed": getattr(args, "seed", None),
        "n": getattr(args, "n", None),
        **extra,
    }
    data["seed_panel"] = used_seed_panel(data.get("seed"), data.get("n"))
    write_manifest(exp, data)
    return exp


def sampler_kwargs(args, lengths=None) -> dict:
    """The knob dict every script forwards to ``EmbeddingDistribution.sample``.

    One place, so a new sampler knob reaches generate/sweeps/explore by being
    added to :func:`add_backend_args` rather than by being copy-pasted four times.
    ``lengths`` is the batch's drawn content lengths (see :func:`resolve_lengths`),
    passed through untouched.
    """
    return dict(
        sampler=args.sampler,
        coherence=getattr(args, "coherence", 0.5),
        components=getattr(args, "components", None),
        comp_lo=getattr(args, "comp_lo", 0),
        equalize=getattr(args, "equalize", False),
        truncation=getattr(args, "truncation", None),
        rho=getattr(args, "rho", 0.0),
        empirical_head=getattr(args, "empirical_head", 0),
        temp_on=getattr(args, "temp_on", 1.0),
        temp_off=getattr(args, "temp_off", 1.0),
        lengths=lengths,
    )


def neg_dists_kwarg(args, dists) -> dict:
    """``dists=`` for the backends whose negative branch can use it.

    ``--neg-mode mean``/``zeros`` need the fitted distributions handed to
    ``generate``; without them the mode silently degrades to the backend default,
    which is how ``--neg-mode`` spent a while being a no-op on sd15/sd2 outside
    ``generate.py``. The flow models' ``generate()`` takes no ``dists`` at all,
    hence the allow-list rather than passing it unconditionally.
    """
    return {"dists": dists} if args.backend in ("sd15", "sd2", "sdxl") else {}


def resolve_lengths(args, backend, dists, n: int, rng, quiet: bool = False):
    """Draw this batch's content lengths, or None when length mode is off.

    ``--length N`` on its own implies ``--length-mode fixed`` -- naming a length
    and then not using it is never what was meant. Prints why it fell back when
    the loaded fit predates the length split, because the alternative is a run
    that silently ignores the knob you came for.
    """
    mode = getattr(args, "length_mode", "off")
    if mode == "off" and getattr(args, "length", None) is not None:
        mode = args.length_mode = "fixed"
    if mode == "off":
        return None
    if mode == "fixed" and getattr(args, "length", None) is None:
        raise SystemExit("[args] --length-mode fixed needs --length N")
    if not any(d.has_length_stats for d in dists.values()):
        print("[args] WARNING: this distribution carries no content/padding "
              "split, so --length-mode is ignored. Re-mine it (the length "
              "dimension is recorded automatically) to enable it.")
        args.length_mode = "off"
        return None
    lengths = backend.draw_lengths(dists, n, rng=rng, mode=mode,
                                   length=getattr(args, "length", None))
    if lengths is not None and not quiet:
        print(f"[args] length-conditional sampling [{mode}]: "
              f"median {int(np.median(lengths))} tokens, "
              f"range {int(lengths.min())}-{int(lengths.max())}")
    return lengths


def warn_sampler_args(args, dists) -> None:
    """Flag knob settings the corpus measurement says are wasted effort.

    Warnings, never errors: reaching past the noise floor on purpose is a
    legitimate experiment (that is where "broken samplers stay in the toolbox"
    comes from), but doing it by accident is just entropy spent on stored noise.
    """
    floors = [d.noise_floor_axes() for d in dists.values()]
    floor = min([f for f in floors if f], default=None)
    if floor and args.sampler in ("pca", "blend", "split"):
        lo = int(getattr(args, "comp_lo", 0) or 0)
        comps = getattr(args, "components", None)
        hi = lo + int(comps) if comps else None
        if lo >= floor:
            print(f"[args] WARNING: --comp-lo {lo} starts at or past the "
                  f"shuffle-null noise floor (~{floor} axes) -- every selected "
                  f"axis is indistinguishable from stored noise.")
        elif hi and hi > floor:
            print(f"[args] note: axes {floor}-{hi} are below the shuffle-null "
                  f"noise floor (~{floor} axes) and carry no corpus structure.")
    head = int(getattr(args, "empirical_head", 0) or 0)
    if head and not any(d.pca_head is not None for d in dists.values()):
        print("[args] WARNING: this distribution has no empirical PCA head; "
              "--empirical-head is ignored. Re-mine to enable it.")
    elif head and int(getattr(args, "comp_lo", 0) or 0) >= head:
        print(f"[args] WARNING: --empirical-head {head} covers axes 0-{head - 1}, "
              f"but --comp-lo {args.comp_lo} skips past all of them, so it does "
              f"nothing.")
    if getattr(args, "rho", 0.0) and args.sampler in ("pca", "hybrid"):
        print(f"[args] note: --rho only shapes the DIAGONAL draw; "
              f"sampler={args.sampler} ignores it "
              f"(blend/split use it for their diagonal half).")


def resolve_gen_defaults(args) -> None:
    """Fill steps/guidance/neg_mode from the backend family if left unset."""
    d = BACKEND_DEFAULTS[args.backend]
    if getattr(args, "steps", None) is None:
        args.steps = d["steps"]
    if getattr(args, "guidance", None) is None:
        args.guidance = d["guidance"]
    if getattr(args, "neg_mode", None) is None:
        # sd15 samples against the house negative prompt (the string every Eden
        # SD1.5 render has used); sdxl pushes from the corpus mean; the rest keep
        # the empty-prompt negative.
        args.neg_mode = {"sd15": "text", "sdxl": "mean"}.get(args.backend, "empty")
    if getattr(args, "height", None) in (None, 0):
        args.height = d["height"]
    if getattr(args, "width", None) in (None, 0):
        args.width = d["width"]


def load_backend(args, encode_only: bool = False):
    """Construct the backend from parsed args (touches torch/diffusers).

    ``encode_only`` is for mining: skip the UNet/VAE, which that path never uses.

    ``--negative`` overrides the CFG negative *text* on the backends that have
    one (sd15 sets the house default; sd2 starts with none). sdxl/flux2/krea2
    have no ``negative_prompt`` attribute and are left alone -- their negative is
    a tensor, chosen by ``--neg-mode``.
    """
    be = make_backend(args.backend, model_id=args.model, ckpt=args.ckpt,
                      device=args.device, encode_only=encode_only)
    neg = getattr(args, "negative", None)
    if neg is not None and hasattr(getattr(be, "model", None), "negative_prompt"):
        be.model.negative_prompt = neg.strip() or None
    return be


def effective_negative(backend, neg_mode: str):
    """The negative prompt text a run actually used, for the image sidecar.

    ``None`` whenever no *text* negative was in play -- an sdxl tensor negative,
    ``--neg-mode empty/mean/zeros``, or a backend with no ``negative_prompt``.
    Recording it means an image says which words it was pushed away from, not
    just which mode was picked (the default text can be overridden per run).
    """
    if neg_mode != "text":
        return None
    return getattr(getattr(backend, "model", None), "negative_prompt", None)


def dist_prefix(args, base: str) -> str:
    """Backend-namespaced distribution prefix so the backends never clash.

    sd15 keeps the bare ``<base>`` (original layout); others append their name
    (``<base>_sd2``, ``<base>_sdxl``). The rule itself lives in
    :mod:`semantic_anarchy.dist_paths`, which the dashboard also reads to work
    out which ``.npz`` files a base has (or hasn't) been encoded to.
    """
    return backend_prefix(base, args.backend)
