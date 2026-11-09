#!/usr/bin/env python3
"""More images inside a curated collection ("aesthetic universe").

Each child picks one member as its ANCHOR, moves away from it by ``--spread``
with ``--method``, and is rendered the way that anchor was rendered (checkpoint,
size, steps, guidance, scheduler, CFG negative). Why member-anchored instead of
fitting a Gaussian to the collection: docs/reference/collection-sampling.md.

    python scripts/sample_collection.py --backend sd15 --n 8 --spread 1 \\
        --images outputs/generated/a.jpg outputs/generated/b.jpg ...
    python scripts/sample_collection.py --from-file outputs/dist_fits/x.sources.json \\
        --method auto --spread 0.5 --n 12

``--spread``: 0 = the anchor's exact conditioning (only the noise seed is new),
1 = the collection's natural diversity, >1 = past its edges.

Needs the FULL tier (torch + diffusers). Each checkpoint is loaded once.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.backend import BACKEND_DEFAULTS, dist_backend
from semantic_anarchy.cli_args import (
    add_experiment_args,
    dist_prefix,
    record_experiment,
)
from semantic_anarchy.collection_sampler import (
    AUTO_METHOD,
    DEFAULT_SPREAD,
    METHODS,
    SEED_MODES,
    Collection,
    group_by_checkpoint,
    load_members,
    read_image_list,
    resolve_seed_mode,
    sample,
)
from semantic_anarchy.io_utils import (
    image_ext,
    save_image,
    unique_image_path,
)

#: In-process diffusers backends whose sidecars carry everything a replay needs.
BACKENDS = ("sd15", "sd2")


def _log(msg: str) -> None:
    print(f"[collection] {msg}", flush=True)


def _settings(member, args) -> dict:
    """The anchor's render settings, holes filled from the backend defaults."""
    d = BACKEND_DEFAULTS[args.backend]
    s = member.render_settings()
    s["model"] = args.ckpt or s.get("model")
    if s["model"] and not Path(str(s["model"])).exists() and not args.ckpt:
        _log(f"WARNING: {member.name}'s checkpoint {s['model']} is not on disk; "
             f"its children use the backend default")
        s["model"] = None
    s["steps"] = int(s.get("steps") or d["steps"])
    s["guidance"] = float(s["guidance"] if s.get("guidance") is not None else d["guidance"])
    s["scheduler"] = s.get("scheduler") or "default"
    s["neg_mode"] = s.get("neg_mode") or {"sd15": "text"}.get(args.backend, "empty")
    s["width"] = int(s.get("width") or d["width"])
    s["height"] = int(s.get("height") or d["height"])
    return s


def _load_dists(backend_name: str, base) -> dict | None:
    if not base:
        return None
    try:
        return dist_backend(backend_name).load_dists(dist_prefix(
            argparse.Namespace(backend=backend_name), str(base)))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        _log(f"can't read distribution {base} ({exc.__class__.__name__})")
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="sd15", choices=BACKENDS)
    parser.add_argument("--images", nargs="*", type=Path, default=[],
                        help="the collection's images (any extension; upscales resolve "
                             "to the original that owns the conditioning)")
    parser.add_argument("--from-file", type=Path, default=None,
                        help="JSON list / {\"images\": [...]} / text file of image paths, "
                             "appended to --images")
    parser.add_argument("--method", default="auto", choices=("auto",) + METHODS,
                        help=f"how a child leaves its anchor (auto = {AUTO_METHOD})")
    parser.add_argument("--spread", type=float, default=DEFAULT_SPREAD,
                        help="0 = the anchor itself (new noise seed only), 1 = the "
                             "collection's natural diversity, >1 = past its edges")
    parser.add_argument("--seed-mode", default="auto", choices=("auto",) + SEED_MODES,
                        help="fresh = new noise per child (new compositions); anchor = "
                             "reuse the anchor's own noise seed (variations of that exact "
                             "image). auto = anchor while spread <= 1, fresh beyond")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None,
                        help="batch seed: drives anchor/partner choice and the fresh "
                             "noise seeds (child i renders with seed + i)")
    parser.add_argument("--dist", type=Path, default=Path("outputs/dist"),
                        help="base distribution the sidecar's distance gauge is measured "
                             "against (the selected base / root corpus)")
    parser.add_argument("--ckpt", default=None,
                        help="render EVERY child with this checkpoint instead of its "
                             "anchor's (breaks the replay; for experiments)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/generated"))
    add_experiment_args(parser)
    args = parser.parse_args(argv)

    paths = list(args.images)
    if args.from_file:
        paths += [Path(p) for p in read_image_list(args.from_file)]
    if not paths:
        raise SystemExit("[collection] no images given (--images / --from-file)")

    names = dist_backend(args.backend).tensor_names
    members, skipped = load_members(paths, names)
    for p, why in skipped[:10]:
        _log(f"skipped {Path(p).name}: {why}")
    if len(skipped) > 10:
        _log(f"... and {len(skipped) - 10} more skipped")
    if not members:
        raise SystemExit("[collection] none of the images has conditioning to sample from")
    method = AUTO_METHOD if args.method == "auto" else args.method
    if len(members) == 1 and method in ("slerp", "crossover"):
        _log(f"one member: {method} needs a partner, using jitter")
        method = "jitter"
    seed_mode = resolve_seed_mode(args.seed_mode, method, args.spread)

    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy) % (2**31)
        _log(f"no --seed -> random seed {seed} (pass --seed {seed} to reproduce)")
    rng = np.random.default_rng(seed)
    experiment = record_experiment(args, seed=seed, dist=str(args.dist),
                                   method=method, spread=args.spread,
                                   collection=[m.name for m in members])
    if experiment:
        _log(f"experiment {experiment} -> outputs/experiments/{experiment}.json")

    coll = Collection(members, names)
    children = sample(coll, method, args.spread, args.n, rng=rng)
    settings = [_settings(members[c.anchor], args) for c in children]
    models = sorted({str(s["model"]) for s in settings})
    _log(f"{len(members)} members -> {args.n} children, method={method} "
         f"spread={args.spread} seed-mode={seed_mode} seed={seed}; "
         f"{len(models)} checkpoint(s)")

    gauge = _load_dists(args.backend, args.dist)
    if gauge is None:
        _log("no readable --dist: sidecars record no distance")
    neg_cache: dict = {}

    from semantic_anarchy.backend import make_backend
    args.outdir.mkdir(parents=True, exist_ok=True)
    written = 0
    # Grouped on the checkpoint each child will actually load (after fallbacks).
    for model, idxs in group_by_checkpoint(settings).items():
        _log(f"loading {args.backend} {model or '(default)'} for {len(idxs)} child(ren) ...")
        be = make_backend(args.backend, ckpt=model, device=args.device, remote_ok=False)
        sched0 = be.model.pipe.scheduler
        default_neg = be.model.negative_prompt
        for i in idxs:
            ch, s = children[i], settings[i]
            anchor = members[ch.anchor]
            be.model.pipe.scheduler = sched0
            be.set_scheduler(s["scheduler"])
            neg_text = s.get("negative") if s["neg_mode"] == "text" else None
            if s["neg_mode"] == "text":
                be.model.negative_prompt = neg_text if neg_text is not None else default_neg
                be.model._neg = None
            dists = None
            if s["neg_mode"] == "mean":
                key = str(s.get("dist"))
                if key not in neg_cache:
                    neg_cache[key] = _load_dists(args.backend, s.get("dist"))
                dists = neg_cache[key] or gauge
            image_seed = (int(anchor.meta["image_seed"])
                          if seed_mode == "anchor" and anchor.meta.get("image_seed") is not None
                          else seed + i)
            named = {k: v[None] for k, v in ch.latents.items()}
            used_neg = be.model.negative_prompt if s["neg_mode"] == "text" else None

            def save(_j, img, i=i, ch=ch, s=s, anchor=anchor, image_seed=image_seed,
                     executor=be.executor, used_neg=used_neg):
                nonlocal written
                path = unique_image_path(
                    args.outdir / f"anarchy_{args.backend}_{seed}_{i:03d}{image_ext()}")
                np.savez(path.with_suffix(".npz"), **ch.latents)
                dist = None
                if gauge is not None:
                    dist = round(float(np.mean([gauge[k].distance(ch.latents[k])
                                                for k in gauge if k in ch.latents])), 3)
                meta = {
                    "kind": "collection", "experiment": experiment,
                    "backend": args.backend, "executor": executor,
                    "method": ch.method, "spread": args.spread, "seed_mode": seed_mode,
                    "anchors": [anchor.name] + ([members[ch.partner].name]
                                                if ch.partner is not None else []),
                    "anchor": anchor.name, "t": ch.t,
                    "collection_size": len(members),
                    "model": s["model"] or "(default)",
                    "steps": s["steps"], "guidance": s["guidance"],
                    "scheduler": s["scheduler"], "neg_mode": s["neg_mode"],
                    "negative": used_neg,
                    "height": s["height"], "width": s["width"],
                    "batch_seed": seed, "image_seed": image_seed, "index": i,
                    "dist": str(args.dist), "distance": dist,
                    "render_from": anchor.name,
                }
                path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
                save_image(img, path)
                written += 1
                _log(f"saved {path}  ({written}/{args.n}; anchor {anchor.name}"
                     + (f", t={ch.t}" if ch.t else "") + ")")

            be.generate(named, dists=dists, guidance=s["guidance"], steps=s["steps"],
                        seed=image_seed, height=s["height"], width=s["width"],
                        neg_mode=s["neg_mode"], on_image=save)
        del be  # free this checkpoint's VRAM before loading the next one
        with contextlib.suppress(Exception):
            import torch
            torch.cuda.empty_cache()
    _log(f"done -- {written} images in {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
