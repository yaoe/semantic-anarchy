#!/usr/bin/env python3
"""Fit a distribution to the latents of images you picked, not prompts you wrote.

Every generated image carries its conditioning in a ``.npz`` sidecar, so a set of
images IS a corpus of latents — no text encoder, no GPU, no torch. Stack the
selection, fit it (mean, std, its own PCA subspace), save it under the ordinary
naming rules, and it is selectable as a base distribution like any mined corpus.

Sampling it with ``--sampler pca`` then draws inside the span of the latents you
picked, which is what "more like these" actually means. (The old
``evolve_favorites.py`` grafted the *corpus* PCA basis onto a re-centred
Gaussian, so its pca draws were corpus-sized deviations around a centre that
basis never saw — see ``semantic_anarchy/selection_fit.py``.)

    python scripts/fit_selection.py --backend sd15 --name keepers \
        --images outputs/generated/anarchy_sd15_7_000.jpg ...
    python scripts/fit_selection.py --backend sd15 --name keepers \
        --from-file outputs/dist_fits/keepers.sources.json
    python scripts/generate.py --backend sd15 --dist outputs/dist_fits/keepers \
        --sampler pca --n 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.backend import dist_backend
from semantic_anarchy.dist_paths import model_slug
from semantic_anarchy.io_utils import find_image
from semantic_anarchy.selection_fit import (
    FIT_DIR, MIN_SAMPLES, fit_base, fit_latents, latents_for, save_fit,
    slug_name, stack_latents, write_manifest,
)

#: Backends whose images this can fit. Anything that writes a per-image .npz
#: qualifies, which is every backend — the list exists to reject typos early.
BACKENDS = ("sd15", "sd2", "sdxl", "flux2", "krea2")


def _read_list(path: Path) -> list[str]:
    """Image paths from a JSON list / ``{"images": [...]}`` / a plain text file.

    The dashboard hands over a JSON file rather than argv because a selection is
    routinely hundreds of images long.
    """
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("images", [])
        if not isinstance(data, list):
            raise SystemExit(f"[fit] {path}: expected a JSON list of image paths")
        return [str(x) for x in data]
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="sd15", choices=BACKENDS)
    parser.add_argument("--images", nargs="*", type=Path, default=[],
                        help="image paths (any extension; the stem is the identity)")
    parser.add_argument("--from-file", type=Path, default=None,
                        help="JSON list / text file of image paths, appended to --images")
    parser.add_argument("--name", default=None,
                        help="fit name -> outputs/dist_fits/<name> (default: the "
                             "--out basename, or 'selection')")
    parser.add_argument("--out", type=Path, default=None,
                        help="explicit base prefix, overriding --name/--dir")
    parser.add_argument("--dir", type=Path, default=FIT_DIR,
                        help=f"where named fits live (default {FIT_DIR})")
    parser.add_argument("--components", type=int, default=0,
                        help="PCA axes to keep; 0 (default) = the full N-1 rank "
                             "of the selection")
    parser.add_argument("--max-corpus", type=int, default=256,
                        help="raw latents retained for the hybrid sampler")
    parser.add_argument("--note", default=None,
                        help="free text stored in the manifest (what this set is)")
    args = parser.parse_args(argv)

    paths = list(args.images)
    if args.from_file:
        paths += [Path(p) for p in _read_list(args.from_file)]
    if not paths:
        raise SystemExit("[fit] no images given (--images / --from-file)")

    base = str(args.out) if args.out else fit_base(
        args.name or "selection", args.dir)
    name = slug_name(args.name or Path(base).name)

    # ---- image -> conditioning sidecar -----------------------------------
    npz_paths, missing, models = [], [], []
    for raw in paths:
        img = find_image(Path(raw))
        if img is None:
            missing.append((raw, "no such image"))
            continue
        npz = latents_for(img)
        if npz is None:
            missing.append((raw, "no conditioning sidecar"))
            continue
        npz_paths.append(npz)
        try:
            meta = json.loads(npz.with_suffix(".json").read_text())
            if meta.get("model"):
                models.append(model_slug(meta["model"]))
        except (OSError, ValueError):
            pass

    print(f"[fit] {len(paths)} image(s) selected -> {len(npz_paths)} with latents")
    for p, why in missing[:10]:
        print(f"[fit]   skipped {Path(p).name}: {why}")
    if len(missing) > 10:
        print(f"[fit]   … and {len(missing) - 10} more skipped")

    names = dist_backend(args.backend).tensor_names
    stacked, used, skipped = stack_latents(npz_paths, names)
    for p, why in skipped[:10]:
        print(f"[fit]   skipped {p.name}: {why}")
    if len(skipped) > 10:
        print(f"[fit]   … and {len(skipped) - 10} more skipped")

    n = len(used)
    if n < MIN_SAMPLES:
        raise SystemExit(
            f"[fit] only {n} usable latent(s) for {args.backend} -- need at least "
            f"{MIN_SAMPLES}. (A selection mixing backends keeps only the first "
            f"shape it sees; check the skip reasons above.)")
    if len(set(models)) > 1:
        print(f"[fit] note: latents come from {len(set(models))} different "
              f"checkpoints ({', '.join(sorted(set(models)))}) -- the fit is still "
              f"valid, but sample it with the one you mostly picked from")

    # ---- the fit ---------------------------------------------------------
    dists = fit_latents(stacked, n_components=args.components or None,
                        max_corpus=args.max_corpus)
    for k, d in dists.items():
        shape = "×".join(str(x) for x in d.feature_shape)
        axes = 0 if d.pca_std is None else int(d.pca_std.shape[0])
        print(f"[fit] {k}: {d.n_samples} latents · {shape} · {axes} PCA axes · "
              f"{d.pca_variance_fraction():.1%} of the selection's variance")

    written = save_fit(dists, base, args.backend)
    manifest = write_manifest(
        base, args.backend,
        [str(p.with_suffix("")) for p in used],
        name=name, note=args.note, models=models,
        skipped=[(str(p), why) for p, why in [*missing, *skipped]],
        extra={"components": args.components or None,
               "feature_shape": [list(d.feature_shape) for d in dists.values()]},
    )
    print(f"[fit] fitted {n} latent(s) -> {', '.join(map(str, written))}")
    print(f"[fit] manifest -> {manifest}")
    print(f"[fit] sample it: --dist {base}  (pick it in the dashboard's "
          f"“Select base distribution…”)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
