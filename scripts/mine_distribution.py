#!/usr/bin/env python3
"""Mine the conditioning-embedding distribution from a corpus of good prompts.

Slide 5, for either backend: encode ~1000 good prompts through the model's text
encoder(s) ONCE, harvest the conditioning tensor(s), and fit an
:class:`EmbeddingDistribution` per tensor. sd15 has one tensor (77x768); sdxl has
two (prompt_embeds 77x2048 + pooled 1280). Distributions are saved
backend-namespaced so the two never clash.

Needs the FULL tier (torch + diffusers). Run::

    python scripts/mine_distribution.py --backend sd15 --ckpt model.safetensors \
        --prompts prompts_1000.txt --out outputs/dist
    python scripts/mine_distribution.py --backend sdxl --model ~/models/sdxl-turbo \
        --prompts prompts_1000.txt --out outputs/dist
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy import plot_marginals
from semantic_anarchy.cli_args import (
    add_backend_args, resolve_gen_defaults, load_backend, dist_prefix,
)
from semantic_anarchy.progress import Progress

#: Default cap on retained PCA axes. Every axis costs a full feature row on disk
#: (59,136 floats = 236KB for sd15), so the full N-1 rank of a 4k-prompt corpus
#: is a ~1GB file -- and the corpus autopsy puts the noise floor at ~400 axes,
#: so most of that is stored noise. Override with --components; --components 0
#: keeps every axis (the old behaviour).
MAX_COMPONENTS = 512


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser, with_steps=False)
    parser.add_argument("--prompts", type=Path, default=Path("prompts_1000.txt"))
    parser.add_argument("--per-token", action="store_true", default=True,
                        help="fit one Gaussian per (token, feature) coordinate")
    parser.add_argument("--pooled", dest="per_token", action="store_false",
                        help="pool over tokens (one (hidden,) Gaussian for all tokens)")
    parser.add_argument("--out", type=Path, default=Path("outputs/dist"),
                        help="path prefix for the saved distribution(s)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="prompts encoded per forward pass (default 8)")
    parser.add_argument("--with-unet", action="store_true",
                        help="load the whole pipeline. Mining only runs the text "
                             "encoder, so the UNet/VAE are skipped by default; "
                             "use this if a model refuses the partial load.")
    parser.add_argument("--n-coords", type=int, default=4, help="marginal subplots to plot")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    resolve_gen_defaults(args)

    prompts = [
        line.strip()
        for line in args.prompts.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    print(f"[mine] backend={args.backend}  loaded {len(prompts)} prompts from {args.prompts}")

    print(f"[mine] loading {args.backend} model {args.ckpt or args.model or '(default)'} ...")
    t0 = time.monotonic()
    backend = load_backend(args, encode_only=not args.with_unet)
    print(f"[mine] model ready in {time.monotonic() - t0:.1f}s")

    print(f"[mine] encoding {len(prompts)} prompts -> conditioning tensor(s) "
          f"(the only text-encoder use), batch size {args.batch_size} ...")
    with Progress(len(prompts), label="encoding", unit="prompt",
                  prefix="[mine]") as bar:
        named = backend.encode(prompts, batch_size=args.batch_size,
                               on_batch=bar.update)
    for k, v in named.items():
        print(f"[mine]   tensor {k!r}: {np.asarray(v).shape}")

    # The length dimension: where each prompt's EOS falls. Tokenizer-only (no
    # GPU), but it has to happen while the model is still loaded, so it goes
    # here rather than beside the fit. It is what lets `sample(lengths=...)`
    # draw from the content lobe or the padding lobe instead of the gap between
    # them -- and it is free, so it is always recorded, never a flag.
    lengths = None
    if getattr(backend, "length_conditional", False):
        try:
            lengths = backend.token_lengths(prompts)
        except Exception as exc:      # a tokenizer that doesn't fit the mould
            print(f"[mine] length dimension unavailable ({exc!r}); "
                  f"fitting without the content/padding split.")
        if lengths is not None:
            print(f"[mine] content lengths: median {int(np.median(lengths))}, "
                  f"range {int(lengths.min())}-{int(lengths.max())} tokens")

    # Free the pipeline before the numpy fit -- on a 30GB-RAM box the
    # cpu-offloaded flow models (flux2/krea2) plus the fit workspace won't
    # coexist. Encoding is done; the model is no longer needed.
    backend.model = None
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # The fit is the other slow half (a Gram-trick PCA over the whole corpus) and
    # has no natural progress signal, so at least bracket it with a timing.
    # `--components 0` means "keep every axis"; unset means the MAX_COMPONENTS cap.
    comps = MAX_COMPONENTS if args.components is None else args.components
    comps = None if comps is not None and comps <= 0 else comps
    print(f"[mine] fitting distribution(s) over {len(prompts)} samples "
          f"(PCA, {comps if comps else 'all'} components; "
          f"no progress signal, sit tight) ...")
    t0 = time.monotonic()
    dists = backend.fit(named, per_token=args.per_token, n_components=comps,
                        lengths=lengths)
    print(f"[mine] fit done in {time.monotonic() - t0:.1f}s")
    first_dist = dists[list(dists)[0]]
    floor = first_dist.noise_floor_axes()
    if floor:
        print(f"[mine] shuffle-null noise floor ~{floor} axes: past that the "
              f"spectrum is indistinguishable from stored noise "
              f"({'capped' if comps and comps <= floor else 'kept ' + str(comps or 'all')}).")
    prefix = dist_prefix(args, str(args.out))
    written = backend.save_dists(dists, prefix)
    print(f"[mine] fitted {len(dists)} distribution(s) -> {', '.join(map(str, written))}")
    for k, d in dists.items():
        print(f"[mine]   {k}: {d.summary()}")

    # Marginal plots (slide 6/7) from the FIRST named tensor's raw embeddings.
    first = list(named.keys())[0]
    emb = np.asarray(named[first])
    args.outdir.mkdir(parents=True, exist_ok=True)
    hist = plot_marginals(emb, n=args.n_coords, style="hist",
                          out_path=args.outdir / f"mined_marginals_hist_{args.backend}.png")
    rug = plot_marginals(emb[:24], n=args.n_coords, style="rug",
                         out_path=args.outdir / f"mined_marginals_rug_{args.backend}.png")
    print(f"[mine] marginal plots ({first}) -> {hist} , {rug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
