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
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy import plot_marginals
from semantic_anarchy.cli_args import (
    add_backend_args, resolve_gen_defaults, load_backend, dist_prefix,
)


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
    backend = load_backend(args)

    print("[mine] encoding prompts -> conditioning tensor(s) (the only text-encoder use) ...")
    named = backend.encode(prompts)
    for k, v in named.items():
        print(f"[mine]   tensor {k!r}: {np.asarray(v).shape}")

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

    dists = backend.fit(named, per_token=args.per_token, n_components=args.components)
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
