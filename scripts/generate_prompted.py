#!/usr/bin/env python3
"""Generate an image FROM A TEXT PROMPT -- the comparison arm of PEZ inversion.

The one place in Semantic Anarchy where the text encoder IS called: after
``invert_prompt.py`` discovers the nearest typeable prompt to an off-grid
image, this renders what that prompt actually produces through the same
backend, so the discovery and the best language can do hang side by side.
The visual gap between them is the point.

    python scripts/generate_prompted.py --backend sdxl --prompt "..." \
        --parent anarchy_sdxl_123_000.png --seed 123
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.io_utils import unique_path
from semantic_anarchy.cli_args import (
    add_backend_args, resolve_gen_defaults, load_backend,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--parent", default=None,
                        help="source image filename this prompt was inverted "
                             "from (recorded in the sidecar, linked in the UI)")
    parser.add_argument("--prompt-kind", default="pez",
                        help="which inversion produced the prompt (pez/native/custom)")
    parser.add_argument("--seed", type=int, default=None,
                        help="use the parent's image_seed for a like-for-like "
                             "comparison (same initial noise)")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/generated"))
    args = parser.parse_args(argv)
    resolve_gen_defaults(args)

    import numpy as np
    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy) % (2**31)

    print(f"[gen-prompt] loading {args.backend} model "
          f"{args.ckpt or args.model or '(default)'} ...", flush=True)
    backend = load_backend(args)
    pipe = backend.model.pipe
    if args.scheduler != "default":
        from semantic_anarchy.pipeline import set_scheduler
        set_scheduler(pipe, args.scheduler)

    import torch
    print(f'[gen-prompt] rendering from prompt: "{args.prompt}" '
          f"(steps={args.steps}, guidance={args.guidance}, seed={seed})",
          flush=True)
    kw = dict(prompt=args.prompt, num_inference_steps=args.steps,
              guidance_scale=args.guidance,
              generator=torch.Generator(device="cpu").manual_seed(seed))
    if args.height:
        kw["height"] = args.height
    if args.width:
        kw["width"] = args.width
    img = pipe(**kw).images[0]

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = (Path(args.parent).stem if args.parent
            else f"anarchy_{args.backend}_{seed}")
    dest = unique_path(args.outdir / f"{stem}_{args.prompt_kind}.png")
    img.save(dest)
    dest.with_suffix(".json").write_text(json.dumps({
        "kind": "from_prompt", "backend": args.backend,
        "model": args.ckpt or args.model or "(default)",
        "prompt": args.prompt, "prompt_kind": args.prompt_kind,
        "parent": args.parent,
        "steps": args.steps, "guidance": args.guidance,
        "scheduler": args.scheduler, "seed": seed,
        "height": args.height, "width": args.width,
    }, indent=2))
    print(f"[gen-prompt] saved {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
