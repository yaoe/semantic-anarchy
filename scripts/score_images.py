#!/usr/bin/env python3
"""Score generated images with the LAION aesthetic predictor -> outputs/scores.json.

Lets the dashboard rank by predicted aesthetic ("Top rated"). Incremental: only
scores images not already in scores.json (use --rescore to redo all). Needs the
full tier (torch + CLIP) and the aesthetic head in weights/.

    python scripts/score_images.py            # score new images
    python scripts/score_images.py --rescore  # rescore everything
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.aesthetic import get_scorer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="outputs/generated/anarchy_*.png")
    ap.add_argument("--out", type=Path, default=Path("outputs/scores.json"))
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)

    from PIL import Image

    scores = {}
    if args.out.exists() and not args.rescore:
        try:
            scores = json.loads(args.out.read_text())
        except Exception:
            scores = {}

    paths = sorted(glob.glob(args.glob))
    todo = [p for p in paths if Path(p).relative_to("outputs").as_posix() not in scores]
    print(f"[score] {len(paths)} images, {len(todo)} to score", flush=True)
    if not todo:
        return 0

    scorer = get_scorer("aesthetic")
    if not scorer.available:
        print(f"[score] aesthetic scorer unavailable: {scorer._reason}", flush=True)
        return 1

    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        vals = scorer.score(imgs)
        for p, v in zip(chunk, vals):
            scores[Path(p).relative_to("outputs").as_posix()] = round(float(v), 3)
        args.out.write_text(json.dumps(scores, indent=0))
        print(f"[score]   {min(i+args.batch, len(todo))}/{len(todo)}", flush=True)

    print(f"[score] wrote {len(scores)} scores -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
