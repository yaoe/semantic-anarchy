#!/usr/bin/env python3
"""The resonance engine: CLIP-embed the gallery, compute NOVELTY and personal
RESONANCE for every image, and write the caches the dashboard reads.

Three passes (all incremental / fast after the first run):

1. EMBED  -- CLIP ViT-L image embeddings for every generated image, cached in
   ``outputs/clip_embeds.npz`` (names + L2-normalized vectors).
2. NOVELTY -- for each image, cosine distance to its NEAREST NEIGHBOR among all
   other gallery images -> ``outputs/novelty.json``. High = unlike anything you
   have generated so far (catches motif repetition, which the corpus-center
   distance gauge cannot).
3. RESONANCE -- train a small logistic head on CLIP features with your ★
   favorites as positives vs sampled non-starred negatives; write P(you'd star
   it) for every image -> ``outputs/resonance.json`` (+ the trained head in
   ``outputs/taste_model.npz``). Skipped (with a message) until enough stars.

    python scripts/resonance.py           # incremental
    python scripts/resonance.py --re-embed  # rebuild embeddings from scratch
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.clip_compat import image_features
from semantic_anarchy.io_utils import IMAGE_EXTS

EMB_FILE = Path("outputs/clip_embeds.npz")
NOVELTY_FILE = Path("outputs/novelty.json")
RESONANCE_FILE = Path("outputs/resonance.json")
MODEL_FILE = Path("outputs/taste_model.npz")
MIN_POSITIVES = 8


def load_cache():
    if EMB_FILE.exists():
        d = np.load(EMB_FILE, allow_pickle=False)
        return list(d["names"]), d["vecs"].astype(np.float32)
    return [], np.zeros((0, 768), np.float32)


def embed_new(names, vecs, batch=16):
    """CLIP-embed gallery images not in the cache yet."""
    all_imgs = sorted(q for e in IMAGE_EXTS
                      for q in glob.glob(f"outputs/generated/anarchy_*{e}"))
    rels = [str(Path(p).relative_to("outputs")) for p in all_imgs]
    known = set(names)
    todo = [(r, p) for r, p in zip(rels, all_imgs) if r not in known]
    print(f"[resonance] {len(rels)} images, {len(todo)} to embed", flush=True)
    if not todo:
        return names, vecs

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    new_vecs = []
    with torch.no_grad():
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            imgs = [Image.open(p).convert("RGB") for _, p in chunk]
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            f = image_features(clip, inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            new_vecs.append(f.float().cpu().numpy())
            print(f"[resonance]   embed {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    names = names + [r for r, _ in todo]
    vecs = np.concatenate([vecs, np.concatenate(new_vecs)], axis=0)
    np.savez_compressed(EMB_FILE, names=np.array(names), vecs=vecs.astype(np.float16))
    return names, vecs


def compute_novelty(names, vecs):
    """Cosine NN-distance of each image to the rest of the gallery."""
    if len(names) < 2:
        return
    v = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
    # chunked so the similarity matrix never exceeds ~2k x N floats
    nov = np.empty(len(names), np.float32)
    for i in range(0, len(names), 2048):
        sims = v[i:i + 2048] @ v.T
        for r in range(sims.shape[0]):
            sims[r, i + r] = -1.0                       # exclude self
        nov[i:i + 2048] = 1.0 - sims.max(axis=1)
    NOVELTY_FILE.write_text(json.dumps(
        {n: round(float(x), 4) for n, x in zip(names, nov)}, indent=0))
    print(f"[resonance] novelty written for {len(names)} images "
          f"(median {np.median(nov):.3f})", flush=True)


def train_resonance(names, vecs, rng):
    """Logistic head on CLIP features: your ★ = positives."""
    favs_file = Path("outputs/favorites.json")
    favs = set(json.loads(favs_file.read_text())) if favs_file.exists() else set()
    idx = {n: i for i, n in enumerate(names)}
    pos = [idx[f] for f in favs if f in idx]
    if len(pos) < MIN_POSITIVES:
        print(f"[resonance] only {len(pos)} starred images have embeddings "
              f"(< {MIN_POSITIVES}) -- star more NEW images to train the taste "
              f"model; skipping resonance", flush=True)
        return
    neg_pool = [i for i in range(len(names)) if i not in set(pos)]
    neg = list(rng.choice(neg_pool, size=min(len(neg_pool), 5 * len(pos)),
                          replace=False))
    X = vecs[pos + neg].astype(np.float64)
    y = np.array([1.0] * len(pos) + [0.0] * len(neg))
    # balanced full-batch logistic regression with L2
    w = np.zeros(X.shape[1]); b = 0.0
    wpos, wneg = len(y) / (2 * max(1, len(pos))), len(y) / (2 * max(1, len(neg)))
    sw = np.where(y == 1, wpos, wneg)
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = (sw * (p - y))
        w -= 0.5 * (X.T @ g / len(y) + 1e-3 * w)
        b -= 0.5 * g.mean()
    np.savez(MODEL_FILE, w=w, b=b)
    p_all = 1.0 / (1.0 + np.exp(-(vecs.astype(np.float64) @ w + b)))
    RESONANCE_FILE.write_text(json.dumps(
        {n: round(float(x), 4) for n, x in zip(names, p_all)}, indent=0))
    p_pos = p_all[pos].mean()
    print(f"[resonance] taste model trained on {len(pos)}★ vs {len(neg)} -- "
          f"mean P(star) on your stars: {p_pos:.2f}; resonance written for "
          f"{len(names)} images", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--re-embed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.re_embed and EMB_FILE.exists():
        EMB_FILE.unlink()
    names, vecs = load_cache()
    names, vecs = embed_new(names, vecs)
    compute_novelty(names, vecs)
    train_resonance(names, vecs, np.random.default_rng(args.seed))
    print("[resonance] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
