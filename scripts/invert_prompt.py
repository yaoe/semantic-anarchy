#!/usr/bin/env python3
"""PEZ hard-prompt inversion -- "what would the prompt be?"

Implements Wen et al. 2023, "Hard Prompts Made Easy" (arXiv:2302.03668):
optimize M continuous token embeddings by Adam, but forward every step through
their nearest REAL vocabulary tokens (straight-through estimator). Two spaces:

* ``--space clip`` (any backend) -- maximize cosine similarity with the image's
  CLIP ViT-L embedding: "what words come closest to this picture, in CLIP's
  eyes". A neutral outside observer.
* ``--space native`` (sd15 / sdxl) -- match the image's ACTUAL sampled
  conditioning tensor (the ``.npz`` sidecar) through the model's own text
  encoder(s): "which typeable prompt lands nearest to this exact off-grid
  point, in the space we sample in". sd15: CLIP ViT-L final hidden states.
  sdxl: BOTH encoders jointly -- the two tokenizers share one BPE vocabulary,
  so a single hard token sequence feeds both; we keep one continuous matrix per
  embedding table, couple them through a joint nearest-neighbor projection in
  the concatenated space, and straight-through each branch.

Either way the caption is a RULER, not a leash: it shows how near words can
get to an off-grid discovery. The residual similarity is an unpromptability
signal. It is never fed back into generation automatically.

    python scripts/invert_prompt.py --src outputs/generated/X.jpg --tokens 12
    python scripts/invert_prompt.py --src ... --space native
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.clip_compat import (causal_mask, encoder_hidden_states,
                                          image_features)

CLIP_ID = "openai/clip-vit-large-patch14"
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def vocab_mask(tokenizer, mode="full"):
    """Candidate token ids for the projection step.

    ``full``  -- everything CLIP knows except special/control tokens: words,
                 punctuation, unicode, and yes, the emoji CLIP's alt-text
                 vocabulary genuinely contains.
    ``clean`` -- plain lowercase-ascii words/subwords only (tidier captions).
    """
    import torch
    keep = torch.zeros(len(tokenizer), dtype=torch.bool)
    special = set(tokenizer.all_special_ids)
    for i in range(len(tokenizer)):
        if i in special:
            continue
        t = tokenizer.convert_ids_to_tokens(i)
        s = t[:-4] if t.endswith("</w>") else t
        if not s or not s.strip():
            continue
        if mode == "clean":
            if all(("a" <= c <= "z") or ("0" <= c <= "9") for c in s):
                keep[i] = True
        else:
            keep[i] = True
    return keep


def pez_optimize(objective, tables, cand_ids, tokens, steps, restarts, lr, seed):
    """Generic PEZ loop over one or more coupled embedding tables.

    ``tables`` is a list; a continuous matrix is kept per table, the hard
    projection picks ONE shared token id per position via nearest neighbor in
    the concatenation of all tables, and each branch gets its own
    straight-through hop. ``objective`` receives a list of straight-through
    row tensors (one per table) and returns a scalar to MAXIMIZE.
    Returns (best_score, best_token_ids).
    """
    import torch
    cands = [t[cand_ids] for t in tables]
    cat = torch.cat(cands, dim=-1)
    cat_n = cat / cat.norm(dim=-1, keepdim=True)

    best_score, best_ids = -1e9, None
    for r in range(restarts):
        g = torch.Generator(device="cpu").manual_seed(seed + r)
        start = torch.randint(len(cand_ids), (tokens,), generator=g)
        Ps = [c[start].clone().detach().requires_grad_(True) for c in cands]
        opt = torch.optim.Adam(Ps, lr=lr)
        for _ in range(steps):
            with torch.no_grad():
                joint = torch.cat(Ps, dim=-1)
                joint = joint / joint.norm(dim=-1, keepdim=True)
                idx = (joint @ cat_n.T).argmax(dim=-1)
                ids = cand_ids[idx]
            rows = [P + (c[idx] - P).detach() for P, c in zip(Ps, cands)]
            score = objective(rows)
            loss = -score
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            s = float(score.detach())
            if s > best_score:
                best_score, best_ids = s, ids.tolist()
        print(f"[invert] restart {r + 1}/{restarts}: best so far "
              f"score={best_score:.4f}", flush=True)
    return best_score, best_ids


def build_seq(table, pad_id, bos_id, eos_id, rows, length, device):
    """[BOS, rows..., EOS, pad...] as embeddings, padded to ``length``."""
    import torch
    n_pad = length - rows.shape[0] - 2
    pad = table[pad_id][None].expand(n_pad, -1) if n_pad > 0 else \
        torch.zeros(0, table.shape[1], device=device)
    return torch.cat([table[bos_id][None], rows, table[eos_id][None], pad],
                     dim=0)[None]


def detect_backend(src: Path) -> str:
    m = re.match(r"anarchy_([a-z0-9]+)_", src.name)
    return m.group(1) if m else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--space", default="clip", choices=["clip", "native"],
                    help="clip = match the image in CLIP's eyes (any backend); "
                         "native = match the stored conditioning through the "
                         "model's own encoder (sd15/sdxl)")
    ap.add_argument("--tokens", type=int, default=12,
                    help="hard-prompt length M (the 'how hard language tried' knob)")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--restarts", type=int, default=3,
                    help="independent runs; keep the best (guards against a bad "
                         "run reading as falsely 'unpromptable')")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--vocab", default="full", choices=["full", "clean"],
                    help="projection vocabulary: full = unicode + emoji too")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.src.exists():
        raise SystemExit(f"[invert] source not found: {args.src}")

    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[invert] space={args.space}, loading {CLIP_ID} ...", flush=True)
    clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)
    tok = proc.tokenizer
    for p in clip.parameters():
        p.requires_grad_(False)

    tm = clip.text_model
    table = tm.embeddings.token_embedding.weight          # (V, d)
    keep = vocab_mask(tok, args.vocab).to(device)
    cand_ids = keep.nonzero(as_tuple=True)[0]
    print(f"[invert] vocab [{args.vocab}]: {len(cand_ids)} of "
          f"{table.shape[0]} tokens", flush=True)
    M = args.tokens
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def cosrows(a, b):                                    # mean per-row cosine
        a = a / a.norm(dim=-1, keepdim=True)
        b = b / b.norm(dim=-1, keepdim=True)
        return (a * b).sum(-1).mean()

    if args.space == "clip":
        from PIL import Image
        with torch.no_grad():
            inputs = proc(images=[Image.open(args.src).convert("RGB")],
                          return_tensors="pt").to(device)
            img_feat = image_features(clip, inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        L = M + 2
        pos = tm.embeddings.position_embedding(torch.arange(L, device=device))
        causal = causal_mask(L, device)

        def objective(rows):
            seq = build_seq(table, pad_id, tok.bos_token_id, tok.eos_token_id,
                            rows[0], L, device) + pos[None]
            h = encoder_hidden_states(tm, seq, causal)[-1]
            pooled = tm.final_layer_norm(h)[:, M + 1]     # EOS position
            f = clip.text_projection(pooled)
            f = f / f.norm(dim=-1, keepdim=True)
            return (f @ img_feat.T).squeeze()

        best, ids = pez_optimize(objective, [table], cand_ids, M,
                                 args.steps, args.restarts, args.lr, args.seed)
        fields = {"inverted_prompt": tok.decode(ids).strip(),
                  "inverted_sim": round(best, 4),
                  "inverted_tokens": M, "inverted_vocab": args.vocab}
        headline = f"CLIP sim {best:.4f}"

    else:
        backend = detect_backend(args.src)
        npz = args.src.with_suffix(".npz")
        if backend not in ("sd15", "sdxl"):
            raise SystemExit(f"[invert] native space supports sd15/sdxl, "
                             f"not '{backend or 'unknown'}'")
        if not npz.is_file():
            raise SystemExit(f"[invert] no conditioning sidecar: {npz.name}")
        import numpy as np
        data = np.load(npz)
        L = 77
        pos = tm.embeddings.position_embedding(torch.arange(L, device=device))
        causal = causal_mask(L, device)

        if backend == "sd15":
            target = torch.tensor(np.asarray(data["embeds"], dtype=np.float32),
                                  device=device)          # (77, 768)

            def objective(rows):
                seq = build_seq(table, pad_id, tok.bos_token_id,
                                tok.eos_token_id, rows[0], L, device) + pos[None]
                h = encoder_hidden_states(tm, seq, causal)[-1]
                out = tm.final_layer_norm(h)[0]           # sd15: final LN states
                return cosrows(out, target)

            best, ids = pez_optimize(objective, [table], cand_ids, M,
                                     args.steps, args.restarts, args.lr,
                                     args.seed)
        else:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
            print("[invert] loading SDXL text_encoder_2 (OpenCLIP bigG) ...",
                  flush=True)
            enc2 = CLIPTextModelWithProjection.from_pretrained(
                SDXL_ID, subfolder="text_encoder_2").to(device).eval()
            tok2 = CLIPTokenizer.from_pretrained(SDXL_ID,
                                                 subfolder="tokenizer_2")
            for p in enc2.parameters():
                p.requires_grad_(False)
            tm2 = enc2.text_model
            table2 = tm2.embeddings.token_embedding.weight
            pos2 = tm2.embeddings.position_embedding(
                torch.arange(L, device=device))
            pad2 = tok2.pad_token_id if tok2.pad_token_id is not None else \
                tok2.eos_token_id
            target = torch.tensor(
                np.asarray(data["prompt_embeds"], dtype=np.float32),
                device=device)                            # (77, 2048)
            target_pooled = torch.tensor(
                np.asarray(data["pooled"], dtype=np.float32),
                device=device).reshape(-1)                # (1280,)
            t_vitl, t_bigg = target[:, :768], target[:, 768:]

            def objective(rows):
                # ViT-L branch (penultimate layer, diffusers convention)
                seq1 = build_seq(table, pad_id, tok.bos_token_id,
                                 tok.eos_token_id, rows[0], L, device) + pos[None]
                h1 = encoder_hidden_states(tm, seq1, causal)[-2][0]
                # bigG branch (penultimate + projected pooled at EOS)
                seq2 = build_seq(table2, pad2, tok2.bos_token_id,
                                 tok2.eos_token_id, rows[1], L, device) + pos2[None]
                hs2 = encoder_hidden_states(tm2, seq2, causal)
                h2 = hs2[-2][0]
                pooled = tm2.final_layer_norm(hs2[-1])[:, M + 1]
                pooled = enc2.text_projection(pooled).reshape(-1)
                pc = torch.nn.functional.cosine_similarity(
                    pooled, target_pooled, dim=0)
                return (cosrows(h1, t_vitl) + cosrows(h2, t_bigg) + pc) / 3.0

            best, ids = pez_optimize(objective, [table, table2], cand_ids, M,
                                     args.steps, args.restarts, args.lr,
                                     args.seed)
        fields = {"native_prompt": tok.decode(ids).strip(),
                  "native_sim": round(best, 4),
                  "native_tokens": M, "native_vocab": args.vocab}
        headline = f"conditioning cosine {best:.4f}"

    prompt = fields.get("inverted_prompt") or fields.get("native_prompt")
    print(f"[invert] nearest hard prompt [{args.space}] ({M} tokens, "
          f"{headline}):", flush=True)
    print(f'[invert]   "{prompt}"', flush=True)

    sidecar = args.src.with_suffix(".json")
    meta = {}
    if sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text())
        except Exception:
            meta = {}
    meta.update(fields)
    sidecar.write_text(json.dumps(meta, indent=2))
    print(f"[invert] saved to {sidecar.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
