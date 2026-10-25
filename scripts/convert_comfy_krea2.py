#!/usr/bin/env python3
"""Turn a ComfyUI Krea 2 checkpoint into a diffusers folder ``Krea2Pipeline`` loads.

ComfyUI ships Krea 2 as ONE transformer file with its own tensor names
(``blocks.N.attn.wq``, ``txtmlp.1``, ...) and, for the fp8 variants, per-tensor
``weight_scale`` siblings. diffusers has no ``from_single_file`` for
``Krea2Transformer2DModel``, so this rewrites the transformer under the diffusers
names (dequantizing fp8 -> bf16) and assembles the rest of the folder WITHOUT
copying anything large:

- ``transformer/``  converted shards (the only real bytes written, ~26GB)
- ``text_encoder/`` Krea's config + symlinks into a local Qwen/Qwen3-VL-4B-Instruct
  snapshot (bit-identical to the text encoder Krea ships)
- ``vae/``          Krea's config + a symlink to Qwen/Qwen-Image's VAE (identical)
- ``model_index.json`` / ``scheduler/`` / ``tokenizer/`` small files from the hub

The rename was checked tensor-for-tensor against ``krea/Krea-2-Raw``'s own
diffusers weights (exact for the bf16 file; fp8 rounding for the fp8 one).

    python scripts/convert_comfy_krea2.py \
        --src ~/SSD2TB/Github_repos/ComfyUI/models/diffusion_models/krea2_raw_bf16.safetensors \
        --repo krea/Krea-2-Raw --out ~/SSD2TB/models/krea2/Krea-2-Raw
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

# ComfyUI name -> diffusers name, applied in order to the key.
RENAMES = [
    (r"^blocks\.", "transformer_blocks."),
    (r"^txtfusion\.", "text_fusion."),
    (r"\.attn\.wq\.", ".attn.to_q."),
    (r"\.attn\.wk\.", ".attn.to_k."),
    (r"\.attn\.wv\.", ".attn.to_v."),
    (r"\.attn\.wo\.", ".attn.to_out.0."),
    (r"\.attn\.gate\.", ".attn.to_gate."),
    (r"\.attn\.qknorm\.qnorm\.scale$", ".attn.norm_q.weight"),
    (r"\.attn\.qknorm\.knorm\.scale$", ".attn.norm_k.weight"),
    (r"\.mlp\.", ".ff."),
    (r"\.prenorm\.scale$", ".norm1.weight"),
    (r"\.postnorm\.scale$", ".norm2.weight"),
    (r"\.mod\.lin$", ".scale_shift_table"),
    (r"^first\.", "img_in."),
    (r"^last\.linear\.", "final_layer.linear."),
    (r"^last\.modulation\.lin$", "final_layer.scale_shift_table"),
    (r"^last\.norm\.scale$", "final_layer.norm.weight"),
    (r"^tmlp\.0\.", "time_embed.linear_1."),
    (r"^tmlp\.2\.", "time_embed.linear_2."),
    (r"^tproj\.1\.", "time_mod_proj."),
    (r"^txtmlp\.0\.scale$", "txt_in.norm.weight"),
    (r"^txtmlp\.1\.", "txt_in.linear_1."),
    (r"^txtmlp\.3\.", "txt_in.linear_2."),
]

SMALL_FILES = ["model_index.json", "scheduler/scheduler_config.json",
               "transformer/config.json", "text_encoder/config.json",
               "vae/config.json", "tokenizer/tokenizer.json",
               "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja"]


def rename(key: str) -> str:
    for pat, rep in RENAMES:
        key = re.sub(pat, rep, key)
    return key


def cached(repo: str, rel: str) -> Path:
    """A file of a repo that is already in the local HF cache (partial snapshots ok)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, rel, local_files_only=True))


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def convert_transformer(src: Path, out: Path, shard_gb: float) -> None:
    import torch
    from accelerate import init_empty_weights
    from diffusers import Krea2Transformer2DModel
    from safetensors import safe_open
    from safetensors.torch import save_file

    cfg = json.loads((out / "transformer/config.json").read_text())
    with init_empty_weights():
        want = Krea2Transformer2DModel.from_config(cfg).state_dict()

    with safe_open(str(src), "pt") as f:
        keys = [k for k in f.keys() if not k.endswith(".weight_scale")]
        mapped = {rename(k): k for k in keys}
        missing, extra = set(want) - set(mapped), set(mapped) - set(want)
        if missing or extra:
            raise SystemExit(f"[krea2] key mismatch: missing={sorted(missing)[:5]} "
                             f"extra={sorted(extra)[:5]}")

        shards, cur, cur_bytes, limit = [], {}, 0, shard_gb * 1e9
        for dk in sorted(mapped):
            ck = mapped[dk]
            t = f.get_tensor(ck)
            if t.dtype == torch.float8_e4m3fn:
                t = (t.float() * f.get_tensor(ck + "_scale").float()).to(torch.bfloat16)
            t = t.reshape(want[dk].shape).contiguous()
            cur[dk] = t
            cur_bytes += t.numel() * t.element_size()
            if cur_bytes >= limit:
                shards.append(cur)
                cur, cur_bytes = {}, 0
        if cur:
            shards.append(cur)

        tdir = out / "transformer"
        for old in tdir.glob("diffusion_pytorch_model*.safetensors"):
            old.unlink()
        weight_map, total = {}, 0
        for i, sd in enumerate(shards, 1):
            name = f"diffusion_pytorch_model-{i:05d}-of-{len(shards):05d}.safetensors"
            save_file(sd, str(tdir / name), metadata={"format": "pt"})
            for k, t in sd.items():
                weight_map[k] = name
                total += t.numel() * t.element_size()
            print(f"[krea2] wrote {name} ({len(sd)} tensors)", flush=True)
        (tdir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="ComfyUI krea2_*.safetensors")
    ap.add_argument("--repo", required=True, help="krea/Krea-2-Raw or krea/Krea-2-Turbo (configs only)")
    ap.add_argument("--out", required=True, help="diffusers folder to create")
    ap.add_argument("--te-repo", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--vae-repo", default="Qwen/Qwen-Image")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download

    out = Path(args.out).expanduser()
    for rel in SMALL_FILES:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(hf_hub_download(args.repo, rel)).resolve(), dst)

    index = "model.safetensors.index.json"
    te_index = cached(args.te_repo, index)
    link(te_index, out / "text_encoder" / index)
    for name in sorted(set(json.loads(te_index.read_text())["weight_map"].values())):
        link(cached(args.te_repo, name), out / "text_encoder" / name)
    vae = "vae/diffusion_pytorch_model.safetensors"
    link(cached(args.vae_repo, vae), out / vae)

    convert_transformer(Path(args.src).expanduser(), out, args.shard_gb)
    print(f"[krea2] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
