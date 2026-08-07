"""Where a distribution lives on disk, and what a ``--dist`` prefix resolves to.

Torch-free. Three naming layers stack, in this order::

    <base>                        what --dist / --out receive
    <base>[_<backend>]            cli_args.dist_prefix (sd15 keeps the bare base)
    <prefix>[__<tensor>].npz      Backend.save_dists (multi-tensor backends only)

The *base* is the only thing anyone hands around: the dashboard stores it, the
CLIs take it, and the two suffix layers are re-derived from the backend name
wherever a concrete filename is needed. :func:`dist_files` walks that stack
forwards, :func:`base_from_npz` walks it back.

A prompt corpus keeps its distribution *next to the .txt file* and tags it with
the checkpoint that encoded it, so one corpus can carry a separate fit per
model::

    xander_prompts.txt  ->  xander_prompts__v1-5-pruned-emaonly.npz
                            xander_prompts__sdxl-base-1.0_sdxl__pooled.npz
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .backend import dist_backend

#: Single-file checkpoint extensions (a folder/HF id is named, not stemmed).
CKPT_EXTS = (".safetensors", ".ckpt")
#: What a prompt corpus looks like on disk.
PROMPTS_EXT = ".txt"
#: Separator between a corpus name and the checkpoint slug that encoded it.
CKPT_SEP = "__"


def backend_prefix(base: str | Path, backend: str) -> str:
    """Backend-namespace a base prefix (mirrors ``cli_args.dist_prefix``).

    sd15 keeps the bare base (the original ``outputs/dist`` layout); every other
    backend appends its own name so the fits never clash.
    """
    return str(base) if backend == "sd15" else f"{base}_{backend}"


def tensor_suffixes(backend: str) -> tuple[str, ...]:
    """Per-file suffixes a backend's fit is saved under.

    ``("",)`` for single-tensor backends (the fit *is* ``<prefix>.npz``),
    ``("__prompt_embeds", "__pooled")`` for sdxl.
    """
    names = dist_backend(backend).tensor_names
    if len(names) == 1:
        return ("",)
    return tuple(f"{CKPT_SEP}{n}" for n in names)


def dist_files(base: str | Path, backend: str) -> list[Path]:
    """Every ``.npz`` a ``--dist <base>`` run of this backend will open."""
    prefix = backend_prefix(base, backend)
    return [Path(f"{prefix}{suffix}.npz") for suffix in tensor_suffixes(backend)]


def dist_ready(base: str | Path, backend: str) -> bool:
    """True when this base is fully encoded for this backend (every .npz there)."""
    return all(p.is_file() for p in dist_files(base, backend))


def dist_meta(base: str | Path, backend: str) -> dict | None:
    """The ``.meta.json`` sidecar of the first tensor (n_samples, shape), if any."""
    files = dist_files(base, backend)
    if not files:
        return None
    meta = files[0].with_suffix(".meta.json")
    try:
        data = json.loads(meta.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def model_slug(model: str) -> str:
    """A filename-safe tag for the checkpoint a corpus was encoded with.

    A single-file checkpoint contributes its stem, a diffusers folder its name,
    and an HF repo id the whole ``org/name`` (the org is what disambiguates the
    dozens of ``*-base-1.0`` repos).
    """
    raw = str(model).strip().rstrip("/")
    p = Path(raw)
    if p.is_absolute() or raw.startswith("~") or raw.startswith("."):
        raw = p.stem if p.suffix.lower() in CKPT_EXTS else p.name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")
    return slug[:64] or "model"


def prompt_dist_base(prompts: str | Path, model: str) -> str:
    """Base prefix for a prompt corpus encoded with ``model``.

    Sits beside the ``.txt`` itself — the corpus and its latents travel together
    — and carries the checkpoint slug, so encoding the same file with a second
    model adds a file rather than overwriting the first fit.
    """
    p = Path(prompts)
    return str(p.with_name(f"{p.stem}{CKPT_SEP}{model_slug(model)}"))


def base_from_npz(path: str | Path, backend: str) -> str:
    """Invert :func:`dist_files`: the base prefix a saved ``.npz`` belongs to.

    Peels the tensor suffix then the backend suffix, so any one file of a
    multi-tensor fit identifies the whole set.
    """
    s = str(path)
    if s.endswith(".npz"):
        s = s[: -len(".npz")]
    for suffix in tensor_suffixes(backend):
        if suffix and s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    tail = f"_{backend}"
    if backend != "sd15" and s.endswith(tail):
        s = s[: -len(tail)]
    return s
