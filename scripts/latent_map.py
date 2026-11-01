#!/usr/bin/env python3
"""Build the latent map: every explored image placed in PCA space, as JSON.

Reads each image's ``.npz`` conditioning sidecar (+ its ``.json`` params), the
backend's fitted distribution, and the prompt corpus, and writes
``outputs/latent_map_<backend>.json`` for the dashboard's 🌌 Map tab:

* nodes   -- per image: corpus-frame + explored-frame coordinates (``--dims``
             axes each), off-manifold share, distance gauge, sampler, lineage
* edges   -- explore lineage (``parent`` -> child, ``parent_b`` for breeds)
* corpus  -- the distribution's retained corpus embeddings, projected the same
             way and labelled with their prompt text when it can be matched
* axes    -- variance share per axis + the corpus prompts at each extreme

Torch-free (NumPy only) -- runs on the dashboard host in a second or two::

    python scripts/latent_map.py --backend sd15
    python scripts/latent_map.py --backend sdxl --dims 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.backend import dist_backend
from semantic_anarchy.latent_map import (
    axis_extremes, corpus_indices, explained, explored_frame, off_manifold,
    project, sketch, sketch_matrix,
)

CHUNK = 256


def read_prompts(path: Path) -> list[str]:
    """Same filter as mine_distribution.py, so row i is prompt i."""
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def find_nodes(outputs: Path, backend: str, tensor: str, shape: tuple) -> list[dict]:
    """Images with an ``.npz`` conditioning sidecar for this backend, oldest first."""
    nodes = []
    for npz in outputs.rglob("*.npz"):
        png = npz.with_suffix(".png")
        if not png.is_file():
            continue          # distributions, evolved branches, other caches
        meta = {}
        js = npz.with_suffix(".json")
        if js.is_file():
            try:
                meta = json.loads(js.read_text())
            except Exception:
                pass
        if meta.get("backend", backend) != backend:
            continue
        nodes.append({"npz": npz, "png": png, "meta": meta, "mtime": png.stat().st_mtime})
    nodes.sort(key=lambda n: n["mtime"])
    return nodes


def load_tensor(npz: Path, tensor: str, shape: tuple):
    try:
        with np.load(npz) as z:
            if tensor not in z:
                return None
            x = np.asarray(z[tensor], dtype=np.float32)
    except Exception:
        return None
    return x if x.shape == shape else None


def lineage(nodes: list[dict]) -> list[dict]:
    """Explore sidecars name their parent(s) by file name; resolve each to the
    newest same-named image older than the child (names repeat across folders)."""
    by_name: dict[str, list[int]] = {}
    for i, n in enumerate(nodes):
        by_name.setdefault(n["png"].name, []).append(i)
    def resolve(name, i):
        older = [j for j in by_name.get(Path(name).name, []) if j != i
                 and nodes[j]["mtime"] <= nodes[i]["mtime"]]
        return older[-1] if older else None

    edges, walks = [], {}
    for i, n in enumerate(nodes):
        m = n["meta"]
        if m.get("walk_frame"):
            # A walk is one path: parent -> frame 1 -> frame 2 -> ...
            walks.setdefault((m.get("parent"), m.get("batch_seed")), []).append(i)
            continue
        for key, kind in (("parent", "parent"), ("parent_b", "breed")):
            j = resolve(m[key], i) if m.get(key) else None
            if j is not None:
                edges.append({"a": j, "b": i, "kind": kind})
    for (parent, _), frames in walks.items():
        frames.sort(key=lambda i: nodes[i]["meta"]["walk_frame"])
        prev = resolve(parent, frames[0]) if parent else None
        for i in frames:
            if prev is not None:
                edges.append({"a": prev, "b": i, "kind": "walk"})
            prev = i
    return edges


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="sd15",
                   choices=["sd15", "sd2", "sdxl", "flux2", "krea2"])
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    p.add_argument("--dist", type=Path, default=None,
                   help="distribution prefix (default: the backend's outputs/dist*)")
    p.add_argument("--prompts", type=Path, default=Path("prompts_1000.txt"),
                   help="the corpus the distribution was mined from (axis labels)")
    p.add_argument("--dims", type=int, default=8, help="axes exported per frame")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    t0 = time.time()

    be = dist_backend(args.backend)
    tensor = be.tensor_names[0]
    base = args.dist or args.outputs / "dist"
    prefix = base if args.backend == "sd15" or args.dist else Path(f"{base}_{args.backend}")
    try:
        dist = be.load_dists(prefix)[tensor]
    except FileNotFoundError:
        print(f"no {args.backend} distribution at {prefix} -- mine one first",
              file=sys.stderr)
        return 3
    if dist.pca_components is None:
        print(f"[map] {prefix}: distribution has no PCA -- re-mine it", file=sys.stderr)
        return 2
    shape = tuple(dist.feature_shape)
    dims = min(args.dims, len(dist.pca_std))
    print(f"[map] {args.backend}: {tensor}{shape}, {len(dist.pca_std)} corpus axes from {prefix}")

    nodes = find_nodes(args.outputs, args.backend, tensor, shape)
    r = sketch_matrix(dist)
    coords, sk, off, kept = [], [], [], []
    for i in range(0, len(nodes), CHUNK):
        batch = [(j, load_tensor(nodes[j]["npz"], tensor, shape))
                 for j in range(i, min(i + CHUNK, len(nodes)))]
        batch = [(j, x) for j, x in batch if x is not None]
        if not batch:
            continue
        x = np.stack([b[1] for b in batch])
        coords.append(project(dist, x, dims))
        sk.append(sketch(dist, x, r))
        off.append(off_manifold(dist, x))
        kept += [b[0] for b in batch]
    nodes = [nodes[j] for j in kept]
    coords = np.concatenate(coords) if coords else np.zeros((0, dims))
    off = np.concatenate(off) if off else np.zeros(0)
    ex_coords, ex_share = explored_frame(np.concatenate(sk) if sk else np.zeros((0, 1)), dims)
    print(f"[map] {len(nodes)} explored image(s) projected")

    # Corpus dust: the retained raw corpus embeddings, labelled when the prompt
    # file lines up with the fit (same count -> fit's seeded subset replays).
    prompts = read_prompts(args.prompts)
    corpus = dist.corpus_embeddings
    c_coords = project(dist, corpus, dims) if corpus is not None else np.zeros((0, dims))
    if len(prompts) == dist.n_samples and corpus is not None:
        labels = [prompts[i] for i in corpus_indices(dist.n_samples, len(corpus))]
    else:
        labels = [f"corpus #{i}" for i in range(len(c_coords))]
        if prompts:
            print(f"[map] {args.prompts} has {len(prompts)} prompts, the fit saw "
                  f"{dist.n_samples} -- corpus points left unlabelled")

    def rnd(a, nd=3):
        return np.round(np.asarray(a, dtype=np.float64), nd).tolist()

    out_nodes = []
    for i, n in enumerate(nodes):
        m = n["meta"]
        out_nodes.append({
            "rel": n["png"].relative_to(args.outputs).as_posix(),
            "name": n["png"].name,
            "t": round(n["mtime"], 1),
            "kind": m.get("kind"), "mode": m.get("mode"),
            "sampler": m.get("sampler"), "temperature": m.get("temperature"),
            "distance": m.get("distance"),
            "off": round(float(off[i]), 3),
            "c": rnd(coords[i]), "e": rnd(ex_coords[i]),
        })
    doc = {
        "backend": args.backend, "tensor": tensor, "dims": dims,
        "dist": str(prefix), "built": time.time(),
        "axes": {
            "corpus": {"share": rnd(explained(dist, dims), 4),
                       "extremes": axis_extremes(c_coords, labels) if len(labels) else []},
            "explored": {"share": rnd(ex_share, 4)},
        },
        "nodes": out_nodes,
        "edges": lineage(nodes),
        "corpus": [{"label": labels[i], "c": rnd(c_coords[i])} for i in range(len(c_coords))],
    }
    out = args.out or args.outputs / f"latent_map_{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"[map] wrote {out} ({len(out_nodes)} nodes, {len(doc['edges'])} edges, "
          f"{len(doc['corpus'])} corpus points) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
