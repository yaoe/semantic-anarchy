#!/usr/bin/env python3
"""Build the latent map: every explored image placed in 3D, as JSON.

Reads each image's ``.npz`` conditioning sidecar (+ its ``.json`` params), its
CLIP embedding (``outputs/clip_embeds.npz``, from resonance.py), the backend's
fitted distribution and the prompt corpus, and writes
``outputs/latent_map_<backend>[_<focus>].json`` for the dashboard's 🌌 Map tab:

* nodes    -- per image: its position in each frame (``l`` look, ``u``
              conditioning, ``c`` corpus PCA), region, thumbnail cell, gauges
* frames   -- for each frame, how far "near on the map" means "looks alike"
* regions  -- clusters of the look layout, with their most typical images
* edges    -- explore lineage (``parent`` -> child, ``parent_b`` for breeds)
* corpus   -- the distribution's retained corpus embeddings, in the corpus frame
* atlas    -- every image's thumbnail in one JPEG, so the map can draw images

No GPU, no model. An image with no CLIP embedding yet has no look position; the
document counts those in ``unembedded`` so the dashboard can queue resonance.py.
A rebuild continues from the previous layout, so images already on the map stay
roughly where they are::

    python scripts/latent_map.py --backend sd15
    python scripts/latent_map.py --backend sdxl --neighbors 60 --min-dist 0.4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy.backend import dist_backend
from semantic_anarchy.cli_args import dist_prefix
from semantic_anarchy.io_utils import find_image
from semantic_anarchy.latent_map import (
    axis_extremes, corpus_indices, explained, find_regions, neighbours_kept,
    off_manifold, project, region_faces, sketch, umap_3d,
)

CHUNK = 256
ATLAS_MAX = 4096     # px per side: every browser's safe WebGL texture size
ATLAS_TILE = 96


def write_text_atomic(path: Path, text: str) -> None:
    """The dashboard serves the last build while the next one runs, so a reader
    must only ever see a whole file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


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
        img = find_image(npz)  # the stem is the identity: png, jpg, whatever
        if img is None:
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
        nodes.append({"npz": npz, "img": img, "meta": meta, "mtime": img.stat().st_mtime})
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
        by_name.setdefault(n["img"].name, []).append(i)
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


def load_clip(outputs: Path, rels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """``(has (n,), vecs (m, D))``: which nodes have a CLIP embedding, and those
    embeddings in node order."""
    has = np.zeros(len(rels), dtype=bool)
    f = outputs / "clip_embeds.npz"
    if not f.is_file():
        return has, np.zeros((0, 768), dtype=np.float32)
    with np.load(f, allow_pickle=False) as z:
        row = {str(name): i for i, name in enumerate(z["names"])}
        vecs = z["vecs"].astype(np.float32)
    idx = []
    for i, rel in enumerate(rels):
        if rel in row:
            has[i] = True
            idx.append(row[rel])
    return has, vecs[idx]


def read_previous(path: Path, focus_tag: str) -> dict:
    """The last build of this map, or {} -- what the new layout continues from."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return doc if doc.get("focus") == focus_tag else {}


def previous_layout(prev: dict, rels: list[str], key: str, dim: int = 3):
    """``(coords (n, dim), known (n,))`` of ``rels`` in the previous build."""
    old = {n["rel"]: n.get(key) for n in prev.get("nodes", [])}
    coords, known = np.zeros((len(rels), dim)), np.zeros(len(rels), dtype=bool)
    for i, rel in enumerate(rels):
        v = old.get(rel)
        if v is not None and len(v) == dim:
            coords[i], known[i] = v, True
    return coords, known


def build_atlas(nodes: list[dict], outputs: Path, jpg: Path) -> dict:
    """Every node's thumbnail in one JPEG. Keeps the existing atlas' order and
    reuses its pixels, so only new images (and earlier failures) are decoded.
    ``broken`` names files that can't be read: no embedding is coming for those."""
    from PIL import Image

    index = jpg.with_suffix(".json")
    rels = [n["img"].relative_to(outputs).as_posix() for n in nodes]
    broken = set()
    old: dict = {}
    old_img = None
    try:
        old = json.loads(index.read_text())
        old_img = Image.open(jpg).convert("RGB")
    except (OSError, ValueError):
        old, old_img = {}, None
    now = set(rels)
    order = [r for r in old.get("rels", []) if r in now]
    order += [r for r in rels if r not in set(order)]
    if not order:
        return {"rels": [], "cols": 1, "tile": ATLAS_TILE, "broken": []}
    cols = int(np.ceil(np.sqrt(len(order))))
    tile = min(ATLAS_TILE, ATLAS_MAX // cols)
    sheet = Image.new("RGB", (cols * tile, cols * tile), (12, 12, 14))
    old_cell = {r: i for i, r in enumerate(old.get("rels", []))} if old_img is not None else {}
    paths = {r: n["img"] for r, n in zip(rels, nodes)}
    for i, rel in enumerate(order):
        x, y = (i % cols) * tile, (i // cols) * tile
        if rel in old_cell and rel not in old.get("broken", []):
            j, ot, oc = old_cell[rel], old["tile"], old["cols"]
            box = ((j % oc) * ot, (j // oc) * ot, (j % oc + 1) * ot, (j // oc + 1) * ot)
            sheet.paste(old_img.crop(box).resize((tile, tile)), (x, y))
            continue
        try:
            with Image.open(paths[rel]) as im:
                im.draft("RGB", (tile * 2, tile * 2))
                im = im.convert("RGB")
                side = min(im.size)
                left, top = (im.width - side) // 2, (im.height - side) // 2
                sheet.paste(im.crop((left, top, left + side, top + side)).resize(
                    (tile, tile), Image.LANCZOS), (x, y))
        except OSError as exc:      # one half-written file costs one dark cell
            broken.add(rel)
            print(f"[map] no thumbnail for {rel}: {exc}", file=sys.stderr)
    jpg.parent.mkdir(parents=True, exist_ok=True)
    tmp = jpg.with_name(jpg.name + ".tmp")
    sheet.save(tmp, format="JPEG", quality=82, optimize=True)
    os.replace(tmp, jpg)
    meta = {"rels": order, "cols": cols, "tile": tile, "broken": sorted(broken)}
    write_text_atomic(index, json.dumps(meta))
    return meta


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="sd15",
                   choices=["sd15", "sd2", "sdxl", "flux2", "krea2"])
    p.add_argument("--outputs", type=Path, default=Path("outputs"))
    p.add_argument("--dist", type=Path, default=None,
                   help="distribution prefix (default: the backend's outputs/dist*)")
    p.add_argument("--prompts", type=Path, default=Path("prompts_1000.txt"),
                   help="the corpus the distribution was mined from (axis labels)")
    p.add_argument("--neighbors", type=int, default=20,
                   help="UMAP n_neighbors: low = tight islands, high = global shape")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="UMAP min_dist: how tightly neighbours may pack")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    t0 = time.time()

    be = dist_backend(args.backend)
    tensor = be.tensor_names[0]
    prefix = dist_prefix(args, str(args.dist or args.outputs / "dist"))
    try:
        dist = be.load_dists(prefix)[tensor]
    except FileNotFoundError:
        print(f"no {args.backend} distribution at {prefix} -- mine one first",
              file=sys.stderr)
        return 3
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"{prefix}: unreadable distribution ({exc}) -- re-mine it", file=sys.stderr)
        return 2
    if dist.pca_components is None:
        print(f"[map] {prefix}: distribution has no PCA -- re-mine it", file=sys.stderr)
        return 2
    shape = tuple(dist.feature_shape)
    dims = min(3, len(dist.pca_std))
    print(f"[map] {args.backend}: {tensor}{shape}, {len(dist.pca_std)} corpus axes from {prefix}")

    nodes = find_nodes(args.outputs, args.backend, tensor, shape)
    coords, sk, off, kept = [], [], [], []
    for i in range(0, len(nodes), CHUNK):
        batch = [(j, load_tensor(nodes[j]["npz"], tensor, shape))
                 for j in range(i, min(i + CHUNK, len(nodes)))]
        batch = [(j, x) for j, x in batch if x is not None]
        if not batch:
            continue
        x = np.stack([b[1] for b in batch])
        coords.append(project(dist, x, dims))
        sk.append(sketch(dist, x))
        off.append(off_manifold(dist, x))
        kept += [b[0] for b in batch]
    nodes = [nodes[j] for j in kept]
    coords = np.concatenate(coords) if coords else np.zeros((0, dims))
    sketches = np.concatenate(sk) if sk else np.zeros((0, 1))
    off = np.concatenate(off) if off else np.zeros(0)
    rels = [n["img"].relative_to(args.outputs).as_posix() for n in nodes]
    print(f"[map] {len(nodes)} explored image(s) projected")

    out = args.out or args.outputs / f"latent_map_{args.backend}.json"
    focus_tag = f"{args.neighbors}:{args.min_dist}"
    prev = read_previous(out, focus_tag)
    has_clip, clip = load_clip(args.outputs, rels)
    warning = None
    look = np.full((len(nodes), 3), np.nan)
    cond = np.full((len(nodes), 3), np.nan)
    try:
        if has_clip.any():
            old, known = previous_layout(prev, [r for r, h in zip(rels, has_clip) if h], "l")
            look[has_clip] = umap_3d(clip, "cosine", args.neighbors, args.min_dist,
                                     prev=old if known.any() else None, known=known)
        if len(nodes):
            old, known = previous_layout(prev, rels, "u")
            cond = umap_3d(sketches, "euclidean", args.neighbors, args.min_dist,
                           prev=old if known.any() else None, known=known)
    except ImportError:
        warning = "umap-learn is not installed on the dashboard's python -- only the corpus frame is available"
        print(f"[map] {warning}", file=sys.stderr)
    has_look, has_cond = ~np.isnan(look[:, 0]), ~np.isnan(cond[:, 0])
    print(f"[map] layouts done at {time.time() - t0:.1f}s "
          f"({int(has_clip.sum())}/{len(nodes)} images have CLIP embeddings)")

    # Regions are clusters of what things LOOK like, and colour every frame: the
    # same colours scattered in another frame is that frame failing to see them.
    region_src, region_mask = (look, has_look) if has_look.any() else (cond, has_cond)
    labels = np.full(len(nodes), -1)
    faces: dict[int, list[int]] = {}
    if region_mask.any():
        old_lab = {n["rel"]: n.get("r", -1) for n in prev.get("nodes", [])}
        sub = [r for r, m in zip(rels, region_mask) if m]
        prior = np.array([old_lab.get(r, -1) for r in sub]) if prev else None
        lab = find_regions(region_src[region_mask], prior)
        labels[region_mask] = lab
        sub_idx = np.where(region_mask)[0]
        faces = {c: sub_idx[rows].tolist() for c, rows in
                 region_faces(region_src[region_mask], lab).items()}

    # Does near on the map mean looks alike? Measured against CLIP, for every frame.
    frames = {}
    for key, y, mask in (("look", look, has_look & has_clip),
                         ("cond", cond, has_cond & has_clip),
                         ("corpus", coords, has_clip)):
        if not mask.any():
            frames[key] = {"alike": None}
            continue
        src = clip[np.isin(np.where(has_clip)[0], np.where(mask)[0])]
        frames[key] = {"alike": neighbours_kept(src, y[mask], "cosine")}
    print("[map] neighbours that look alike: " + ", ".join(
        f"{k}={v['alike']:.3f}" if v["alike"] is not None else f"{k}=n/a" for k, v in frames.items()))

    atlas = build_atlas(nodes, args.outputs, out.parent / f"latent_map_{args.backend}.atlas.jpg")
    cell = {r: i for i, r in enumerate(atlas["rels"])}

    # Corpus dust: the retained raw corpus embeddings, labelled when the prompt
    # file lines up with the fit (same count -> fit's seeded subset replays).
    prompts = read_prompts(args.prompts)
    corpus = dist.corpus_embeddings
    c_coords = project(dist, corpus, dims) if corpus is not None else np.zeros((0, dims))
    labelled = len(prompts) == dist.n_samples and corpus is not None
    if labelled:
        labels_c = [prompts[i] for i in corpus_indices(dist.n_samples, len(corpus))]
    else:
        labels_c = [f"corpus #{i}" for i in range(len(c_coords))]
        if prompts:
            print(f"[map] {args.prompts} has {len(prompts)} prompts, the fit saw "
                  f"{dist.n_samples} -- corpus points left unlabelled")

    def rnd(a, nd=3):
        return np.round(np.asarray(a, dtype=np.float64), nd).tolist()

    out_nodes = []
    for i, n in enumerate(nodes):
        m = n["meta"]
        out_nodes.append({
            "rel": rels[i],
            "name": n["img"].name,
            "t": round(n["mtime"], 1),
            "kind": m.get("kind"), "mode": m.get("mode"),
            "sampler": m.get("sampler"), "temperature": m.get("temperature"),
            "distance": m.get("distance"),
            "off": round(float(off[i]), 3),
            "c": rnd(coords[i]),
            "l": rnd(look[i]) if has_look[i] else None,
            "u": rnd(cond[i]) if has_cond[i] else None,
            "r": int(labels[i]),
            "a": cell.get(rels[i], -1),
        })
    regions = [{"id": c, "count": int((labels == c).sum()), "faces": rows}
               for c, rows in sorted(faces.items(), key=lambda kv: -(labels == kv[0]).sum())]
    doc = {
        "backend": args.backend, "tensor": tensor, "dims": dims,
        "dist": str(prefix), "built": time.time(), "focus": focus_tag,
        "warning": warning,
        "unembedded": sum(1 for r, h in zip(rels, has_clip) if not h and r not in atlas["broken"]),
        "frames": frames,
        "regions": regions,
        "axes": {"corpus": {"share": rnd(explained(dist, dims), 4),
                            "extremes": axis_extremes(c_coords, labels_c) if labelled else []}},
        "atlas": {"cols": atlas["cols"], "tile": atlas["tile"]},
        "nodes": out_nodes,
        "edges": lineage(nodes),
        "corpus": [{"label": labels_c[i], "c": rnd(c_coords[i])} for i in range(len(c_coords))],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, json.dumps(doc, separators=(",", ":")))
    print(f"[map] wrote {out} ({len(out_nodes)} nodes, {len(regions)} regions, "
          f"{len(doc['edges'])} edges, {len(doc['corpus'])} corpus points) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
