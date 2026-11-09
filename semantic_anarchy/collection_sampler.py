"""Sample MORE images inside a hand-curated collection ("aesthetic universe").

The obvious move -- fit a Gaussian/PCA to the members' conditioning and sample
it (:mod:`semantic_anarchy.selection_fit`) -- does not work, and the reason is
geometric, not a tuning problem (measured; see
``docs/reference/collection-sampling.md``):

* Members of a visual collection are drawn independently, so in conditioning
  space they are as uncorrelated as N random vectors can be (pairwise cosine
  of their deviations from the collection mean is ``-1/(N-1)`` to three
  decimals). CLIP-image similarity and conditioning similarity are unrelated
  (rank correlation ~0.03 over the gallery): an aesthetic universe is NOT a
  region of conditioning space a mean and covariance can describe.
* So the collection mean is just the corpus centre (gauge ~0.3 vs ~1.4 for a
  member) and a PCA draw is a dense superposition of every member at ~1/sqrt(N)
  amplitude -- near none of them, and rendered with nobody's settings.

What does carry an image's look is (a) that member's own conditioning, and
(b) how it was rendered (checkpoint, size, negative). Every strategy here is
therefore **member-anchored**: each child picks one member as its anchor,
moves away from it by a controlled amount, and is rendered with that anchor's
settings. ``spread`` is the one knob:

* ``0``   -- the anchor's exact conditioning; only the noise seed is new
  (new composition, same family).
* ``1``   -- the collection's natural diversity: as far as the method's
  calibration says a child can go and still read as the same universe.
* ``> 1`` -- explore beyond its edges.

Torch-free: NumPy at module scope only (CLAUDE.md's two-tier rule). PIL is
imported lazily, and only to read a member's true pixel size when its sidecar
did not record one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .upscale import conditioning_source

#: The sidecar fields a child copies from its anchor to be rendered like it.
#: Order matters only for the log line.
RENDER_FIELDS = ("model", "steps", "guidance", "scheduler", "neg_mode", "negative",
                 "width", "height", "dist")


# --------------------------------------------------------------- members ----
@dataclass
class Member:
    """One collection image: its conditioning and how it was rendered."""

    image: Path                      # what the user picked (may be an upscale)
    source: Path                     # the ancestor that owns the .npz
    latents: dict                    # tensor name -> (77, 768) etc.
    meta: dict = field(default_factory=dict)
    size: tuple | None = None     # true (width, height) of ``source``

    @property
    def name(self) -> str:
        return self.image.name

    def render_settings(self) -> dict:
        """The render knobs this member was made with (``None`` = unknown).

        Size comes from the image itself: explore children before the replay
        fix recorded no width/height, and an upscale's size is not the size its
        conditioning was rendered at -- ``source`` is.
        """
        s = {k: self.meta.get(k) for k in RENDER_FIELDS}
        if self.size:
            s["width"], s["height"] = self.size
        return s


def _image_size(path: Path) -> tuple | None:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return tuple(im.size)
    except (OSError, ImportError, ValueError):
        return None


def load_members(
    images: Iterable[str | Path],
    tensor_names: Sequence[str] = ("embeds",),
    read_size: bool = True,
) -> tuple[list[Member], list[tuple[str, str]]]:
    """Resolve picked images to members; return ``(members, skipped)``.

    Follows ``refined_from`` (like :func:`selection_fit.latents_for`), so an
    upscale contributes its original's conditioning and size. Duplicates (the
    same source picked twice, e.g. an image and its upscale) are kept once. A
    sidecar missing a tensor or carrying a different shape than the first
    member is skipped with a reason, never fatal.
    """
    members: list[Member] = []
    skipped: list[tuple[str, str]] = []
    seen: set = set()
    shapes: dict = {}
    for raw in images:
        p = Path(raw)
        try:
            src, meta = conditioning_source(p)
        except FileNotFoundError:
            skipped.append((str(raw), "no conditioning sidecar"))
            continue
        key = src.resolve()
        if key in seen:
            skipped.append((str(raw), f"duplicate of {src.name}"))
            continue
        try:
            with np.load(src.with_suffix(".npz")) as data:
                lat = {n: np.asarray(data[n], dtype=np.float32)
                       for n in tensor_names if n in data.files}
        except (OSError, ValueError) as exc:
            skipped.append((str(raw), f"unreadable ({exc.__class__.__name__})"))
            continue
        missing = [n for n in tensor_names if n not in lat]
        if missing:
            skipped.append((str(raw), f"no {'/'.join(missing)} tensor"))
            continue
        bad = next((n for n in tensor_names
                    if n in shapes and lat[n].shape != shapes[n]), None)
        if bad is not None:
            why = f"{bad} is {lat[bad].shape}, expected {shapes[bad]}"
            skipped.append((str(raw), why))
            continue
        for n in tensor_names:
            shapes.setdefault(n, lat[n].shape)
        seen.add(key)
        size = _image_size(src) if read_size else None
        if size is None and meta.get("width") and meta.get("height"):
            size = (int(meta["width"]), int(meta["height"]))
        members.append(Member(image=p, source=src, latents=lat, meta=meta, size=size))
    return members, skipped


def read_image_list(path: str | Path) -> list[str]:
    """Image paths from a JSON list / ``{"images": [...]}`` / a text file."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("images", data.get("sources", []))
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list of image paths")
        return [str(x) for x in data]
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def group_by_checkpoint(settings: Sequence[dict]) -> dict:
    """``{checkpoint: [child index, ...]}`` in first-seen order.

    Takes each child's *resolved* render settings (after any missing-checkpoint
    fallback), so the CLI loads every checkpoint exactly once.
    """
    groups: dict = {}
    for i, s in enumerate(settings):
        groups.setdefault(s.get("model"), []).append(i)
    return groups


# ------------------------------------------------------------ the sampler ---
#: Strategies, in the order the CLI lists them. ``auto`` resolves to
#: :data:`AUTO_METHOD`.
METHODS = ("member", "slerp", "jitter", "crossover")
AUTO_METHOD = "slerp"

#: slerp draws t ~ U(FLOOR, 1) * spread/2: never (almost) zero, so a child
#: rendered at its anchor's own noise seed is a variation, not a re-render.
SLERP_T_FLOOR = 0.25

#: The spread the CLI defaults to: t in [0.06, 0.25] toward a partner member.
DEFAULT_SPREAD = 0.5

#: Shrinkage pseudo-count for the per-coordinate collection std: with N members
#: the estimate is blended ``N/(N+k)`` with its per-row average, so a 6-image
#: collection does not hand a near-zero sigma to a coordinate that happened to
#: agree across those six.
STD_SHRINK = 8.0


@dataclass
class Child:
    """One sampled conditioning set and where it came from."""

    latents: dict                    # tensor name -> one embedding
    anchor: int                      # index into the members list
    partner: int | None = None    # second member (slerp / crossover)
    t: float = 0.0                   # how far toward the partner / noise
    method: str = "member"


class Collection:
    """The members' conditioning, centred on the collection mean.

    Everything is measured in the collection's OWN frame -- deviations from its
    mean, scaled by its own per-coordinate spread -- so a member drawn at
    temperature 1.7 carries that spread in and nothing compounds it.
    """

    def __init__(self, members: Sequence, tensor_names: Sequence[str] | None = None):
        if tensor_names is None:
            first = members[0]
            tensor_names = list(first.latents if isinstance(first, Member) else first)
        self.names = list(tensor_names)
        lat = [m.latents if isinstance(m, Member) else m for m in members]
        self.n = len(lat)
        if self.n < 1:
            raise ValueError("a collection needs at least one member")
        self.stack = {k: np.stack([np.asarray(x[k], np.float64) for x in lat])
                      for k in self.names}
        self.mean = {k: v.mean(0) for k, v in self.stack.items()}
        self.dev = {k: v - self.mean[k] for k, v in self.stack.items()}
        self.sd = {}
        for k, v in self.stack.items():
            var = v.var(0, ddof=1) if self.n > 1 else np.ones(v.shape[1:])
            row = var.mean(-1, keepdims=True) if var.ndim > 1 else np.full_like(var, var.mean())
            lam = self.n / (self.n + STD_SHRINK)
            sd = np.sqrt(lam * var + (1 - lam) * row)
            self.sd[k] = np.maximum(sd, 0.1 * float(sd.mean()) + 1e-12)

    def member(self, i: int) -> dict:
        return {k: self.stack[k][i].astype(np.float32) for k in self.names}


def choose_anchors(n_members: int, n: int, rng) -> list[int]:
    """Balanced anchors: walk shuffled passes over the members, so a batch of
    n >= N touches every member before any repeats."""
    out: list[int] = []
    while len(out) < n:
        out.extend(int(i) for i in rng.permutation(n_members))
    return out[:n]


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation of two deviation vectors (norm-preserving when
    |a| == |b|; the members' deviations are near-orthogonal, so this is the
    move that keeps a child at a member's own distance from the centre instead
    of shrinking it toward the bland mean the way a straight lerp does)."""
    af, bf = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(af), np.linalg.norm(bf)
    if na == 0 or nb == 0:
        return ((1 - t) * af + t * bf).reshape(a.shape)
    om = float(np.arccos(np.clip(af @ bf / (na * nb), -1.0, 1.0)))
    if om < 1e-6:
        return ((1 - t) * af + t * bf).reshape(a.shape)
    v = (np.sin((1 - t) * om) * af + np.sin(t * om) * bf) / np.sin(om)
    return v.reshape(a.shape)


def _rotate(d: np.ndarray, eps: np.ndarray, angle: float, keep_rows=()) -> np.ndarray:
    """Turn deviation ``d`` toward random direction ``eps`` by ``angle`` radians,
    keeping its length. ``keep_rows`` (e.g. the constant BOS row) never move."""
    e = eps.copy()
    for r in keep_rows:
        e[r] = 0.0
    df, ef = d.ravel(), e.ravel()
    nd = np.linalg.norm(df)
    if nd == 0:
        return d
    ef = ef - df * (df @ ef) / (nd * nd)
    ne = np.linalg.norm(ef)
    if ne == 0:
        return d
    ef *= nd / ne
    return (np.cos(angle) * df + np.sin(angle) * ef).reshape(d.shape)


def _noise(coll: Collection, k: str, rng) -> np.ndarray:
    return rng.standard_normal(coll.sd[k].shape) * coll.sd[k]


def sample(coll: Collection, method: str, spread: float, n: int, rng=None,
           anchors: Sequence[int] | None = None) -> list[Child]:
    """Draw ``n`` children, each anchored on one member.

    ``spread`` (>= 0) per method:

    * ``member``    -- ignored: the anchor's exact conditioning.
    * ``slerp``     -- t ~ U(spread/8, spread/2) toward a random partner member
      (``spread=1`` reaches the midpoint; the floor keeps an anchor-seeded child
      from being a near-copy of its anchor); above 1 the result is additionally
      rotated toward collection-shaped noise by ``atan(spread-1)``.
    * ``jitter``    -- rotate the anchor toward collection-shaped noise by
      ``atan(spread)`` (``spread=1`` is 45 deg), keeping its distance from the
      collection centre.
    * ``crossover`` -- each token row is swapped for a random other member's
      same-position row with probability ``spread/2`` (capped at 0.9).

    Row 0 (CLIP's BOS, constant across every prompt) is never touched.
    """
    if method == "auto":
        method = AUTO_METHOD
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r} (choose from {METHODS})")
    rng = rng or np.random.default_rng()
    spread = max(0.0, float(spread))
    if anchors is None:
        anchors = choose_anchors(coll.n, n, rng)
    others = lambda a: [j for j in range(coll.n) if j != a]
    out = []
    for a in anchors[:n]:
        a = int(a)
        devs = {k: coll.dev[k][a] for k in coll.names}
        partner, t = None, 0.0
        if method == "slerp" and coll.n > 1 and spread > 0:
            partner = int(rng.choice(others(a)))
            t = float(rng.uniform(SLERP_T_FLOOR, 1.0)) * 0.5 * min(spread, 1.0)
            devs = {k: _slerp(devs[k], coll.dev[k][partner], t) for k in coll.names}
            if spread > 1:
                ang = float(np.arctan(spread - 1.0))
                devs = {k: _rotate(devs[k], _noise(coll, k, rng), ang, _bos(devs[k]))
                        for k in coll.names}
        elif method == "jitter" and spread > 0:
            t = float(np.arctan(spread))
            devs = {k: _rotate(devs[k], _noise(coll, k, rng), t, _bos(devs[k]))
                    for k in coll.names}
        elif method == "crossover" and coll.n > 1 and spread > 0:
            t = min(0.5 * spread, 0.9)
            new = {}
            for k in coll.names:
                d = devs[k].copy()
                if d.ndim == 2:
                    rows = np.where(rng.random(d.shape[0]) < t)[0]
                    for r in rows:
                        if r == 0:
                            continue
                        d[r] = coll.dev[k][int(rng.choice(others(a)))][r]
                new[k] = d
            devs = new
        lat = {k: (coll.mean[k] + devs[k]).astype(np.float32) for k in coll.names}
        out.append(Child(latents=lat, anchor=a, partner=partner, t=round(t, 4),
                         method=method))
    return out


def _bos(d: np.ndarray) -> tuple:
    """Rows to keep fixed: the BOS row of a token-sequence tensor."""
    return (0,) if d.ndim == 2 else ()


#: ``fresh`` = new noise per child (new compositions); ``anchor`` = the
#: anchor's own ``image_seed`` (variations of that exact picture).
SEED_MODES = ("fresh", "anchor")


def resolve_seed_mode(mode: str, method: str, spread: float) -> str:
    """``auto`` -> a concrete seed mode.

    The init noise carries as much of an image's look as its conditioning does
    (same conditioning, new seed: CLIP cosine to the original only 0.67), so
    inside the collection (``spread <= 1``) a child reuses its anchor's noise --
    measured best at staying in the universe -- and past its edges (``> 1``)
    it gets fresh noise for new compositions. ``member`` at the anchor's own
    seed would just re-render the anchor, so it is always fresh.
    """
    if method == "member":
        return "fresh"
    if mode in SEED_MODES:
        return mode
    return "anchor" if spread <= 1.0 else "fresh"
