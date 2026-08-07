#!/usr/bin/env python3
"""Deep-dive the distributional statistics of a mined conditioning corpus.

Encodes (once, cached) a prompt corpus through the text encoder, then produces a
set of figures that answer the questions a *sampler designer* needs answered:

* where along the 77 token positions the variance actually lives
* whether per-coordinate sigma modulation has anything to bite on
* how non-Gaussian the marginals are, and *why* the non-Gaussian ones are
* how many principal axes carry structure rather than sampling noise
* how wrong coordinate independence is, and in which direction
* how much of any of it the UNet can even read (cross-attention to_k / to_v)
* what each shipped sampler actually produces, measured on all of the above

Run::

    python scripts/analyze_distribution.py --ckpt MODEL.safetensors      # first run
    python scripts/analyze_distribution.py --only 05,09                  # iterate

Everything lands in ``outputs/analysis/`` (plots + ``stats.json``). The corpus and
the heavy statistics are cached, so re-runs are seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from semantic_anarchy.analysis import (
    BOS, GAUSSIAN_CENTRAL_MASS, CorpusStats, block_participation, central_mass,
    cross_attn_sensitivity, cross_attn_weights, effective_dimension,
    encode_corpus, gmm_bic, jarque_bera, pad_gating, readable_variance_ratio,
)

# ------------------------------------------------------------------ style ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"

SEQ_BLUE = LinearSegmentedColormap.from_list("seqblue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("bluered", [
    "#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
    "#f6c9c9", "#e88a8a", "#d03b3b", "#7a1f1f"])

# Token regions carry one fixed identity across every figure.
REGION_COLORS = {0: VIOLET, 1: BLUE, 2: ORANGE}
REGION_NAMES = {0: "BOS (pos 0)", 1: "content", 2: "post-EOS pad"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.labelsize": 8.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 7,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "figure.dpi": 130,
})


def _ax(ax, title=None, xlabel=None, ylabel=None, grid="y"):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, axis=grid if grid in ("x", "y") else "both", alpha=0.9)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def _note(ax, text, x=0.98, y=0.95, color=INK, ha="right", va="top", size=8):
    """A SHORT callout pinned inside the axes -- for pointing, not explaining."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size,
            color=color, linespacing=1.35)


# Caption geometry, in inches. Captions are laid out only once the panels
# exist, so everything here is measured rather than guessed.
CAP_FS, CAP_LS = 8.0, 1.5
CAP_GAP_IN = 0.12       # a panel's lowest ink -> the top of its caption
CAP_PAD_IN = 0.20       # the bottom of a caption -> the next row's highest ink


def _caption(ax, text):
    """Register the finding for this panel; laid out by :func:`_finish`.

    Long-form reading is what this whole report is for, so the prose goes BELOW
    the panel where it can never collide with data. It is stashed rather than
    drawn because both the wrap width and the vertical room it needs depend on
    the panel's final size, which only exists after the layout has run.
    """
    ax._sa_caption = " ".join(text.split())


def _wrap(fig, ax, text):
    """Wrap a caption to its own panel's width (DejaVu Sans 8pt ~ 15.5 char/in)."""
    width_in = ax.get_window_extent().width / fig.dpi
    return textwrap.fill(text, max(38, int(width_in * 15.5)))


def _rows(fig):
    """Group the axes into visual rows, top-down; colorbars ride with theirs."""
    boxes = [(ax, ax.get_position()) for ax in fig.axes]
    rows, seen = [], set()
    for ax, box in sorted(boxes, key=lambda t: -t[1].y1):
        if ax in seen:
            continue
        row = [a for a, b in boxes if a not in seen and abs(b.y1 - box.y1) < 0.06]
        seen.update(row)
        rows.append(row)
    return rows


def _title_artist(ax):
    """The Text holding this axes' title -- ``ax.title`` only if it is centred."""
    for loc, attr in (("left", "_left_title"), ("center", "title"),
                      ("right", "_right_title")):
        if ax.get_title(loc=loc):
            return getattr(ax, attr, None)
    return None


def _fit_titles(fig):
    """Re-wrap any panel title that is wider than its own panel.

    Titles are left-aligned here, so an over-long one does not merely look
    cramped: it runs straight into the next panel's title.
    """
    rend = fig.canvas.get_renderer()
    for ax in fig.axes:
        obj = _title_artist(ax)
        if obj is None or "\n" in obj.get_text():
            continue
        raw = obj.get_text()
        avail = ax.get_window_extent(rend).width
        for _ in range(3):
            w = obj.get_window_extent(rend).width
            if w <= avail:
                break
            longest = max(len(ln) for ln in obj.get_text().split("\n"))
            obj.set_text(textwrap.fill(raw, max(8, int(longest * avail / w))))


def _reflow(fig, rows, top, bottom=0.03):
    """Re-deal the vertical space so every caption has room under its panel.

    ``tight_layout`` runs before the captions exist, so on a multi-row figure
    the top row's caption lands squarely on the bottom row's title. Each row's
    decorations (title above, ticks/xlabel/caption below) are a fixed height;
    only the data box itself is elastic, so measure the former and scale the
    latter to whatever is left.
    """
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    fh = fig.get_figheight()
    plan = []
    for row in rows:
        box = Bbox.union([ax.get_position() for ax in row])
        tight = Bbox.union([inv.transform_bbox(ax.get_tightbbox(rend))
                            for ax in row])
        lines = max([_wrap(fig, ax, ax._sa_caption).count("\n") + 1
                     for ax in row if getattr(ax, "_sa_caption", None)] or [0])
        cap = (lines * CAP_FS * CAP_LS / 72 + CAP_GAP_IN) / fh if lines else 0.0
        plan.append((row, box, max(0.0, tight.y1 - box.y1),
                     max(0.0, box.y0 - tight.y0), cap))

    fixed = sum(over + under + cap for _, _, over, under, cap in plan)
    elastic = sum(box.height for _, box, _, _, _ in plan)
    room = top - bottom - fixed - CAP_PAD_IN / fh * (len(plan) - 1)
    if elastic <= 0 or room <= 0:
        return
    scale, y = room / elastic, top
    for row, box, over, under, cap in plan:
        y -= over
        nt, nb = y, y - box.height * scale
        span = max(box.height, 1e-9)
        for ax in row:
            p = ax.get_position()
            ax.set_position([p.x0, nb + (p.y0 - box.y0) / span * (nt - nb),
                             p.width, max(p.height / span * (nt - nb), 1e-4)])
        y = nb - under - cap - CAP_PAD_IN / fh


def _draw_captions(fig, rows):
    """Hang each caption below its own panel, on a shared baseline per row."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    gap = CAP_GAP_IN / fig.get_figheight()
    for row in rows:
        if not any(getattr(ax, "_sa_caption", None) for ax in row):
            continue
        y = min(inv.transform_bbox(ax.get_tightbbox(rend)).y0 for ax in row) - gap
        for ax in row:
            text = getattr(ax, "_sa_caption", None)
            if text:
                fig.text(ax.get_position().x0, y, _wrap(fig, ax, text),
                         ha="left", va="top", fontsize=CAP_FS, color=INK2,
                         linespacing=CAP_LS)


def _finish(fig, top, layout=True, **kw):
    """Lay the figure out, fit the titles, then hang the captions underneath."""
    if layout:
        fig.tight_layout(rect=[0, 0, 1, top], **kw)
    fig.canvas.draw()
    _fit_titles(fig)                     # may add a line, so lay out again
    if layout:
        fig.tight_layout(rect=[0, 0, 1, top], **kw)
        fig.canvas.draw()
    rows = _rows(fig)
    if layout and len(rows) > 1:
        _reflow(fig, rows, top)
    _draw_captions(fig, rows)


def _header(fig, title, subtitle):
    """Title block measured in inches, so it never collides at any figure size."""
    h = fig.get_size_inches()[1]
    title, subtitle = title.format(**CORPUS), subtitle.format(**CORPUS)
    fig.text(0.012, 1 - 0.26 / h, title, ha="left", va="top", fontsize=15,
             fontweight="bold", color=INK)
    fig.text(0.012, 1 - 0.56 / h, subtitle, ha="left", va="top", fontsize=9.5,
             color=INK2)
    return 1 - 0.92 / h          # tight_layout rect top


def _fig(fn, nrows=1, ncols=1, figsize=(13.5, 4.8), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    top = _header(fig, fn.meta[1], fn.meta[2])
    return fig, axes, top


FIGURES = {}

# Figure titles and subtitles are declared at import time but quote the corpus's
# own shape, which is only known once it is loaded. They may therefore contain
# ``{n}`` / ``{T}`` / ``{H}`` / ``{D}`` / ``{k}`` placeholders, filled here.
CORPUS = {"n": 0, "T": 0, "H": 0, "D": 0, "k": 0}


def describe(S) -> None:
    """Publish the loaded corpus's shape to the title/subtitle namespace."""
    CORPUS.update(n=f"{S.n:,}", T=S.T, H=S.H, D=f"{S.D:,}", k=f"{S.n - 1:,}")


def expected_max_abs_z(n: int) -> float:
    """The |z| the largest of ``n`` standard-normal draws typically reaches.

    The median of the max-|z| order statistic: solve ``(2*Phi(z) - 1)^n = 0.5``.
    Grows with the corpus, so a fixed "3 sigma is a lot" rule of thumb gets
    looser as the corpus grows -- which is exactly the point figure 03 makes.
    """
    target = 0.5 ** (1.0 / max(n, 2))          # required 2*Phi(z) - 1
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.erf(mid / math.sqrt(2.0)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def figure(num, title, subtitle):
    def deco(fn):
        fn.meta = (num, title, subtitle)
        FIGURES[num] = fn
        return fn
    return deco


# =============================================================== figures ===
@figure("01", "Anatomy of the 77 token positions",
        "The sequence axis is not homogeneous. One position is frozen solid, one "
        "is wildly non-Gaussian, and the 'padding' holds most of the variance.")
def fig_sequence_anatomy(S, ctx):
    fig, axes, top = _fig(fig_sequence_anatomy, 2, 2, figsize=(13.5, 9.2))
    pos = np.arange(S.T)
    reg = S.token_region()
    med = int(np.median(S.eos_pos))
    last = int(S.eos_pos.max())

    ax = _ax(axes[0][0], "Per-position spread", "token position",
             f"sigma over the {S.n:,} prompts")
    ax.plot(pos, np.maximum(S.std.mean(axis=1), 1e-6), color=BLUE,
            label=f"mean over {S.H} channels")
    ax.plot(pos, np.maximum(S.std.max(axis=1), 1e-6), color=ORANGE, lw=1.4,
            label="widest channel")
    ax.set_yscale("log"); ax.set_ylim(3e-7, 12)
    ax.plot([0], [1e-6], "v", ms=9, color=VIOLET, zorder=5, clip_on=False)
    ax.annotate(f"BOS: sigma = 0.0 exactly\n(bit-identical in all {S.n:,} prompts)",
                (0, 1e-6), xytext=(7, 8e-6), fontsize=8, color=VIOLET,
                arrowprops=dict(arrowstyle="-", color=VIOLET, lw=0.9))
    ax.axvline(last, color=MUTED, lw=1, ls=":")
    ax.legend(loc="center right")
    _caption(ax, f"Position 0 is a constant: its {S.H} coordinates have exactly zero "
                 f"variance, so they are already\nfrozen and nothing a sampler does "
                 f"to them matters. Every other position sits within a factor of ~2\n"
                 f"of the same sigma -- including all positions past {last}, where "
                 f"no prompt has content left.")

    ax = _ax(axes[0][1], "Share of total corpus variance per position",
             "token position", "% of total variance")
    share = (S.std ** 2).sum(axis=1) / (S.std ** 2).sum() * 100
    ax.bar(pos, share, color=[REGION_COLORS[r] for r in reg], width=0.9)
    ax.set_ylim(0, share.max() * 1.35)
    tail = share[med:].sum()
    always_pad = share[last:].sum()
    # Proxy handles: an empty bar container legends as a default-coloured patch,
    # which would show all three regions in the same blue.
    ax.legend(handles=[Patch(color=REGION_COLORS[r], label=REGION_NAMES[r])
                       for r in (0, 1, 2)], loc="upper center", ncol=3)
    # How big the always-padding tail is depends entirely on how long this
    # corpus's prompts are: a corpus of short tag-style prompts leaves a wide
    # tail, one that fills the 77-token window leaves none. Say which it is.
    tail_note = (f"The {S.T - last} positions that are ALWAYS padding still hold "
                 f"{always_pad:.0f}% of the corpus's\ntotal variance, and everything"
                 if S.T - last > 1 else
                 f"This corpus reaches position {last}, so there is no always-padding "
                 f"tail to speak of --\nbut everything")
    _caption(ax, f"The padding is not filler. CLIP's text encoder is causal, so a "
                 f"row at position 60 has attended to\nthe entire prompt -- it is a "
                 f"running summary, not a blank. {tail_note} past the median EOS "
                 f"(position {med}) holds {tail:.0f}% of it.")

    ax = _ax(axes[1][0], "Magnitude: the shared mean vs the per-prompt deviation",
             "token position", f"L2 norm of the ({S.H},) row")
    dev_norm = np.linalg.norm(S.X - S.mean[None], axis=2)
    mean_norm = np.linalg.norm(S.mean, axis=1)
    ax.plot(pos, mean_norm, color=INK, label="||mean row||")
    ax.plot(pos, dev_norm.mean(axis=0), color=BLUE, label="mean ||deviation||")
    ax.fill_between(pos, np.percentile(dev_norm, 5, axis=0),
                    np.percentile(dev_norm, 95, axis=0), color=BLUE, alpha=0.18,
                    lw=0)
    ax.set_ylim(0, max(np.percentile(dev_norm, 95, axis=0).max(),
                       mean_norm.max()) * 1.22)
    ax.legend(loc="upper right")
    ratio = float(np.median(dev_norm.mean(axis=0)[4:] / np.maximum(mean_norm[4:], 1e-9)))
    _caption(ax, f"Past position 3 a prompt's deviation from the mean is about "
                 f"{ratio:.1f}x the mean itself. The corpus\nis a broad ball around "
                 f"mu, not a narrow cone -- so 'the average prompt' is not a strong "
                 f"attractor,\nand moving a full sigma away is an ordinary step "
                 f"rather than an extreme one.")

    ax = _ax(axes[1][1], "Non-Gaussianity along the sequence", "token position",
             f"mean over {S.H} channels")
    kmean = S.kurt.mean(axis=1)
    ax.plot(pos, np.abs(S.skew).mean(axis=1), color=BLUE, label="mean |skew|")
    ax.plot(pos, kmean, color=ORANGE, label="mean excess kurtosis")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(-2, max(4.0, kmean.max() * 2.2))
    ax.axhline(0, color=MUTED, lw=0.8)
    worst = int(np.argmax(kmean))
    k1 = float(kmean[worst])
    ax.annotate(f"position {worst}: mean excess kurtosis {k1:.0f}", (worst, k1),
                xytext=(9, k1 * 0.5), fontsize=8, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    ax.legend(loc="center right")
    second = float(np.sort(kmean)[-2])
    _caption(ax, f"One position carries most of the sequence's shape anomaly: "
                 f"position {worst}, at mean excess\nkurtosis {k1:.0f} against "
                 f"{second:.1f} for the next worst. Position 1 is the first content "
                 f"token, so\nwhenever a corpus reuses a small set of opening words "
                 f"its marginals go nearly categorical --\nspikes with long tails, "
                 f"not bells. Everywhere else is close to Gaussian on average.")

    _finish(fig, top)
    ctx["stats"].update(variance_share_after_median_eos=float(tail),
                        variance_share_always_pad=float(always_pad),
                        bos_sigma=float(S.std[BOS].mean()),
                        n_prompts=int(S.n), n_coords=int(S.D),
                        seq_len=int(S.T), n_channels=int(S.H),
                        total_var=float(S.total_var),
                        last_content_pos=last, median_eos_pos=med,
                        n_always_pad=int(S.T - last),
                        most_kurtotic_pos=worst,
                        pos1_mean_excess_kurtosis=float(kmean[1]),
                        max_pos_mean_excess_kurtosis=k1)
    return fig


@figure("02", "The sigma landscape is flat -- there is nothing to freeze but BOS",
        "The premise of per-coordinate sigma modulation is that coordinates "
        "differ wildly in spread. Across {D} of them, they barely differ at all.")
def fig_sigma_landscape(S, ctx):
    fig, axes, top = _fig(fig_sigma_landscape, 1, 4, figsize=(15, 5.4))

    ax = _ax(axes[0], "Every coordinate's sigma", "sigma", "coordinates")
    sd = S.std.ravel()
    ax.hist(sd[sd > 0], bins=140, color=BLUE, alpha=0.9, lw=0)
    ax.axvline(np.median(sd[sd > 0]), color=ORANGE, lw=1.6)
    ax.set_yscale("log")
    ax.plot([0], [1], "v", ms=9, color=VIOLET, clip_on=False, zorder=5)
    # High up and to the right: the only part of this panel with no bars in it.
    ax.annotate(f"{int((sd == 0).sum())} coordinates at\nsigma = 0 (the BOS row)",
                (0.02, 1.4), xytext=(sd.max() * 0.55, 700), fontsize=8,
                color=VIOLET,
                arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1.0))
    p1, p99 = np.percentile(sd[sd > 0], [1, 99])
    _caption(ax, f"One narrow bulk and one spike at zero. 98% of live coordinates "
                 f"fall between sigma {p1:.2f} and {p99:.2f}\n-- a factor of "
                 f"{p99 / p1:.1f}, where the intuition behind per-coordinate sigma "
                 f"modulation expects orders of\nmagnitude. There is no population "
                 f"of 'much wider' or 'much narrower' coordinates to exploit.")

    ax = _ax(axes[1], "Sorted per-coordinate sigma",
             f"coordinate rank (of {S.D:,})", "sigma")
    v = np.sort(S.std.ravel())[::-1]
    ax.plot(np.arange(v.size), v, color=BLUE)
    ax.set_ylim(0, v.max() * 1.05)
    ax.axvspan(v.size - S.H, v.size, color=VIOLET, alpha=0.3, lw=0)
    ax.annotate(f"the only cliff is exactly\n{S.H} coordinates wide:\n"
                f"it IS the BOS row",
                (v.size - S.H, v.max() * 0.2), xytext=(0.30 * v.size, v.max() * 0.85),
                fontsize=8, color=VIOLET,
                arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1))
    _caption(ax, "Sorted, the landscape is a gentle ramp, not a cliff -- there is no "
                 "natural cut point at which\ncoordinates stop mattering. The single "
                 "exception is position 0, and the fit already handles it:\nsigma = 0 "
                 "means the sampler reproduces BOS exactly without being told to.")

    ax = _ax(axes[2], "Cumulative variance vs coordinates",
             "% of coordinates (sorted by variance)", "% of total variance",
             grid="both")
    cv = np.sort(S.coord_var().ravel())[::-1]
    cum = np.cumsum(cv) / cv.sum()
    x = np.arange(1, cv.size + 1) / cv.size * 100
    ax.plot(x, cum * 100, color=BLUE, label="corpus")
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1.2, ls=":",
            label="perfectly flat sigma")
    marks = {}
    for frac, col in ((0.5, AQUA), (0.9, ORANGE)):
        i = int(np.searchsorted(cum, frac))
        marks[frac] = (i + 1) / cv.size * 100
        ax.plot([marks[frac]], [frac * 100], "o", ms=7, color=col, zorder=5)
    ax.legend(loc="upper left")
    # The curve fills the upper-left half of the panel; the two callouts go in
    # the empty lower-right triangle, colour-matched to their markers, rather
    # than on top of the line they are describing.
    for i, (frac, col) in enumerate(((0.9, ORANGE), (0.5, AQUA))):
        ax.text(0.98, 0.05 + i * 0.13,
                f"{frac:.0%} of the variance\nneeds {marks[frac]:.0f}% of coordinates",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color=col, linespacing=1.35)
    _caption(ax, "For comparison: in a typical CLIP *image* embedding this curve "
                 "hugs the top-left corner. Here it\nbarely leaves the diagonal. "
                 "Dropping the narrowest half of all coordinates would still cost\n"
                 f"{100 - cum[cv.size // 2] * 100:.0f}% of the corpus's variance.")

    ax = _ax(axes[3], "Effective coordinate count", "", "participation ratio")
    pr = effective_dimension(cv)
    bars = {"corpus\ncoordinates": pr, f"all {cv.size:,}\n(flat sigma)": float(cv.size)}
    ax.bar(list(bars), list(bars.values()), color=[BLUE, MUTED], width=0.5)
    for i, val in enumerate(bars.values()):
        ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=11,
                color=INK)
    ax.set_ylim(0, cv.size * 1.32)
    _caption(ax, f"The participation ratio says the corpus uses {pr / cv.size:.0%} of a "
                 f"perfectly uniform variance budget.\nThe conclusion for sampler "
                 f"design is a negative one, and worth stating plainly: reshaping "
                 f"sigma\nper coordinate has almost nothing to bite on. The leverage "
                 f"is in SHAPE (fig 03-05) and\nCORRELATION (fig 09), not in "
                 f"per-coordinate scale.")

    _finish(fig, top)
    ctx["stats"].update(coord_pct_for_90pct_var=marks[0.9],
                        coord_participation_ratio=pr,
                        coord_participation_frac=float(pr / cv.size),
                        sigma_p1=float(p1), sigma_p99=float(p99),
                        sigma_spread=float(p99 / p1),
                        var_lost_dropping_narrow_half=float(
                            100 - cum[cv.size // 2] * 100))
    return fig


@figure("03", "Marginal shape: how far from Gaussian, and where",
        "Per coordinate: how far the empirical marginal departs from the Gaussian "
        "the diagonal sampler substitutes for it.")
def fig_marginal_shape(S, ctx):
    fig, axes, top = _fig(fig_marginal_shape, 1, 4, figsize=(15, 5.4))
    live = S.live_mask(1e-3)
    sk, ku = S.skew[live], S.kurt[live]
    se_s, se_k = 2 * np.sqrt(6 / S.n), 2 * np.sqrt(24 / S.n)

    ax = _ax(axes[0], "Skew vs excess kurtosis", "skewness", "excess kurtosis",
             grid="both")
    hb = ax.hexbin(sk, ku, gridsize=80, cmap=SEQ_BLUE, bins="log", mincnt=1,
                   extent=(-4, 4, -3, 12))
    fig.colorbar(hb, ax=ax, pad=0.01, label="coordinates")
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
    ax.add_patch(plt.Rectangle((-se_s, -se_k), 2 * se_s, 2 * se_k, fill=False,
                               ec=ORANGE, lw=1.6, ls="--", zorder=5))
    inside = float(((np.abs(sk) < se_s) & (np.abs(ku) < se_k)).mean() * 100)
    off = float(((np.abs(sk) > 4) | (ku > 12) | (ku < -3)).mean() * 100)
    # Whether the shape distribution has a tail worth warning about is itself a
    # corpus property: a formulaic corpus reaches skew 12, a broad one skew 3.
    reach = (f"And {off:.1f}% fall outside this frame altogether, reaching skew "
             f"{np.abs(sk).max():.0f}\nand excess kurtosis {ku.max():.0f} -- the "
             f"distribution of shapes is itself heavy-tailed."
             if off > 0.2 else
             f"The extremes are mild: nothing exceeds skew\n{np.abs(sk).max():.1f} "
             f"or excess kurtosis {ku.max():.0f}, and only {off:.2f}% of coordinates "
             f"leave this frame at all.")
    _caption(ax, f"The orange box is the sampling noise a genuinely Gaussian "
                 f"coordinate shows at N={S.n:,};\n{inside:.0f}% of live coordinates "
                 f"sit inside it. {reach}")

    ax = _ax(axes[1], "Normality rejection by sigma decile",
             "sigma decile (1 = narrowest)", "% rejecting normality (JB, p<0.001)")
    jb = jarque_bera(S.n, S.skew.ravel(), S.kurt.ravel())
    sd = S.std.ravel()
    dec = np.clip(np.digitize(sd, np.percentile(sd, np.arange(10, 100, 10))), 0, 9)
    rej = [float((jb[dec == d] > 13.82).mean() * 100) for d in range(10)]
    ax.bar(np.arange(1, 11), rej, color=SEQ_BLUE(np.linspace(0.35, 1.0, 10)))
    ax.set_ylim(0, max(rej) * 1.35)
    _caption(ax, "Most coordinates pass a normality test at every width. "
                 "Non-Gaussianity here is not a diffuse\nproperty of the space that "
                 "a global reshaping would fix -- it is concentrated in particular\n"
                 "coordinates, and figures 04 and 05 show which ones and why.")

    ax = _ax(axes[2], "Asymmetry: sigma+ / sigma- per coordinate",
             "ratio of the two half-spreads", "coordinates")
    ratio = S.sig_pos[live] / np.maximum(S.sig_neg[live], 1e-9)
    ax.hist(np.clip(ratio, 0.55, 1.8), bins=110, color=BLUE, alpha=0.9, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 0.8, 1.0, 1.25, 1.6])
    ax.set_xticklabels(["0.6", "0.8", "1.0", "1.25", "1.6"])
    ax.minorticks_off()
    ax.axvline(1.0, color=ORANGE, lw=1.8)
    skewed = float(((ratio > 1.25) | (ratio < 0.8)).mean() * 100)
    _caption(ax, f"Measuring the spread above and below the mean separately: only "
                 f"{skewed:.0f}% of live coordinates are\nmore than 25% lopsided. The "
                 f"original implementation's asymmetric percentile clamp was\n"
                 f"therefore correcting something real but small -- a refinement, not "
                 f"a different regime.")

    zexp = expected_max_abs_z(S.n)
    ax = _ax(axes[3], "Tail reach: largest |z| each coordinate attains",
             f"max |z| observed in {S.n:,} draws", "coordinates")
    zmax = np.maximum((S.q[100] - S.mean), -(S.q[0] - S.mean))[live] / \
        np.maximum(S.std[live], 1e-9)
    hi = max(12.0, float(np.ceil(np.percentile(zmax, 99.9))))
    ax.hist(np.clip(zmax, 2, hi), bins=100, color=BLUE, alpha=0.9, lw=0)
    ax.axvline(zexp, color=ORANGE, lw=1.8)
    ax.set_yscale("log")
    if zmax.max() > hi:      # only label a clip that actually happened
        ax.annotate(f"clipped at {hi:.0f}\nfor display", (hi, 200),
                    xytext=(2 + 0.7 * (hi - 2), 1500),
                    fontsize=7.5, color=MUTED,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    _caption(ax, f"Orange marks {zexp:.2f} -- the largest |z| {S.n:,} truly Gaussian "
                 f"draws would typically reach.\nThe bulk sits there, but "
                 f"{(zmax > 5).mean() * 100:.1f}% of coordinates go past 5 sigma and "
                 f"{(zmax > 8).mean() * 100:.2f}% past 8. Truncation\nat 2-3 sigma is "
                 f"therefore not a safety rail: it clips behaviour the real corpus "
                 f"exhibits.")

    _finish(fig, top)
    ctx["stats"].update(pct_live_coords_gaussian_like=inside,
                        pct_lopsided_coords=skewed,
                        pct_off_frame=off,
                        max_abs_skew=float(np.abs(sk).max()),
                        max_excess_kurtosis=float(ku.max()),
                        expected_max_abs_z=float(zexp),
                        pct_reject_normality_widest_decile=rej[-1],
                        pct_coords_beyond_5sigma=float((zmax > 5).mean() * 100),
                        pct_coords_beyond_8sigma=float((zmax > 8).mean() * 100))
    return fig


@figure("04", "What twenty real marginals actually look like",
        "Each panel is one (position, channel) coordinate: the corpus histogram, "
        "and the Gaussian the diagonal sampler draws in its place.")
def fig_marginal_gallery(S, ctx):
    live = np.flatnonzero(S.live_mask(1e-3))
    sk, ku = S.skew.ravel(), S.kurt.ravel()
    hole = GAUSSIAN_CENTRAL_MASS - ctx["central"].ravel()
    picks, labels = [], []

    def add(idx, label):
        if int(idx) not in picks:
            picks.append(int(idx)); labels.append(label)

    add(live[np.argmax(hole[live])], "biggest hole at the centre")
    add(live[np.argmax(ku[live])], "heaviest tails")
    add(live[np.argmax(sk[live])], "most right-skewed")
    add(live[np.argmin(sk[live])], "most left-skewed")
    order = live[np.argsort(np.abs(sk[live]) + np.abs(ku[live]))]
    for f in (0.02, 0.25, 0.5, 0.75, 0.9, 0.98):
        add(order[int(f * (order.size - 1))], f"shape percentile {f:.0%}")
    med = int(np.median(S.eos_pos))
    add(BOS * S.H + int(np.argmax(S.std[BOS])), "BOS (constant in every prompt)")
    add(med * S.H + int(np.argmax(S.std[med])), f"pos {med} (median EOS), widest")
    add((S.T - 1) * S.H + int(np.argmax(S.std[-1])), "pos 76 (always pad), widest")
    # The widest coordinates are dominated by a handful of channels; take the
    # widest coordinate per DISTINCT channel so the gallery shows variety
    # rather than the same channel at six neighbouring positions.
    seen_ch, rank = set(), 0
    for idx in np.argsort(S.std.ravel())[::-1]:
        ch = int(idx) % S.H
        if ch in seen_ch:
            continue
        seen_ch.add(ch)
        rank += 1
        add(idx, f"#{rank} widest channel")
        if len(picks) >= 20:
            break
    picks, labels = picks[:20], labels[:20]

    fig, axes = plt.subplots(4, 5, figsize=(15, 9.4))
    top = _header(fig, fig_marginal_gallery.meta[1], fig_marginal_gallery.meta[2])
    fig.text(0.012, top + 0.012, "blue = corpus histogram    orange = the fitted "
             "Gaussian    dashed teal = empirical q05 / q95 (the envelope the "
             "original implementation clamped to)", fontsize=8.5, color=INK2,
             va="bottom")
    flat = S.flat()
    for a, idx, lab in zip(axes.ravel(), picks, labels):
        t, c = divmod(idx, S.H)
        vals = flat[:, idx]
        mu, sdv = S.mean[t, c], max(S.std[t, c], 1e-9)
        if sdv <= 1e-8:
            # A constant coordinate has no histogram, and the "Gaussian" fitted
            # to it is a delta rendered as a bell only because the x-window
            # collapses to 1e-9 wide. Drawing either just gives the note
            # something to collide with -- say what the panel is instead.
            a.text(0.5, 0.5, f"all {S.n:,} prompts share\nthis exact value",
                   transform=a.transAxes, ha="center", va="center", fontsize=7,
                   color=VIOLET)
            a.set_xticks([])
        else:
            a.hist(vals, bins=60, color=BLUE, alpha=0.9, density=True, lw=0)
            lo = min(vals.min(), mu - 3.5 * sdv)
            hi = max(vals.max(), mu + 3.5 * sdv)
            xs = np.linspace(lo, hi, 300)
            a.plot(xs, np.exp(-0.5 * ((xs - mu) / sdv) ** 2)
                   / (sdv * np.sqrt(2 * np.pi)), color=ORANGE, lw=1.7)
            for p in (5, 95):
                a.axvline(S.q[p][t, c], color=AQUA, lw=1.0, ls="--")
            a.set_xlim(lo, hi)
        a.set_title(f"{lab}\npos {t}, ch {c}   skew {S.skew[t, c]:+.1f}   "
                    f"kurt {S.kurt[t, c]:+.1f}", fontsize=7, color=INK)
        a.set_yticks([]); a.tick_params(labelsize=6.5)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
    for a in axes.ravel()[len(picks):]:
        a.axis("off")
    fig.tight_layout(rect=[0, 0, 1, top - 0.012], h_pad=1.8)
    return fig


@figure("05", "The hidden binary: is this position past the EOS?",
        "CLIP pads with EOS, so at any middle position the corpus is a MIXTURE of "
        "two populations. That latent switch is what makes the wide coordinates "
        "bimodal -- and no fitted model here contains it.")
def fig_pad_gating(S, ctx):
    fig, axes, top = _fig(fig_pad_gating, 1, 4, figsize=(15, 5.4))
    eta = ctx["eta"]
    med = int(np.median(S.eos_pos))
    keep = 220

    ax = _ax(axes[0], "Variance explained by the past-EOS switch",
             f"the {keep} most gated channels", "token position", grid=None)
    order = np.argsort(eta.max(axis=0))[::-1][:keep]
    im = ax.imshow(eta[:, order], aspect="auto", origin="lower", cmap=SEQ_BLUE,
                   vmin=0, vmax=0.30, interpolation="nearest")
    fig.colorbar(im, ax=ax, pad=0.01, label="eta^2")
    _caption(ax, "Each column is a channel, each row a position; brightness is the "
                 "share of that coordinate's\nvariance explained by nothing more "
                 "than whether the prompt had already ended. The\nstructure is "
                 "channel-specific and position-banded -- a few dozen channels act "
                 "as an\n'am I still inside the prompt?' flag.")

    ax = _ax(axes[1], "The gate is strongest where prompts end",
             "token position", "eta^2")
    emean = eta.mean(axis=1)
    ax.plot(np.arange(S.T), emean, color=BLUE, label="mean over channels")
    ax.plot(np.arange(S.T), eta.max(axis=1), color=ORANGE, lw=1.5,
            label="most gated channel")
    ax.set_ylim(0, 1.05)
    ax.axvline(med, color=MUTED, lw=1, ls=":")
    ax.text(med + 1.5, 0.30, "median\nEOS", fontsize=8, color=MUTED)
    ax.legend(loc="upper right")
    # Where the mixture actually exists: the span over which the gate carries a
    # tenth of its peak strength. Both ends are corpus properties (the shortest
    # and longest prompts), so they are measured rather than quoted.
    on = np.flatnonzero(emean > 0.1 * emean.max())
    gate_lo, gate_hi = (int(on[0]), int(on[-1])) if on.size else (0, 0)
    _caption(ax, f"The effect switches on around position {gate_lo} and off again "
                 f"past {gate_hi} -- exactly the span of\nthe prompt-length "
                 f"distribution. Outside that window every prompt agrees (all "
                 f"content, or\nall padding) and the mixture vanishes. Inside it, the "
                 f"corpus is two populations wearing\none Gaussian.")

    t, c = np.unravel_index(int(np.argmax(eta)), eta.shape)
    ax = _ax(axes[2], f"The most gated coordinate (pos {t}, ch {c})", "value",
             "density")
    vals = S.X[:, t, c].astype(np.float64)
    m = S.eos_pos <= t
    ax.hist(vals[~m], bins=45, density=True, color=BLUE, alpha=0.75, lw=0,
            label=f"still content (n={(~m).sum()})")
    ax.hist(vals[m], bins=45, density=True, color=ORANGE, alpha=0.75, lw=0,
            label=f"already padding (n={m.sum()})")
    mu, sd = vals.mean(), vals.std()
    xs = np.linspace(vals.min() - 1.5, vals.max() + 1.5, 400)
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            color=INK, lw=2.0, label="the fitted single Gaussian")
    ax.legend(loc="upper left", fontsize=7.5)
    inside = float(((vals > mu - sd) & (vals < mu + sd)).mean())
    _caption(ax, f"The fitted Gaussian places its peak precisely in the gap between "
                 f"the two lobes. Only {inside:.0%} of\nreal prompts fall within one "
                 f"sigma of the fitted mean, where a Gaussian would put 68%.\nEvery "
                 f"draw the diagonal sampler makes near this mean is a value no real "
                 f"prompt ever\nproduced at this coordinate.")

    ax = _ax(axes[3], "How many coordinates are gated",
             "eta^2 (variance explained by the switch)", "coordinates")
    ax.hist(eta[eta > 0].ravel(), bins=80, color=BLUE, alpha=0.9, lw=0)
    ax.set_yscale("log")
    for thr, col in ((0.25, ORANGE), (0.5, RED)):
        ax.axvline(thr, color=col, lw=1.4, ls="--")
    n25 = int((eta > 0.25).sum()); n50 = int((eta > 0.5).sum())
    ax.text(0.26, 3000, ">25%", fontsize=8, color=ORANGE)
    ax.text(0.51, 3000, ">50%", fontsize=8, color=RED)
    _caption(ax, f"{n25:,} coordinates take more than a quarter of their variance "
                 f"from this one binary, {n50:,} take\nmore than half. It is the "
                 f"clearest actionable finding here: draw a prompt length first,\n"
                 f"then sample conditionally on it, and the bimodality disappears "
                 f"without any new machinery --\nsimply because the mixture was "
                 f"never modelled in the first place.")

    _finish(fig, top)
    ctx["stats"].update(gated_coords_eta_gt_25=n25, gated_coords_eta_gt_50=n50,
                        max_eta_coord=[int(t), int(c)], max_eta=float(eta.max()),
                        max_eta_lobes=[float(vals[m].mean()),
                                       float(vals[~m].mean())],
                        max_eta_central_mass=inside,
                        gate_window=[gate_lo, gate_hi])
    return fig


@figure("06", "How many principal axes are real",
        "The PCA spectrum against a null built by shuffling every coordinate "
        "independently: same {D} marginals, all correlation destroyed.")
def fig_spectrum(S, ctx):
    fig, axes, top = _fig(fig_spectrum, 1, 3, figsize=(13.5, 5.6))
    v, nv = S.pca_sval ** 2, S.null_sval ** 2
    k = np.arange(1, v.size + 1)
    cross = int(np.searchsorted(-(v - nv), 0))

    ax = _ax(axes[0], "Eigenvalue spectrum vs the shuffled null", "component",
             "variance", grid="both")
    ax.plot(k, v, color=BLUE, label="corpus")
    ax.plot(k, nv, color=ORANGE, lw=1.8, label="independent-coordinate null")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvspan(cross, v.size, color=ORANGE, alpha=0.13, lw=0)
    ax.axvline(cross, color=RED, lw=1.4, ls="--")
    ax.text(cross * 1.12, v.max() * 0.02, f"noise floor\nat component {cross}",
            fontsize=8, color=RED)
    ax.legend(loc="lower left")
    _caption(ax, f"The null keeps every marginal and destroys every correlation, so "
                 f"where the two curves meet is\nwhere real structure ends. Only "
                 f"~{cross} of the {v.size:,} fitted components clear it. Everything "
                 f"in the\n"
                 f"shaded region is sampling noise dressed as a principal axis -- so "
                 f"--comp-lo above {cross}, the\n'ride the weird minor axes' knob, is "
                 f"riding estimation error rather than rare structure.")

    ax = _ax(axes[1], "Cumulative variance explained", "components kept",
             "% of total variance", grid="both")
    cum = np.cumsum(v) / v.sum() * 100
    ax.plot(k, cum, color=BLUE)
    ax.set_xscale("log")
    for frac, col in ((50, AQUA), (90, ORANGE), (99, RED)):
        i = int(np.searchsorted(cum, frac)) + 1
        ax.plot([i], [frac], "o", ms=7, color=col, zorder=5)
        ax.annotate(f"{frac}% @ {i} components", (i, frac), xytext=(9, -13),
                    textcoords="offset points", fontsize=8, color=col)
    ax.axvline(cross, color=RED, lw=1.0, ls="--")
    ax.text(cross * 1.1, 8, f"noise floor\nat {cross}", fontsize=8, color=RED)
    ctx["stats"]["pca_var_frac_above_null"] = float(cum[cross - 1])
    _caption(ax, f"The variance is spread wide: half of it needs "
                 f"{int(np.searchsorted(cum, 50)) + 1} components. But note where "
                 f"the noise\nfloor lands on this curve -- the ~{cross} genuine axes "
                 f"account for about {cum[cross - 1]:.0f}% of the total.\nThe "
                 f"remaining {100 - cum[cross - 1]:.0f}% is real variance with no "
                 f"reliable direction attached to it.")

    ax = _ax(axes[2], "Effective dimensionality", "", "participation ratio")
    pr_c, pr_n = effective_dimension(v), effective_dimension(nv)
    bars = {"corpus\nPCA spectrum": pr_c, "shuffled\nnull": pr_n}
    ax.bar(list(bars), list(bars.values()), color=[BLUE, ORANGE], width=0.45)
    for i, val in enumerate(bars.values()):
        ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=12,
                color=INK)
    ax.set_ylim(0, max(bars.values()) * 1.35)
    _caption(ax, f"{S.n:,} prompts buy roughly {pr_c:.0f} usable directions in a "
                 f"{S.D:,}-dimensional space. Had the\ncoordinates been independent, "
                 f"the same data would have shown ~{pr_n:.0f}. That gap is the whole\n"
                 f"reason the pca sampler works: the corpus is a thin, structured "
                 f"sheet, and the number of\ndegrees of freedom a sampler should be "
                 f"exercising is closer to {pr_c:.0f} than to {S.D:,}.")

    _finish(fig, top)
    ctx["stats"].update(pca_components_above_null=cross,
                        pca_components_fitted=int(v.size),
                        pca_participation_ratio=pr_c,
                        null_participation_ratio=pr_n,
                        pca_components_for_50pct=int(np.searchsorted(cum, 50)) + 1)
    return fig


@figure("07", "The corpus in its own basis",
        "The pca sampler draws N(0,1) coefficients per axis. How well the corpus "
        "scores justify that is a property of the corpus, not of the method.")
def fig_pca_scores(S, ctx):
    sc = S.pca_scores
    kk = sc.shape[1]
    z = sc / np.maximum(sc.std(axis=0, keepdims=True), 1e-12)
    sk = (z ** 3).mean(axis=0)
    ku = (z ** 4).mean(axis=0) - 3

    fig = plt.figure(figsize=(14.5, 8.6))
    top = _header(fig, fig_pca_scores.meta[1], fig_pca_scores.meta[2])
    gs = fig.add_gridspec(1, 3, top=top - 0.02, bottom=0.56, wspace=0.24,
                          left=0.055, right=0.985)

    ax = _ax(fig.add_subplot(gs[0, 0]), "Score non-Gaussianity by component",
             "component index", "value", grid="both")
    ax.plot(np.arange(1, kk + 1), np.abs(sk), color=BLUE, lw=1.1, label="|skew|")
    ax.plot(np.arange(1, kk + 1), ku, color=ORANGE, lw=1.1,
            label="excess kurtosis")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhspan(-2 * np.sqrt(24 / S.n), 2 * np.sqrt(24 / S.n), color=GRID, zorder=0)
    ax.set_xscale("log"); ax.set_ylim(-1.2, 2.2)
    ax.legend(loc="upper left", ncol=2)
    # The noise floor, recomputed here so --only 07 stands on its own.
    cross = int(np.searchsorted(-(S.pca_sval ** 2 - S.null_sval ** 2), 0))
    band = 2 * np.sqrt(24 / S.n)
    lead10 = min(10, kk)
    inband = int((np.abs(ku[:lead10]) < band).sum())
    _caption(ax, f"The grey band is the sampling noise a Gaussian shows at "
                 f"N={S.n:,}. {inband} of the first {lead10} axes sit\ninside it "
                 f"(PC1 kurtosis {ku[0]:+.2f}, PC2 {ku[1]:+.2f}). The drift upward "
                 f"past component ~{cross} is the noise\nfloor from figure 06 "
                 f"asserting itself, not structure appearing.")

    ax = _ax(fig.add_subplot(gs[0, 1]), "Corpus radius in whitened PCA space",
             "chi-distance over the top 200 components", "prompts")
    top200 = min(200, kk)
    r = np.sqrt((z[:, :top200] ** 2).sum(axis=1))
    ax.hist(r, bins=45, color=BLUE, alpha=0.9, lw=0)
    ax.axvline(np.sqrt(top200), color=ORANGE, lw=2.0)
    _caption(ax, f"Orange marks sqrt({top200}) = {np.sqrt(top200):.1f}, where a "
                 f"200-dimensional Gaussian would concentrate.\nThe corpus mean is "
                 f"{r.mean():.1f}. The manifold is not merely low-rank -- inside its "
                 f"own subspace it is\nGaussian-shaped, which is exactly the "
                 f"assumption the pca sampler makes.")

    ax = _ax(fig.add_subplot(gs[0, 2]), "Where non-Gaussianity actually lives", "",
             "mean |excess kurtosis|")
    live = S.live_mask(1e-3)
    lead = max(1, min(cross, kk))
    bars = {"raw\ncoordinates": float(np.abs(S.kurt[live]).mean()),
            f"PCA scores\n(top {lead})": float(np.abs(ku[:lead]).mean()),
            f"PCA scores\n(all {kk:,})": float(np.abs(ku).mean())}
    ax.bar(list(bars), list(bars.values()), color=[ORANGE, BLUE, MUTED], width=0.5)
    for i, val in enumerate(bars.values()):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=11,
                color=INK)
    ax.set_ylim(0, max(bars.values()) * 1.3)
    raw_k, lead_k = bars["raw\ncoordinates"], bars[f"PCA scores\n(top {lead})"]
    # Whether the rotation helps depends on how non-Gaussian the raw marginals
    # were to begin with. On a corpus whose raw coordinates are already close to
    # Gaussian there is nothing left for a change of basis to absorb.
    verdict = (
        "Rotating into the corpus's own basis removes most of the shape problem: "
        "the bimodality of\nfigure 05 is a coordinate-basis artefact, a linear "
        "mixture effect that a linear change of\nbasis largely absorbs."
        if lead_k < 0.8 * raw_k else
        f"Here the rotation does NOT help -- the leading scores ({lead_k:.2f}) are "
        f"further from Gaussian\nthan the raw coordinates ({raw_k:.2f}) already are. "
        f"This corpus's marginals are close to\nGaussian to begin with, so there is "
        f"no coordinate-basis artefact left for a change of\nbasis to absorb."
    )
    _caption(ax, verdict)

    gs2 = fig.add_gridspec(1, 8, top=0.285, bottom=0.07, left=0.045, right=0.985,
                           wspace=0.22)
    for i, comp in enumerate([0, 1, 2, 9, 49, 199, 499, 900]):
        a = fig.add_subplot(gs2[0, i])
        if comp >= kk:
            a.axis("off"); continue
        a.hist(z[:, comp], bins=40, density=True, color=BLUE, alpha=0.9, lw=0)
        xs = np.linspace(-4, 4, 200)
        a.plot(xs, np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi), color=ORANGE, lw=1.5)
        a.set_title(f"PC{comp + 1}\nskew {sk[comp]:+.2f}  kurt {ku[comp]:+.2f}",
                    fontsize=7.5, color=INK)
        a.set_xlim(-4.2, 4.2); a.set_yticks([]); a.tick_params(labelsize=6.5)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
    fig.text(0.045, 0.375, "Standardised score histograms, leading to trailing axes",
             fontsize=10.5, color=INK, fontweight="bold")
    agree = ("They agree from the first axis to the last."
             if np.abs(ku[:lead10]).max() < 4 * band else
             "The leading axes do NOT agree -- PC1 is two lobes, which is the "
             "prompt-length split of figure 08 seen edge-on. From roughly PC3 "
             "onward they do.")
    fig.text(0.045, 0.352, "Blue = corpus scores, orange = the unit Gaussian the "
             f"pca sampler substitutes. {agree}", fontsize=8.5, color=INK2)
    _finish(fig, top, layout=False)
    ctx["stats"].update(pc1_kurtosis=float(ku[0]),
                        mean_abs_kurt_raw=float(np.abs(S.kurt[live]).mean()),
                        mean_abs_kurt_pca_lead=float(np.abs(ku[:lead]).mean()),
                        pca_lead_components=int(lead),
                        whitened_radius_dims=int(top200),
                        whitened_radius_mean=float(r.mean()))
    return fig


@figure("08", "Global geometry: how many blobs, and what the first axes mean",
        "Whether one Gaussian is the right global model is something the corpus "
        "answers -- and the leading directions encode something mundane, which is "
        "worth knowing before riding them.")
def fig_geometry(S, ctx):
    fig, axes, top = _fig(fig_geometry, 1, 4, figsize=(15, 5.4))
    sc, eos = S.pca_scores, S.eos_pos

    ax = _ax(axes[0], "PC1 vs PC2, coloured by prompt length", "PC1", "PC2",
             grid="both")
    sp = ax.scatter(sc[:, 0], sc[:, 1], c=eos, cmap=SEQ_BLUE, s=12, alpha=0.85,
                    lw=0)
    fig.colorbar(sp, ax=ax, pad=0.01, label="EOS position")
    # A corpus that mixes short prompts with ones long enough to fill the window
    # shows the split here as a detached lobe, not just a gradient.
    full = float((eos >= S.T - 2).mean())
    lobe = (f" The detached lobe is the {full:.0%} of prompts long enough to fill "
            f"the whole\n77-token window -- a visibly separate population, not a "
            f"tail." if full > 0.02 else "")
    _caption(ax, f"The colour gradient runs diagonally across the plane: the two "
                 f"dominant directions of the\nentire corpus are largely a "
                 f"reparametrisation of how many words the prompt had.{lobe}")

    ax = _ax(axes[1], "PC3 vs PC4", "PC3", "PC4", grid="both")
    ax.scatter(sc[:, 2], sc[:, 3], c=eos, cmap=SEQ_BLUE, s=12, alpha=0.85, lw=0)
    _caption(ax, "Deeper in, the length signal is gone and what remains is an "
                 "ordinary elliptical cloud: no\nclusters, no arms, no holes. This "
                 "is what a single Gaussian is supposed to look like.")

    ax = _ax(axes[2], "Correlation of each axis with prompt length",
             "component", "|correlation| with EOS position")
    kk = min(40, sc.shape[1])
    r = np.array([abs(np.corrcoef(sc[:, i], eos)[0, 1]) for i in range(kk)])
    ax.bar(np.arange(1, kk + 1), r,
           color=[ORANGE if x > 0.3 else BLUE for x in r])
    ax.axhline(2 / np.sqrt(S.n), color=MUTED, lw=1, ls=":")
    ax.set_ylim(0, max(0.8, r.max() * 1.15))
    strong = np.flatnonzero(r > 0.3)
    lastc = int(strong[-1]) + 1 if strong.size else 0
    _caption(ax, f"PC1 and PC2 carry |r| = {r[0]:.2f} and {r[1]:.2f} with prompt "
                 f"length; past component ~{lastc} nothing\ndoes. The 'dominant, "
                 f"tasteful' axes the pca sampler starts from are partly a length "
                 f"knob in\ndisguise -- which is worth knowing before treating low "
                 f"--comp-lo as a semantic control.")

    ax = _ax(axes[3], "Mixture evidence (diagonal GMM on the top 20 PCs)",
             "number of components k", "BIC (lower = better)")
    bic = ctx.get("bic")
    if bic:
        ks = [b[0] for b in bic]; vs = [b[1] for b in bic]
        ax.plot(ks, vs, color=BLUE, marker="o", ms=5)
        best = ks[int(np.argmin(vs))]
        ax.plot([best], [min(vs)], "o", ms=12, mfc="none", mec=ORANGE, mew=2.2)
        edge = " (the largest k tested -- the true optimum is higher)" \
            if best == ks[-1] else ""
        _caption(ax, (
            f"BIC picks k = {best}: adding components only costs. The original "
            f"design called for a 'mixture\nof Gaussians'; globally, this corpus "
            f"does not need one. The mixture that IS present is\nlocal and "
            f"per-position, and figure 05 names it."
            if best == 1 else
            f"BIC picks k = {best}{edge}, not 1: this corpus IS globally "
            f"multimodal. The original\ndesign's call for a 'mixture of Gaussians' "
            f"was right for a corpus like this one -- a single\nGaussian is "
            f"averaging over genuinely distinct populations of prompt, on top of "
            f"the local\nper-position mixture figure 05 names."))
        ctx["stats"]["gmm_best_k"] = best
        ctx["stats"]["gmm_k_at_range_edge"] = bool(best == ks[-1])
    else:
        ax.text(0.5, 0.5, "sklearn not installed", ha="center", color=MUTED)

    _finish(fig, top)
    ctx["stats"].update(pc1_corr_prompt_length=float(r[0]),
                        pc2_corr_prompt_length=float(r[1]),
                        last_length_correlated_pc=lastc)
    return fig


@figure("09", "Correlation lives ACROSS token positions, not within them",
        "Coordinate independence fails -- but not uniformly. Channels inside one "
        "position are nearly independent; whole rows are not.")
def fig_correlation(S, ctx):
    fig, axes, top = _fig(fig_correlation, 1, 4, figsize=(15, 5.6))
    med = int(np.median(S.eos_pos))
    dev = (S.X - S.mean[None]).astype(np.float64)
    rng = np.random.default_rng(0)

    ax = _ax(axes[0], f"Channel x channel within position {med}", "channel",
             "channel", grid=None)
    d = dev[:, med, :]
    d /= np.maximum(d.std(axis=0, keepdims=True), 1e-9)
    c = (d.T @ d) / S.n
    im = ax.imshow(c, cmap=DIVERGING, vmin=-0.5, vmax=0.5, aspect="auto",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, pad=0.01, label="correlation")
    _caption(ax, f"Almost blank. Inside a single token row the {S.H} channels are "
                 f"close to independent -- the one\nassumption the diagonal sampler "
                 f"makes that the corpus actually supports.")

    ax = _ax(axes[1], "Position x position coupling", "token position",
             "token position", grid=None)
    f = dev / np.maximum(np.linalg.norm(dev, axis=(0, 2), keepdims=True), 1e-12)
    tc = np.einsum("ntc,nsc->ts", f, f)
    im = ax.imshow(tc, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, pad=0.01, label="cosine of the deviation blocks")
    _caption(ax, "The same measurement across rows is nearly saturated. The tail is "
                 "one rigid block: every\npadding row moves with every other, "
                 "because each is a running summary of the same prompt.")

    ax = _ax(axes[2], "|correlation| by pair type", "|correlation|", "density")
    idx = rng.choice(S.D, 4000, replace=False)
    xs = dev.reshape(S.n, -1)[:, idx]
    xs /= np.maximum(xs.std(axis=0, keepdims=True), 1e-9)
    cm = (xs.T @ xs) / S.n
    ti, ci = idx // S.H, idx % S.H
    iu = np.triu_indices_from(cm, 1)
    same_pos = ti[iu[0]] == ti[iu[1]]
    same_ch = ci[iu[0]] == ci[iu[1]]
    vals = np.abs(cm[iu])
    groups = {
        "same position, different channel": (vals[same_pos], BLUE),
        "same channel, different position": (vals[same_ch & ~same_pos], ORANGE),
        "different position and channel": (vals[~same_pos & ~same_ch], MUTED),
    }
    med_r = {}
    for lab, (v, col) in groups.items():
        if v.size < 20:
            continue
        med_r[lab] = float(np.median(v))
        ax.hist(v, bins=60, density=True, histtype="step", lw=2.2, color=col,
                label=f"{lab}\nmedian {med_r[lab]:.3f}")
    ax.hist(np.abs(rng.standard_normal(40000) / np.sqrt(S.n)), bins=60,
            density=True, histtype="step", lw=1.5, color=AQUA, ls="--",
            label=f"independent null\nmedian {0.674 / np.sqrt(S.n):.3f}")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="upper right", fontsize=7, labelspacing=0.9)
    within = med_r.get("same position, different channel", 0.0)
    across = med_r.get("same channel, different position", 0.0)
    _caption(ax, f"This is the decomposition that matters. Two channels inside one "
                 f"row correlate at {within:.2f} --\nbarely above the "
                 f"{0.674 / np.sqrt(S.n):.2f} an independent corpus would show. The "
                 f"SAME channel at two\ndifferent positions correlates at "
                 f"{across:.2f}. Coordinate independence is not uniformly wrong;\n"
                 f"it is wrong along exactly one axis, the sequence axis.")

    ax = _ax(axes[3], "Effective dimensions per block of positions",
             "", "participation ratio")
    blocks = ctx["blocks"]
    names = list(blocks)
    raw = [b[0] for b in blocks.values()]
    eff = [b[1] for b in blocks.values()]
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, raw, width=0.38, color=MUTED, label="coordinates in block")
    ax.bar(xpos + 0.2, eff, width=0.38, color=BLUE, label="effective directions")
    ax.set_yscale("log"); ax.set_ylim(30, 1e5)
    ax.set_xticks(xpos); ax.set_xticklabels(names, fontsize=8)
    ax.legend(loc="upper right")
    waste = raw[-1] / max(eff[-1], 1e-9)
    _caption(ax, f"The mostly-padding tail spends {raw[-1]:,} coordinates on about "
                 f"{eff[-1]:.0f} real directions. A diagonal\ndraw gives that block "
                 f"{raw[-1]:,} independent ones -- {waste:.0f}x more entropy than the "
                 f"corpus puts there,\npoured into a subspace the corpus treats as "
                 f"almost rigid. That is the single largest\nstructural mismatch in "
                 f"the current design.")

    _finish(fig, top)
    ctx["stats"]["median_abs_corr_by_pair_type"] = med_r
    ctx["stats"]["corr_independent_null"] = float(0.674 / np.sqrt(S.n))
    ctx["stats"]["pad_block_coords"] = int(raw[-1])
    ctx["stats"]["pad_block_effective_dims"] = float(eff[-1])
    return fig


@figure("10", "The shell the real prompts live on",
        "In {D} dimensions a Gaussian is a razor-thin shell. The corpus is a "
        "thicker one, and the samplers land on neither by accident.")
def fig_radius(S, ctx):
    fig, axes, top = _fig(fig_radius, 1, 3, figsize=(13.5, 5.6))
    sets, cols = ctx["samples"], ctx["sample_colors"]

    ax = _ax(axes[0], "Deviation norm ||c - mu||", "L2 norm", "density")
    shown = []
    for k in ("corpus", "diagonal T=1", "pca T=1", "hybrid"):
        v = sets[k].reshape(len(sets[k]), -1) - S.mean.ravel()[None]
        norms = np.linalg.norm(v, axis=1)
        shown.append(norms)
        ax.hist(norms, bins=45, density=True, histtype="step",
                lw=2.2, color=cols[k], label=k)
    # The corpus is a broad band and the parametric samplers are near-deltas
    # inside it; window on the union so no curve is cropped out of the panel.
    allv = np.concatenate(shown)
    pad = 0.06 * (allv.max() - allv.min())
    ax.set_xlim(allv.min() - pad, allv.max() + pad)
    ax.legend(loc="upper left")      # the diagonal spike owns the upper right
    dn = np.linalg.norm(S.centered(), axis=1)
    dd = np.linalg.norm(sets["diagonal T=1"].reshape(len(sets["diagonal T=1"]), -1)
                        - S.mean.ravel()[None], axis=1)
    _caption(ax, f"Real prompts spread over a band of radii (relative sd "
                 f"{dn.std() / dn.mean():.3f}); the diagonal sampler is a\nspike "
                 f"({dd.std() / dd.mean():.3f}), "
                 f"{dn.std() / dn.mean() / (dd.std() / dd.mean()):.0f}x tighter, "
                 f"because averaging {S.D:,} independent squares concentrates\nhard. "
                 f"Every diagonal draw therefore sits at the same distance from the "
                 f"centre, and temperature\nis the only radial control available -- "
                 f"a global one, applied to all draws at once.")

    ax = _ax(axes[1], "The repo's whitened distance gauge", "RMS z-score",
             "prompts")
    sfl = np.maximum(S.std, 0.1 * S.std.mean())
    d = np.sqrt((((S.X - S.mean[None]) / sfl[None]) ** 2).reshape(S.n, -1).mean(1))
    ax.hist(d, bins=50, color=BLUE, alpha=0.9, lw=0)
    ax.axvline(d.mean(), color=ORANGE, lw=1.8)
    _caption(ax, f"The gauge the dashboard reports is honestly calibrated: mean "
                 f"{d.mean():.2f}, sd {d.std():.2f} on the corpus\nit was fitted to. "
                 f"But real prompts only ever span {d.min():.2f} to {d.max():.2f}. "
                 f"Everything the taste loop calls\n'the periphery' (d > 1.3) is "
                 f"territory no real prompt occupies -- which is the point, but\n"
                 f"worth stating: it is extrapolation, not a sparsely-visited corner "
                 f"of the corpus.")

    ax = _ax(axes[2], "Per-position deviation norm", "token position",
             "||deviation row||")
    for k, col in (("corpus", INK), ("diagonal T=1", ORANGE), ("pca T=1", BLUE)):
        pn = np.linalg.norm(sets[k].reshape(-1, S.T, S.H) - S.mean[None], axis=2)
        ax.plot(np.arange(S.T), np.median(pn, axis=0), color=col,
                lw=2.4 if k == "corpus" else 1.6, label=k)
        if k == "corpus":
            hi95 = np.percentile(pn, 95, axis=0)
            ax.fill_between(np.arange(S.T), np.percentile(pn, 5, axis=0), hi95,
                            color=INK, alpha=0.12, lw=0)
    ax.set_ylim(0, hi95.max() * 1.2)
    ax.legend(loc="lower right")
    _caption(ax, "Both samplers reproduce the corpus's per-position energy budget "
                 "almost exactly. Taken with the\nnear-unit readable-variance ratio "
                 "in figure 11, this settles what is and is not broken: the\n"
                 "magnitudes are right everywhere. What the diagonal sampler gets "
                 "wrong is direction and\nagreement between rows, not scale.")

    _finish(fig, top)
    ctx["stats"].update(dev_norm_rel_sd=float(dn.std() / dn.mean()),
                        diagonal_norm_rel_sd=float(dd.std() / dd.mean()),
                        corpus_distance_mean=float(d.mean()),
                        corpus_distance_sd=float(d.std()),
                        corpus_distance_min=float(d.min()),
                        corpus_distance_max=float(d.max()))
    return fig


@figure("11", "What the UNet can actually read",
        "Conditioning enters SD only through the cross-attention to_k / to_v "
        "projections. If some channels were far more 'sensitive', it would show "
        "up here.")
def fig_sensitivity(S, ctx):
    sens, wr = ctx.get("sens"), ctx.get("readable")
    if sens is None:
        return None
    fig, axes, top = _fig(fig_sensitivity, 1, 4, figsize=(15, 5.6))
    sk, sv = sens["to_k"], sens["to_v"]
    chan_sigma = S.std[1:].mean(axis=0)

    ax = _ax(axes[0], "Per-channel sensitivity (16 blocks)",
             "channel rank (by to_k)", "RMS column norm")
    o = np.argsort(sk)[::-1]
    ax.plot(np.arange(S.H), sk[o], color=BLUE, label="to_k")
    ax.plot(np.arange(S.H), sv[o], color=ORANGE, lw=1.2, alpha=0.85, label="to_v")
    ax.set_ylim(0, max(sk.max(), sv.max()) * 1.15)
    ax.legend(loc="center right")
    spread = np.percentile(sk, 99) / np.percentile(sk, 1)
    _caption(ax, f"A channel's column in to_k / to_v is its ONLY path into the "
                 f"image, so that column's norm is a\ndirect, gradient-free sensitivity. "
                 f"Between the 1st and 99th percentile the whole range spans\n"
                 f"{spread:.1f}x. The hypothesis that some conditioning channels are "
                 f"far more sensitive than others\ndoes not survive this measurement "
                 f"-- the UNet reads all {S.H} about equally.")

    ax = _ax(axes[1], "Corpus spread vs sensitivity", "channel sigma",
             "to_k sensitivity", grid="both")
    ax.scatter(chan_sigma, sk, s=12, color=BLUE, alpha=0.55, lw=0)
    r = float(np.corrcoef(chan_sigma, sk)[0, 1])
    _caption(ax, f"Correlation r = {r:+.2f}: a cloud, not a trend. What the corpus "
                 f"varies and what the model listens\nto are unrelated. Two "
                 f"consequences -- sigma is a poor proxy for a coordinate's "
                 f"importance,\nand there is no cheap 'amplify the channels that "
                 f"matter' move hiding here.")

    ax = _ax(axes[2], "Effective drive: sigma x sensitivity", "channel rank",
             "cumulative share (%)", grid="both")
    drive = np.sort(chan_sigma * sk)[::-1]
    ax.plot(np.arange(1, S.H + 1), np.cumsum(drive) / drive.sum() * 100,
            color=BLUE, label="sigma x sensitivity")
    ax.plot([0, S.H], [0, 100], color=MUTED, lw=1.2, ls=":", label="perfectly flat")
    ax.legend(loc="lower right")
    n80 = int(np.searchsorted(np.cumsum(drive) / drive.sum(), 0.8) + 1)
    _caption(ax, f"Combining the two: {n80} of {S.H} channels carry 80% of the "
                 f"effective drive, where a perfectly\nflat budget would need "
                 f"{int(0.8 * S.H)}. The curve barely leaves the diagonal. Selecting "
                 f"or freezing channels\nis not where a better sampler comes from.")

    ax = _ax(axes[3], "Readable variance: corpus / diagonal",
             "token position", "true / diagonal (one dot per matrix)")
    if wr:
        for i, (t, ratios) in enumerate(sorted(wr.items())):
            vals = np.array(list(ratios.values()))
            ax.scatter(np.full(vals.size, i) + np.linspace(-0.22, 0.22, vals.size),
                       vals, s=16, color=BLUE, alpha=0.7, lw=0)
            ax.plot([i - 0.3, i + 0.3], [vals.mean()] * 2, color=ORANGE, lw=2.4)
        ax.set_xticks(range(len(wr)))
        ax.set_xticklabels([f"pos {t}" for t in sorted(wr)])
        ax.axhline(1.0, color=MUTED, lw=1.4, ls="--")
        ax.set_ylim(0.5, 2.3)
        allr = [v for r_ in wr.values() for v in r_.values()]
        nmat = max(len(r_) for r_ in wr.values())
        _caption(ax, f"Projecting the true channel covariance through each of the "
                     f"{nmat} matrices and comparing with what\nthe diagonal model "
                     f"claims: "
                     f"the ratio stays inside {min(allr):.2f}-{max(allr):.2f} "
                     f"(orange = per-position mean). The\ndiagonal model gets the "
                     f"MAGNITUDE of readable variance right to about 20%. Its failure "
                     f"is not\nscale -- it is that the 77 rows it hands the model do "
                     f"not agree with one another (figure 09).")
        ctx["stats"]["readable_variance_ratio_range"] = [float(min(allr)),
                                                         float(max(allr))]
    _finish(fig, top)
    ctx["stats"].update(channels_for_80pct_drive=n80,
                        channels_for_80pct_flat=int(0.8 * S.H),
                        sensitivity_spread=float(spread),
                        n_cross_attn_matrices=int(sens["per_layer_k"].shape[0]
                                                  + sens["per_layer_v"].shape[0]),
                        sigma_sensitivity_corr=r)
    return fig


@figure("12", "Sampler autopsy",
        "512 draws from every shipped sampler, scored against the corpus on the "
        "statistics above. This is the scoreboard the next sampler has to beat.")
def fig_sampler_autopsy(S, ctx):
    sets, cols = ctx.get("samples"), ctx["sample_colors"]
    if sets is None:
        return None
    fig, axes, top = _fig(fig_sampler_autopsy, 1, 4, figsize=(15.5, 5.8))

    ax = _ax(axes[0], "Nearest-neighbour cosine to a real prompt",
             f"max cosine over the {S.n:,} corpus rows", "density")
    ref = S.flat().astype(np.float64)
    ref /= np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
    nn, peaks = {}, {}
    for k, v in sets.items():
        f = v.reshape(len(v), -1).astype(np.float64)
        f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        s = f @ ref.T
        if k == "corpus":
            s[np.arange(len(v)), np.arange(len(v))] = -1
        nn[k] = s.max(axis=1)
        peaks[k] = float(np.histogram(nn[k], bins=45, density=True)[0].max())
    # The parametric samplers are near-deltas; let the tallest run off the top
    # rather than flattening every other distribution onto the baseline. The
    # clip is announced in the legend, which is the one place in this panel
    # guaranteed not to have a curve running through it.
    tall = max(peaks, key=peaks.get)
    for k in sets:
        lab = f"{k}  (peak {peaks[k]:.0f}, clipped)" if k == tall else k
        ax.hist(nn[k], bins=45, density=True, histtype="step", lw=2.2,
                color=cols[k], label=lab)
    ax.set_ylim(0, sorted(peaks.values())[-2] * 1.35)
    ax.legend(loc="upper right", fontsize=7.5)
    _caption(ax, f"How close does a sample get to the nearest thing a human actually "
                 f"wrote? Real prompts sit at\n{np.median(nn['corpus']):.2f} from "
                 f"each other. diagonal draws sit at "
                 f"{np.median(nn['diagonal T=1']):.2f} -- further from every real "
                 f"prompt than any\nreal prompt is from any other. hybrid at "
                 f"{np.median(nn['hybrid']):.2f} is the opposite failure: near-copies. "
                 f"pca at\n{np.median(nn['pca T=1']):.2f} is the only sampler in "
                 f"between, and even it does not reach the corpus.")

    ax = _ax(axes[1], "Whitened distance gauge", "RMS z-score", "density")
    sfl = np.maximum(S.std, 0.1 * S.std.mean())
    gpeaks, gmax = {}, 0.0
    for k, v in sets.items():
        d = np.sqrt((((v.reshape(-1, S.T, S.H) - S.mean[None]) / sfl[None]) ** 2)
                    .reshape(len(v), -1).mean(axis=1))
        counts, _, _ = ax.hist(d, bins=45, density=True, histtype="step", lw=2.2,
                               color=cols[k])
        gpeaks[k] = float(counts.max())
        gmax = max(gmax, float(d.max()))
    gtall = max(gpeaks, key=gpeaks.get)
    ax.set_xlim(0.5, gmax * 1.05)
    ax.set_ylim(0, sorted(gpeaks.values())[-2] * 1.35)
    ax.text(0.02, 0.97, f"{gtall} spike clipped", transform=ax.transAxes,
            fontsize=7, color=cols[gtall], va="top")
    _caption(ax, "Temperature moves the gauge cleanly and predictably -- that part "
                 "of the design works. But every\nsampler is a spike where the corpus "
                 "is a band (figure 10, middle). No shipped sampler\nreproduces the "
                 "corpus's own radial spread, so 'distance' selects a shell rather "
                 "than a\npopulation.")

    ax = _ax(axes[2], "Marginal shape reproduced", "", "mean |excess kurtosis|")
    live = S.live_mask(1e-3).ravel()
    vals = {}
    for k, v in sets.items():
        f = v.reshape(len(v), -1)[:, live]
        z = (f - f.mean(0)) / np.maximum(f.std(0), 1e-9)
        vals[k] = float(np.abs((z ** 4).mean(0) - 3).mean())
    ax.barh(list(vals)[::-1], list(vals.values())[::-1],
            color=[cols[k] for k in list(vals)[::-1]], height=0.6)
    ax.tick_params(labelsize=8)
    for i, k in enumerate(list(vals)[::-1]):
        ax.text(vals[k] + 0.015, i, f"{vals[k]:.2f}", va="center", fontsize=8,
                color=INK)
    ax.set_xlim(0, max(vals.values()) * 1.25)
    _caption(ax, "Only the corpus and hybrid carry non-Gaussian marginals; the three "
                 "parametric samplers flatten\nthem to a common floor. Whether that "
                 "matters is an open question this analysis cannot\nsettle -- but it "
                 "is the one property no current sampler even tries to reproduce.")

    ax = _ax(axes[3], "Position-to-position coherence", "",
             "mean |cosine| between position blocks")
    coh = {}
    for k, v in sets.items():
        d = v.reshape(-1, S.T, S.H).astype(np.float64) - S.mean[None]
        d /= np.maximum(np.linalg.norm(d, axis=(0, 2), keepdims=True), 1e-12)
        tc = np.abs(np.einsum("ntc,nsc->ts", d, d))
        np.fill_diagonal(tc, np.nan)
        coh[k] = float(np.nanmean(tc[2:, 2:]))
    ax.barh(list(coh)[::-1], list(coh.values())[::-1],
            color=[cols[k] for k in list(coh)[::-1]], height=0.6)
    ax.tick_params(labelsize=8)
    for i, k in enumerate(list(coh)[::-1]):
        ax.text(coh[k] + 0.012, i, f"{coh[k]:.2f}", va="center", fontsize=8,
                color=INK)
    ax.set_xlim(0, max(coh.values()) * 1.22)
    _caption(ax, f"The scoreboard's clearest line. The corpus's 77 rows move together "
                 f"at {coh['corpus']:.2f}; the diagonal\nsampler's move at "
                 f"{coh['diagonal T=1']:.2f} -- it is generating 77 mutually unrelated "
                 f"summaries of 77 different\nprompts and handing them to the model as "
                 f"one. pca ({coh['pca T=1']:.2f}) and hybrid ({coh['hybrid']:.2f}) "
                 f"recover it; blend at\ncoherence 0.5 lands at {coh['blend 0.5']:.2f}, "
                 f"midway, exactly as its covariance interpolation predicts.")

    _finish(fig, top)
    ctx["stats"].update(nn_cosine_median={k: float(np.median(v)) for k, v in nn.items()},
                        sampler_abs_kurtosis=vals, sampler_position_coherence=coh)
    return fig


# ================================================================== main ===
def build_context(S, args) -> dict:
    """Shared expensive intermediates several figures need."""
    ctx = {"stats": {}}
    z = (S.X - S.mean[None]) / np.maximum(S.std, 1e-9)[None]
    ctx["central"] = central_mass(z)
    ctx["eta"] = pad_gating(S.X, S.eos_pos)
    med = int(np.median(S.eos_pos))
    # Block boundaries follow the corpus's OWN prompt-length quartiles rather
    # than fixed positions. Keyed to eos_pos.max(), the "always pad" block
    # collapses to a single position on any corpus that fills the 77-token
    # window, and the panel stops measuring anything.
    lo, hi = (int(np.percentile(S.eos_pos, 25)), int(np.percentile(S.eos_pos, 75)))
    lo, hi = max(lo, 2), max(hi, lo + 1)
    ctx["blocks"] = {
        f"pos 1-{lo}\n(content in >75%)": ((lo - 1) * S.H,
                                           block_participation(S.X, 1, lo)),
        f"pos {lo}-{hi}\n(mixed)": ((hi - lo) * S.H,
                                    block_participation(S.X, lo, hi)),
        f"pos {hi}-{S.T - 1}\n(pad in >75%)": ((S.T - hi) * S.H,
                                               block_participation(S.X, hi, S.T)),
    }
    # BIC on the leading scores: the range must reach past any plausible optimum,
    # otherwise "BIC picks the largest k offered" gets misread as a real answer.
    ctx["bic"] = gmm_bic(S.pca_scores[:, :20],
                         ks=(1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 45, 65, 90))

    if args.ckpt and Path(args.ckpt).exists():
        print("[analyze] reading cross-attention weights ...")
        ctx["sens"] = cross_attn_sensitivity(Path(args.ckpt))
        ctx["readable"] = readable_variance_ratio(
            S.X, S.mean, cross_attn_weights(Path(args.ckpt)), [1, 15, med, 60])

    try:
        from semantic_anarchy.distribution import EmbeddingDistribution
        dist = EmbeddingDistribution.load(args.dist)
        rng = np.random.default_rng(0)
        n = 512
        ctx["samples"] = {
            "corpus": S.X[:n].astype(np.float64),
            "diagonal T=1": dist.sample(n, sampler="diagonal", rng=rng),
            "pca T=1": dist.sample(n, sampler="pca", rng=rng),
            "pca T=2": dist.sample(n, temperature=2.0, sampler="pca", rng=rng),
            "blend 0.5": dist.sample(n, sampler="blend", coherence=0.5, rng=rng),
            "hybrid": dist.sample(n, sampler="hybrid", rng=rng),
        }
    except Exception as e:                                   # pragma: no cover
        print(f"[analyze] no fitted distribution ({e}); figures 10/12 reduced")
    ctx["sample_colors"] = {"corpus": INK, "diagonal T=1": ORANGE, "pca T=1": BLUE,
                            "pca T=2": AQUA, "blend 0.5": VIOLET, "hybrid": MAGENTA}
    return ctx


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=Path("prompts_1000.txt"))
    p.add_argument("--backend", default="sd15")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint: needed to (re)encode and for figure 11")
    p.add_argument("--model", default=None)
    p.add_argument("--dist", type=Path, default=Path("outputs/dist"),
                   help="fitted distribution prefix, for the sampler autopsy")
    p.add_argument("--outdir", type=Path, default=Path("outputs/analysis"))
    p.add_argument("--only", default=None, help="comma-separated figure numbers")
    p.add_argument("--refit", action="store_true", help="recompute cached stats")
    args = p.parse_args(argv)

    plots = args.outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    cache = args.outdir / f"corpus_{args.backend}.npz"
    scache = args.outdir / f"stats_{args.backend}.npz"

    if not cache.exists():
        print(f"[analyze] encoding {args.prompts} -> {cache} (one time) ...")
        encode_corpus(args.prompts, cache, args.backend, args.ckpt, args.model)
    if args.refit and scache.exists():
        scache.unlink()

    print("[analyze] loading corpus + statistics ...")
    S = CorpusStats.load(cache, keep_components=64, stats_cache=scache)
    print(f"[analyze]   {S.n} prompts, {S.T}x{S.H}, total_var {S.total_var:.0f}")
    describe(S)

    ctx = build_context(S, args)
    ctx["stats"].update(corpus_prompts=str(args.prompts), corpus_ckpt=str(args.ckpt),
                        corpus_backend=args.backend, dist_prefix=str(args.dist))
    want = set(args.only.split(",")) if args.only else set(FIGURES)
    for num in sorted(FIGURES):
        if num not in want:
            continue
        fn = FIGURES[num]
        print(f"[analyze] figure {num}: {fn.meta[1]}")
        fig = fn(S, ctx)
        if fig is None:
            print("[analyze]   skipped (missing inputs)")
            continue
        out = plots / f"{num}_{fn.__name__.replace('fig_', '')}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[analyze]   -> {out}")

    sp = args.outdir / "stats.json"
    prev = json.loads(sp.read_text()) if sp.exists() else {}
    prev.update(ctx["stats"])
    sp.write_text(json.dumps(prev, indent=2, default=float))
    print(f"[analyze] stats -> {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
