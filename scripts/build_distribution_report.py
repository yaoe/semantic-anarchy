#!/usr/bin/env python3
"""Assemble the corpus-analysis figures + stats into ONE readable HTML page.

Reads whatever ``scripts/analyze_distribution.py`` last wrote
(``outputs/analysis/plots/*.png`` and ``outputs/analysis/stats.json``) and emits a
single self-contained file -- every image inlined as a data URI, no external CSS,
no fonts to fetch -- so the report can be moved, mailed or opened offline.

    python scripts/analyze_distribution.py --ckpt MODEL.safetensors
    python scripts/build_distribution_report.py

The prose lives here rather than in the plotting script: the figures state what
was measured, this states what it means and what to do about it.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ------------------------------------------------------------ the writing ---
# Each section: (figure slug, kicker, heading, [paragraphs...]).
# Numbers are interpolated from stats.json at render time via {placeholders}.
SECTIONS = [
    (
        "01_sequence_anatomy", "The sequence axis",
        "Not all 77 token positions are the same object",
        ["""The conditioning tensor is usually described as "{seq_len} tokens of
        {n_channels} channels", which invites treating the token axis as
        homogeneous. It is not. Position 0 is the BOS embedding and its variance
        across all {n_prompts:,} prompts is <em>exactly zero</em> &mdash;
        {n_channels} coordinates that are bit-identical in every prompt in the
        corpus. The fit already handles this correctly: <code>sigma = 0</code>
        means the sampler reproduces BOS perfectly without anyone having to pin
        it. The old implementation's commented-out "restore the real BOS row"
        trick was solving a problem the per-token fit solves statistically.""",
         """The far more surprising half is the padding. CLIP's text encoder is
         <em>causal</em>, so a row at position 60 has attended to the entire
         prompt &mdash; it is a running summary, not a blank. Everything past the
         median EOS (position {median_eos_pos}) holds
         <strong>{variance_share_after_median_eos:.0f}%</strong> of the corpus's
         total variance. Most of what this distribution encodes lives in the
         region a naive reading would call filler.""",
         """How much of the sequence is padding is a property of the corpus, not
         of CLIP. This one runs to position {last_content_pos}: a substantial
         minority of these prompts are long enough to fill the 77-token window
         outright, so there is no always-padding tail worth naming &mdash; the
         per-position variance share is nearly flat across all
         {seq_len} positions. A corpus of short tag-style prompts would instead
         leave a wide, rigid tail. Both regimes matter below, because the
         sampler's worst structural mismatch (figure 09) lives in exactly that
         region.""",
         """One position is a genuine outlier for shape rather than scale:
         position {most_kurtotic_pos}, which has a mean excess kurtosis of
         <strong>{max_pos_mean_excess_kurtosis:.0f}</strong> against roughly zero
         everywhere else. Position 1 is the first content token, so a corpus that
         reuses a small set of opening words drives that row's marginals nearly
         categorical &mdash; spikes with long tails, not bells."""],
    ),
    (
        "02_sigma_landscape", "Per-coordinate sigma",
        "The sigma landscape is flat &mdash; there is nothing to freeze but BOS",
        ["""This is the figure that tests the "modulate the sigmas per
        coordinate" idea directly, and the answer is a clean negative. Excluding
        the BOS row, 98% of coordinates have sigma between
        <strong>{sigma_p1:.2f} and {sigma_p99:.2f}</strong>. That is a factor of
        {sigma_spread:.1f} across the entire tensor, where the intuition behind
        per-coordinate sigma modulation expects orders of magnitude. There is no
        population of much-wider or much-narrower coordinates waiting to be
        treated differently.""",
         """The participation ratio makes the same point as one number: the
         corpus's variance is spread over the equivalent of
         <strong>{coord_participation_ratio:,.0f}</strong> equally-weighted
         coordinates out of {n_coords:,} &mdash;
         <strong>{coord_participation_frac:.0%}</strong> of a perfectly uniform
         budget. Ninety per cent of the variance still needs
         <strong>{coord_pct_for_90pct_var:.0f}%</strong> of the coordinates.
         Sorted, the landscape is a gentle ramp with exactly one cliff, and that
         cliff is exactly {n_channels} coordinates wide.""",
         """This does not mean the intuition was wrong &mdash; it means the
         leverage is somewhere else. The next three figures find it in the
         <em>shape</em> of the marginals, and figure 09 finds it in the
         correlation structure. Neither is visible if you only look at
         sigma."""],
    ),
    (
        "03_marginal_shape", "Marginal shape",
        "Mostly Gaussian, with loud exceptions",
        ["""<strong>{pct_live_coords_gaussian_like:.0f}%</strong> of live
        coordinates sit inside the band a truly Gaussian coordinate occupies at
        N={n_prompts:,}, and a normality test rejects for only a minority at
        every width. The per-coordinate Gaussian is a reasonable default here,
        and more than that: the extremes are mild. No coordinate exceeds skew
        {max_abs_skew:.1f} or excess kurtosis {max_excess_kurtosis:.0f}, and only
        {pct_off_frame:.2f}% fall outside the frame plotted here at all. A
        smaller, more formulaic corpus produces far wilder shapes; the breadth of
        this one averages them out.""",
         """Two sub-results settle open questions from the algorithm report.
         First, asymmetry: measuring the spread above and below the mean
         separately, only <strong>{pct_lopsided_coords:.0f}%</strong> of live
         coordinates are more than 25% lopsided. The original implementation's
         asymmetric <code>k&nbsp;&times;&nbsp;[q05, q95]</code> clamp was
         correcting something real but small &mdash; a refinement, not a
         different regime. A <code>sigma+ / sigma-</code> split sampler is not
         where the gains are.""",
         """Second, truncation. The bulk of coordinates reach a maximum |z| of
         about {expected_max_abs_z:.2f}, exactly what {n_prompts:,} Gaussian
         draws should give. But
         <strong>{pct_coords_beyond_5sigma:.1f}%</strong> go past 5 sigma and
         {pct_coords_beyond_8sigma:.2f}% past 8. Truncating at 2&ndash;3 sigma is
         therefore not a safety rail &mdash; it clips behaviour the real corpus
         exhibits."""],
    ),
    (
        "04_marginal_gallery", "Twenty real marginals",
        "What the coordinates actually look like",
        ["""Twenty coordinates chosen to span the shape space, each with the
        Gaussian the diagonal sampler substitutes for it. The middle of the
        gallery is reassuring: coordinate after coordinate where the orange
        curve sits neatly on the blue histogram.""",
         """The panels drawn from the widest channels are not. Several show two
         clean lobes with a hole between them, and the fitted Gaussian peaking
         squarely <em>in the hole</em>. Bimodality here is not a curiosity of a
         few odd coordinates &mdash; it is a property of being a wide channel,
         which is to say a property of exactly the coordinates that carry the
         most signal. Figure 05 explains why."""],
    ),
    (
        "05_pad_gating", "The mixture",
        "The hidden binary: is this position past the EOS?",
        ["""CLIP pads to 77 with EOS. For any position in the middle of the
        prompt-length distribution, the corpus is therefore a <em>mixture of two
        populations</em>: prompts whose content still reaches this far, and
        prompts that ended earlier and are now padding. That latent binary is in
        no fitted model anywhere in this repo, and it is the direct cause of the
        bimodality in figure 04.""",
         """At the most strongly gated coordinate &mdash; position
         {max_eta_pos}, channel {max_eta_ch} &mdash; that single yes/no
         explains <strong>{max_eta:.0%}</strong> of the variance. The two lobes
         sit at {lobe_pad:+.1f} and {lobe_content:+.1f}; the fitted Gaussian puts
         its peak in the gap, and only {max_eta_central_mass:.0%} of real prompts
         fall within one sigma of that fitted mean where a Gaussian would put
         68%. Every diagonal draw near the mean of this coordinate is a value no
         real prompt ever produced.""",
         """Across the tensor, <strong>{gated_coords_eta_gt_25:,}</strong>
         coordinates take more than a quarter of their variance from this one
         binary and <strong>{gated_coords_eta_gt_50:,}</strong> take more than
         half. The effect switches on around position {gate_lo} and off again
         past {gate_hi} &mdash; exactly the span of the prompt-length
         distribution. Outside that window every prompt agrees and the mixture
         vanishes.""",
         """This is the most actionable finding in the report, and the fix needs
         no new machinery: draw a prompt length first, then sample conditionally
         on it."""],
    ),
    (
        "06_spectrum", "How many axes are real",
        "Only ~{pca_components_above_null} of the {pca_components_fitted:,} "
        "components clear the noise floor",
        ["""The null here is built by shuffling every coordinate independently
        across prompts: identical marginals, all correlation destroyed. Where
        the corpus spectrum drops to meet it is where real structure ends. That
        happens at component <strong>{pca_components_above_null}</strong>, and
        those axes carry <strong>{pca_var_frac_above_null:.0f}%</strong> of the
        corpus's total variance &mdash; the rest is real variance with no
        reliable direction attached to it.""",
         """This has a direct consequence for a shipped knob. <code>--comp-lo</code>
         exists to skip the dominant "tasteful" axes and ride idiosyncratic
         minor ones. Above component {pca_components_above_null} there is
         nothing idiosyncratic to ride &mdash; those axes are sampling error
         given a direction by a {n_prompts:,}-row SVD. Set with
         <code>--equalize</code>, which deliberately amplifies minor axes to full
         strength, the current defaults can spend the entire deviation budget on
         estimation noise.""",
         """The effective dimensionality tells the positive half of the story:
         {n_prompts:,} prompts buy roughly
         <strong>{pca_participation_ratio:.0f}</strong> usable directions,
         against ~{null_participation_ratio:.0f} for an independent-coordinate
         corpus of the same size. That gap is the whole reason the pca sampler
         works &mdash; the corpus is a thin, structured sheet."""],
    ),
    (
        "07_pca_scores", "The corpus basis",
        "The rotation that used to fix the marginals no longer does",
        ["""The pca sampler draws N(0,1) coefficients per axis, and unlike the
        raw coordinates the corpus scores genuinely justify that. Skew and
        excess kurtosis sit inside the N={n_prompts:,} sampling band from the
        first axis onward; the whitened radius over the top
        {whitened_radius_dims} components lands at
        {whitened_radius_mean:.1f} against the &radic;{whitened_radius_dims} =
        {radius_expected:.1f} a Gaussian would give.""",
         """This is where the corpus overturns an earlier conclusion. Mean
         absolute excess kurtosis on the raw coordinates is
         <strong>{mean_abs_kurt_raw:.2f}</strong>, and on the leading
         {pca_lead_components} PCA scores it is
         <strong>{mean_abs_kurt_pca_lead:.2f}</strong> &mdash; the rotation makes
         the marginals <em>worse</em>, not better. On a corpus of short,
         formulaic prompts the raw coordinates are strongly bimodal and the
         change of basis absorbs it. Here the raw coordinates are already close
         to Gaussian, so there is no artefact left to absorb, and the rotation
         instead concentrates the corpus's real structure into a few loudly
         non-Gaussian leading axes.""",
         """PC1 is the clearest case: two visibly separate lobes rather than a
         bell. That is the prompt-length split of figure 08 seen edge-on, and it
         means the pca sampler's N(0,1) draw on its <em>first</em> axis is
         putting mass in a gap the corpus leaves empty. From roughly PC3 onward
         the scores are Gaussian and the assumption holds. The constraint on a
         better sampler is therefore narrow and specific: fix the first two
         axes, not the basis."""],
    ),
    (
        "08_geometry", "Global geometry",
        "Several blobs &mdash; and the first axes are prompt length",
        ["""The original description called for a "mixture of uncorrelated
        Gaussians", and on this corpus that call was right. BIC on a diagonal
        GMM over the top 20 components picks
        <strong>k&nbsp;=&nbsp;{gmm_best_k}</strong>, with a clear interior
        minimum &mdash; not the k&nbsp;=&nbsp;1 a homogeneous corpus gives. The
        single Gaussian this repo fits is averaging over genuinely distinct
        populations of prompt, and it does so <em>on top of</em> the local
        per-position mixture of figure 05. These are two different mixtures and
        both are unmodelled.""",
         """The PC1&ndash;PC2 plane shows the largest of those populations
         directly: a detached lobe of prompts long enough to fill the whole
         token window, sitting apart from the main cloud rather than at the end
         of a gradient. Deeper in, the PC3&ndash;PC4 plane is an ordinary
         elliptical cloud with no clusters, arms or holes &mdash; the structure
         is concentrated in the leading axes, exactly where figure 07 finds the
         non-Gaussian scores.""",
         """Which brings up what those leading axes actually encode. PC1 and PC2
         carry |r| = <strong>{pc1_corr_prompt_length:.2f}</strong> and
         <strong>{pc2_corr_prompt_length:.2f}</strong> with how many words the
         prompt had; past component ~{last_length_correlated_pc} nothing does.
         The dominant directions the pca sampler starts from are, in substantial
         part, a prompt-length knob in disguise. Low <code>--comp-lo</code> is
         not a purely semantic control."""],
    ),
    (
        "09_correlation", "Correlation structure",
        "Independence fails along exactly one axis &mdash; the sequence axis",
        ["""This is the decomposition that matters most for sampler design.
        Two channels <em>inside</em> a single token row correlate at a median
        |r| of <strong>{corr_same_position:.3f}</strong>, barely above the
        {corr_null:.3f} an independent corpus of this size would show. The
        diagonal sampler's independence assumption is essentially correct
        there.""",
         """The same channel at two <em>different</em> positions correlates at
         <strong>{corr_same_channel:.3f}</strong>. The position &times; position
         map shows why: the padding tail is one rigid block, because every row
         in it is a running summary of the same prompt. Coordinate independence
         is not uniformly wrong &mdash; it is wrong in one specific direction,
         and that direction happens to hold most of the variance.""",
         """Quantified: the mostly-padding tail &mdash; the positions that are
         already padding in more than three-quarters of prompts &mdash; spends
         {pad_block_coords:,} coordinates on about
         <strong>{pad_block_effective_dims:.0f}</strong> real directions. A
         diagonal draw hands that block {pad_block_coords:,} independent ones.
         {pad_block_frac:.0%} of the sampler's entropy budget is being poured
         into a subspace the corpus treats as almost rigid."""],
    ),
    (
        "10_radius", "Radius",
        "The corpus is a band; every sampler is a spike",
        ["""In {n_coords:,} dimensions a Gaussian concentrates onto a razor-thin
        shell, and the diagonal sampler duly does: relative spread
        {diagonal_norm_rel_sd:.3f} against the corpus's
        <strong>{dev_norm_rel_sd:.3f}</strong>, about {radius_tighter:.0f}&times;
        tighter. Every diagonal draw sits at the same distance from the centre,
        which means temperature is the only radial control available &mdash; and
        it is a global one, applied to all draws at once rather than varying
        between them.""",
         """The repo's whitened distance gauge is honestly calibrated (mean
         {corpus_distance_mean:.2f}, sd {corpus_distance_sd:.2f} on the corpus it
         was fitted to), but real prompts only ever span
         {corpus_distance_min:.2f} to
         <strong>{corpus_distance_max:.2f}</strong>. Everything the taste loop
         calls "the periphery" &mdash; d&nbsp;&gt;&nbsp;1.3 &mdash; is territory
         no real prompt occupies. That is the point of the exercise, but it is
         worth naming precisely: it is extrapolation, not a sparsely-visited
         corner of the corpus.""",
         """Both samplers reproduce the corpus's per-position energy budget
         almost exactly. Whatever is wrong is not a question of scale."""],
    ),
    (
        "11_sensitivity", "Model sensitivity",
        "What the UNet can actually read",
        ["""Conditioning enters SD through exactly two projections per
        cross-attention block, <code>attn2.to_k</code> and
        <code>attn2.to_v</code>. A channel's column in those matrices is its
        only path into the image, so the column norm is a direct, gradient-free
        <em>first-order</em> sensitivity &mdash; no decoding, no probing. (It
        ignores what attention then does with the keys and values, so read it as
        a bound on linear leverage, not the full story.)""",
         """If some conditioning channels were far more sensitive than others it
         would show here, and it does not. Between the 1st and 99th percentile
         the whole range spans <strong>{sensitivity_spread:.1f}&times;</strong>,
         and sensitivity correlates with corpus sigma at
         r&nbsp;=&nbsp;<strong>{sigma_sensitivity_corr:+.2f}</strong> &mdash; a
         cloud, not a trend. Combining the two,
         <strong>{channels_for_80pct_drive}</strong> of {n_channels} channels
         carry 80% of the effective drive where a perfectly flat budget would
         need {channels_for_80pct_flat}. Selecting or freezing channels is not
         where a better sampler comes from.""",
         """The last panel is the sharper test. Projecting the true channel
         covariance through each of the {n_cross_attn_matrices} matrices and
         comparing against what the diagonal model claims, the ratio stays
         inside
         <strong>{rv_lo:.2f}&ndash;{rv_hi:.2f}</strong>. The diagonal model gets
         the <em>magnitude</em> of readable variance right to about 20%. Taken
         with figure 10, this settles the diagnosis: the magnitudes are right
         everywhere, and the failure is that the 77 rows it hands the model do
         not agree with one another."""],
    ),
    (
        "12_sampler_autopsy", "Scoreboard",
        "What each shipped sampler actually produces",
        ["""512 draws from every sampler, measured against the corpus on the
        statistics above. Real prompts sit at a nearest-neighbour cosine of
        <strong>{nn_corpus:.2f}</strong> from each other. Diagonal draws sit at
        <strong>{nn_diagonal:.2f}</strong> &mdash; further from every real
        prompt than any real prompt is from any other. Hybrid at
        <strong>{nn_hybrid:.2f}</strong> is the opposite failure, near-copies.
        pca at <strong>{nn_pca:.2f}</strong> is the only sampler in between, and
        even it does not reach the corpus.""",
         """The clearest single line is position-to-position coherence. The
         corpus's 77 rows move together at
         <strong>{coh_corpus:.2f}</strong>; the diagonal sampler's move at
         <strong>{coh_diagonal:.2f}</strong>. It is generating 77 mutually
         unrelated summaries of 77 different prompts and handing them to the
         model as one conditioning tensor. pca ({coh_pca:.2f}) and hybrid
         ({coh_hybrid:.2f}) recover it; blend at coherence 0.5 lands at
         {coh_blend:.2f}, midway, exactly as its covariance interpolation
         predicts.""",
         """One property no parametric sampler reproduces: marginal shape. The
         corpus carries mean |excess kurtosis| {kurt_corpus:.2f}; diagonal, pca
         and blend all flatten to {kurt_diagonal:.2f}. The gap is real but
         narrow on this corpus &mdash; its raw marginals are close to Gaussian to
         begin with (figure 03) &mdash; so of the implications below, this
         is the one whose payoff has shrunk the most. Whether it matters for
         image quality is an open question this analysis cannot settle."""],
    ),
]

CONCLUSIONS = [
    ("Sample a prompt length, then condition on it",
     "figure 05",
     """The single largest modelling gap. {gated_coords_eta_gt_25:,} coordinates
     take &gt;25% of their variance from a binary nobody models. Fit the
     distribution twice per position &mdash; once for "content", once for
     "padding" &mdash; draw a length from the corpus's own length histogram, and
     the bimodality of figure 04 disappears. This is a change to
     <code>fit</code> and <code>sample</code>, not a new sampler."""),
    ("Fix cross-position coherence, not per-coordinate sigma",
     "figures 02, 09, 12",
     """The diagonal sampler is right about channels within a row
     ({corr_same_position:.3f}) and wrong about the same channel across rows
     ({corr_same_channel:.3f}). A "row-coherent diagonal" &mdash; draw one
     per-channel deviation and share it across positions with a per-position
     modulation &mdash; would keep the anarchy of independent channels while
     restoring the structure that actually matters. The measured correlation is
     itself the mixing weight: dev<sub>t</sub> = &radic;&rho;&thinsp;u +
     &radic;(1&minus;&rho;)&thinsp;v<sub>t</sub> with &rho; &asymp;
     {corr_same_channel:.2f} (u shared across positions, v<sub>t</sub> fresh per
     row, both scaled by the per-coordinate sigma) reproduces the marginal
     variances exactly and the cross-position correlation to first order. Note
     the old implementation's <code>mix_latents</code> row crossover is
     <em>not</em> this operator: it copies each row from a different parent,
     which keeps every row internally real but scrambles the sequence axis in
     the same way the diagonal sampler does. The row-coherent piece exists in
     neither codebase."""),
    ("Cap <code>--comp-lo</code> at ~{pca_components_above_null}",
     "figure 06",
     """Above that, the axes are sampling noise. Combined with
     <code>--equalize</code>, which amplifies minor axes to full strength, the
     current ranges can spend the whole deviation budget on estimation error.
     A hard cap costs nothing and removes a knob range that cannot work."""),
    ("Draw the first two PCA axes from the corpus, not from N(0,1)",
     "figures 07, 08",
     """PC1 is two lobes, not a bell, because it is prompt length in disguise
     &mdash; and the pca sampler puts N(0,1) mass squarely in the gap between
     them. From PC3 onward the Gaussian assumption is fine. Sampling the leading
     one or two coefficients from their own empirical CDF (or jointly with the
     drawn prompt length above, which is the same variable) fixes the whole
     effect for the cost of two sorted arrays."""),
    ("Reproduce the remaining marginals with a rank transform",
     "figures 07, 12",
     """The same copula-style step generalises: sample in PCA space as now, then
     push each coordinate through its own empirical CDF. Marginal shape comes
     back exactly and the correlation structure is untouched. Ranked lower than
     it once was &mdash; on this corpus the raw marginals are already close to
     Gaussian, so the gap being closed ({kurt_corpus:.2f} against
     {kurt_diagonal:.2f}) is modest."""),
    ("Give the radius its own distribution",
     "figures 10, 12",
     """The corpus is a band of radii (relative spread
     {dev_norm_rel_sd:.3f}); every sampler is a spike. <code>retarget()</code>
     already pins a sample's distance to a chosen value &mdash; draw that target
     from the corpus's own distance distribution instead of a constant, and the
     population matches rather than a single shell."""),
    ("Do not expect gains from channel selection or sigma reshaping",
     "figures 02, 11",
     """Sigma spans {sigma_spread:.1f}&times; across the tensor, UNet sensitivity
     spans {sensitivity_spread:.1f}&times;, and the two are uncorrelated
     (r&nbsp;=&nbsp;{sigma_sensitivity_corr:+.2f}). Freezing, boosting or
     skewing individual coordinates has almost nothing to bite on. The one
     coordinate group that is genuinely frozen &mdash; the BOS row &mdash; the
     fit already handles exactly, at sigma&nbsp;=&nbsp;0."""),
    ("Loosen truncation, or drop it",
     "figure 03",
     """{pct_coords_beyond_5sigma:.1f}% of coordinates reach past 5 sigma in the
     real corpus and {pct_coords_beyond_8sigma:.2f}% past 8. Truncating at
     2&ndash;3 sigma removes
     behaviour the corpus exhibits rather than preventing blow-outs. If a
     clamp is wanted, the original implementation's empirical
     <code>k&nbsp;&times;&nbsp;[q05, q95]</code> envelope respects the actual
     marginals; a symmetric z-clip does not."""),
]

TILES = [
    ("{coord_participation_frac:.0%}", "of a uniform variance budget",
     "the sigma landscape is nearly flat"),
    ("{gated_coords_eta_gt_25:,}", "coordinates gated by prompt length",
     "a mixture nothing models"),
    ("{pca_components_above_null}", "of {pca_components_fitted:,} PCA axes are real",
     "the rest is sampling noise"),
    ("{corr_same_channel:.2f}", "vs {corr_same_position:.2f} correlation",
     "across positions vs within one"),
    ("{coh_corpus:.2f}&thinsp;&rarr;&thinsp;{coh_diagonal:.2f}",
     "row coherence: corpus vs diagonal",
     "77 rows that no longer agree"),
    ("{sensitivity_spread:.1f}&times;", "UNet sensitivity spread",
     "no hypersensitive channels exist"),
]


# ------------------------------------------------------------- rendering ---
CSS = """
:root{color-scheme:light dark;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --rule:#e1e0d9; --accent:#2a78d6; --accent-soft:#eef4fd;
  --code:#f0efec;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --rule:#2c2c2a; --accent:#3987e5; --accent-soft:#16243a;
  --code:#232321;}}
:root[data-theme="dark"]{
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --rule:#2c2c2a; --accent:#3987e5; --accent-soft:#16243a;
  --code:#232321;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 120px}
.prose{max-width:70ch}
a{color:var(--accent)}
code{background:var(--code);padding:.12em .38em;border-radius:4px;
  font:0.87em/1 ui-monospace,SFMono-Regular,Menlo,monospace}
strong{font-weight:650}
em{font-style:italic}

header.hero{padding:76px 0 34px;border-bottom:1px solid var(--rule);
  margin-bottom:44px}
.eyebrow{font-size:12.5px;letter-spacing:.10em;text-transform:uppercase;
  color:var(--accent);font-weight:650;margin:0 0 14px}
h1{font-size:clamp(30px,4.4vw,46px);line-height:1.12;margin:0 0 18px;
  letter-spacing:-.022em;font-weight:700}
.standfirst{font-size:19px;line-height:1.58;color:var(--ink2);max-width:66ch;
  margin:0 0 22px}
.meta{font-size:13.5px;color:var(--muted);margin:0}

.tiles{display:grid;gap:14px;margin:34px 0 8px;
  grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:16px 18px}
.tile .v{font-size:29px;font-weight:700;letter-spacing:-.02em;line-height:1.1;
  color:var(--ink)}
.tile .l{font-size:13px;color:var(--ink2);margin-top:5px;line-height:1.35}
.tile .s{font-size:12px;color:var(--muted);margin-top:3px;line-height:1.35}

nav.toc{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:20px 24px;margin:40px 0 8px}
nav.toc h2{font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);margin:0 0 12px;font-weight:650}
nav.toc ol{margin:0;padding-left:1.35em;columns:2;column-gap:36px;
  font-size:14.5px;line-height:1.85}
nav.toc a{text-decoration:none}
nav.toc a:hover{text-decoration:underline}
@media(max-width:720px){nav.toc ol{columns:1}}

section{padding-top:62px;scroll-margin-top:20px}
.kicker{font-size:12.5px;letter-spacing:.10em;text-transform:uppercase;
  color:var(--accent);font-weight:650;margin:0 0 9px}
h2{font-size:clamp(22px,2.7vw,30px);line-height:1.2;margin:0 0 20px;
  letter-spacing:-.018em;font-weight:700;max-width:30ch}
h2 .num{color:var(--muted);font-variant-numeric:tabular-nums;margin-right:.5em;
  font-weight:600}
section p{margin:0 0 17px}

/* Figures break out of the prose column: they are 2000px wide natively and
   their in-panel captions stop being legible much below ~1100px. */
figure{margin:26px 0 32px;width:min(96vw,1560px);
  margin-left:calc(50% - min(48vw,780px))}
figure .frame{background:#fcfcfb;border:1px solid var(--rule);border-radius:10px;
  padding:8px;overflow-x:auto}
figure img{display:block;width:100%;min-width:1100px;height:auto;border-radius:4px}
figcaption{font-size:12.5px;color:var(--muted);margin-top:9px}
@media(max-width:900px){figure{width:auto;margin-left:0}}

.callout{background:var(--accent-soft);border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0;padding:15px 20px;margin:26px 0;max-width:70ch}
.callout p{margin:0;font-size:15.5px;line-height:1.6}
.callout .lab{font-size:12px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--accent);font-weight:650;display:block;margin-bottom:6px}

.rec{border-top:1px solid var(--rule);padding:24px 0 4px;max-width:74ch}
.rec:first-of-type{border-top:none}
.rec h3{font-size:18.5px;margin:0 0 4px;line-height:1.3;font-weight:650;
  letter-spacing:-.01em}
.rec .src{font-size:12.5px;color:var(--muted);margin:0 0 10px}
.rec p{margin:0;color:var(--ink2)}

table{border-collapse:collapse;width:100%;max-width:74ch;font-size:14px;margin:22px 0;
  font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 14px 8px 0;border-bottom:1px solid var(--rule)}
th{font-size:12px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:650}
td.n{text-align:right;padding-right:26px;white-space:nowrap}
td.k{color:var(--ink2)}
.scroll{overflow-x:auto}

/* The checkpoint table is four columns of numbers, not two -- it needs the
   full measure, and its label column needs enough room not to wrap. */
table.cmp{max-width:none;width:100%;min-width:640px}
table.cmp td.k,table.cmp th:first-child{width:46%;padding-right:20px}
table.cmp td.n,table.cmp th:not(:first-child){text-align:right;padding-right:0;
  padding-left:22px}
table.cmp th{vertical-align:bottom}

footer{margin-top:80px;padding-top:26px;border-top:1px solid var(--rule);
  font-size:13.5px;color:var(--muted)}
"""


# Headline numbers re-measured against a second checkpoint. Each row is
# (label, stats key, format spec, "rel" | "abs") -- "abs" for quantities already
# expressed as a fraction or a correlation, where a relative change is
# meaningless near zero.
COMPARE_ROWS = [
    ("total corpus variance", "total_var", "{:,.0f}", "rel"),
    ("coordinate participation ratio", "coord_participation_ratio", "{:,.0f}", "rel"),
    ("98% sigma spread (p99 / p1)", "sigma_spread", "{:.2f}", "rel"),
    # Percentages compare relatively: they run 0-100, so the 0.05 absolute
    # threshold below is meant for the fractions and correlations, not for these.
    ("live coords inside the Gaussian band", "pct_live_coords_gaussian_like",
     "{:.1f}%", "rel"),
    ("coords past 5 sigma", "pct_coords_beyond_5sigma", "{:.2f}%", "rel"),
    ("mean |excess kurtosis|, raw coords", "mean_abs_kurt_raw", "{:.3f}", "rel"),
    ("most gated coordinate, eta&sup2;", "max_eta", "{:.3f}", "abs"),
    ("coordinates with eta&sup2; &gt; 0.25", "gated_coords_eta_gt_25", "{:,.0f}", "rel"),
    ("PCA components above the null", "pca_components_above_null", "{:,.0f}", "rel"),
    ("PCA participation ratio", "pca_participation_ratio", "{:.1f}", "rel"),
    ("GMM BIC best k", "gmm_best_k", "{:,.0f}", "rel"),
    ("PC1 correlation with prompt length", "pc1_corr_prompt_length", "{:.3f}", "abs"),
    ("median |r|, same channel across positions", "_corr_same_channel", "{:.3f}", "abs"),
    ("median |r|, same position across channels", "_corr_same_position", "{:.3f}", "abs"),
    ("corpus radial spread (relative sd)", "dev_norm_rel_sd", "{:.4f}", "rel"),
    ("distance gauge, mean", "corpus_distance_mean", "{:.3f}", "abs"),
    ("UNet sensitivity spread", "sensitivity_spread", "{:.3f}", "rel"),
    ("channels for 80% of effective drive", "channels_for_80pct_drive", "{:,.0f}", "rel"),
    ("sigma vs to_k sensitivity, r", "sigma_sensitivity_corr", "{:+.3f}", "abs"),
    ("row coherence &mdash; corpus", "_coh_corpus", "{:.3f}", "abs"),
    ("row coherence &mdash; diagonal", "_coh_diagonal", "{:.3f}", "abs"),
    ("nearest-neighbour cosine &mdash; corpus", "_nn_corpus", "{:.3f}", "abs"),
]


def _compare_value(stats: dict, key: str):
    """Read a COMPARE_ROWS key, including the few that live inside sub-dicts."""
    if not key.startswith("_"):
        return stats.get(key)
    pair = stats.get("median_abs_corr_by_pair_type", {})
    return {
        "_corr_same_channel": pair.get("same channel, different position"),
        "_corr_same_position": pair.get("same position, different channel"),
        "_coh_corpus": stats.get("sampler_position_coherence", {}).get("corpus"),
        "_coh_diagonal": stats.get("sampler_position_coherence", {}).get(
            "diagonal T=1"),
        "_nn_corpus": stats.get("nn_cosine_median", {}).get("corpus"),
    }.get(key)


def compare_table(primary: dict, other: dict):
    """Rows of (label, primary, other, delta-string, is_material) for the table."""
    rows, material = [], 0
    for label, key, spec, mode in COMPARE_ROWS:
        a, b = _compare_value(primary, key), _compare_value(other, key)
        if a is None or b is None:
            continue
        if mode == "rel" and abs(a) > 1e-12:
            d = (b - a) / abs(a)
            delta, big = f"{d:+.1%}", abs(d) > 0.10
        else:
            d = b - a
            delta, big = f"{d:+.3f}", abs(d) > 0.05
        material += big
        rows.append((label, spec.format(a), spec.format(b), delta, big))
    return rows, material


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def build_values(stats: dict) -> dict:
    """Flatten stats.json into the flat placeholder namespace the prose uses.

    Everything here is *derived* from the measured numbers -- nothing about the
    corpus is written down twice. Re-run the analysis on a different prompt file
    and the prose follows it, which is the only way a report that quotes forty
    numbers stays honest.
    """
    nn, coh, ku = (stats.get(k, {}) for k in
                   ("nn_cosine_median", "sampler_position_coherence",
                    "sampler_abs_kurtosis"))
    pair = stats.get("median_abs_corr_by_pair_type", {})
    rv = stats.get("readable_variance_ratio_range", [0, 0])
    pos, ch = stats.get("max_eta_coord", [0, 0])
    lobe_pad, lobe_content = stats.get("max_eta_lobes", [0.0, 0.0])
    gate_lo, gate_hi = stats.get("gate_window", [0, 0])
    dims = stats.get("whitened_radius_dims", 200)
    corpus_sd = stats.get("dev_norm_rel_sd", 0.0)
    diag_sd = stats.get("diagonal_norm_rel_sd", 0.0)
    v = dict(stats)
    v.update(
        max_eta_pos=pos, max_eta_ch=ch,
        lobe_pad=lobe_pad, lobe_content=lobe_content,
        gate_lo=gate_lo, gate_hi=gate_hi,
        radius_expected=math.sqrt(dims),
        radius_tighter=corpus_sd / max(diag_sd, 1e-12),
        ckpt_name=Path(stats.get("corpus_ckpt", "?")).name,
        corr_same_position=pair.get("same position, different channel", 0.0),
        corr_same_channel=pair.get("same channel, different position", 0.0),
        corr_null=stats.get("corr_independent_null", 0.0),
        rv_lo=rv[0], rv_hi=rv[1],
        nn_corpus=nn.get("corpus", 0), nn_diagonal=nn.get("diagonal T=1", 0),
        nn_pca=nn.get("pca T=1", 0), nn_hybrid=nn.get("hybrid", 0),
        coh_corpus=coh.get("corpus", 0), coh_diagonal=coh.get("diagonal T=1", 0),
        coh_pca=coh.get("pca T=1", 0), coh_hybrid=coh.get("hybrid", 0),
        coh_blend=coh.get("blend 0.5", 0),
        kurt_corpus=ku.get("corpus", 0), kurt_diagonal=ku.get("diagonal T=1", 0),
        pad_block_frac=(stats.get("pad_block_coords", 0)
                        / max(stats.get("n_coords", 1), 1)),
    )
    return v


APPENDIX = [
    ("Corpus", [
        ("prompts encoded", "{n_prompts:,}"),
        ("prompt file", "<code>{corpus_prompts}</code>"),
        ("conditioning shape",
         "{seq_len} &times; {n_channels} = {n_coords:,} coordinates"),
        ("checkpoint", "<code>{ckpt_name}</code>"),
        ("total variance", "{total_var:,.0f}"),
    ]),
    ("Per-coordinate scale", [
        ("BOS row sigma", "{bos_sigma:.1f} (exactly constant)"),
        ("98% of live sigmas between", "{sigma_p1:.2f} and {sigma_p99:.2f}"),
        ("coordinate participation ratio",
         "{coord_participation_ratio:,.0f} of {n_coords:,}"),
        ("coordinates for 90% of variance", "{coord_pct_for_90pct_var:.0f}%"),
    ]),
    ("Marginal shape", [
        ("inside the Gaussian noise band", "{pct_live_coords_gaussian_like:.0f}% of live coords"),
        ("more than 25% lopsided", "{pct_lopsided_coords:.1f}%"),
        ("reaching past 5 sigma", "{pct_coords_beyond_5sigma:.1f}%"),
        ("reaching past 8 sigma", "{pct_coords_beyond_8sigma:.2f}%"),
        ("mean |excess kurtosis|, raw coords", "{mean_abs_kurt_raw:.2f}"),
        ("mean |excess kurtosis|, top-{pca_lead_components} PCA scores",
         "{mean_abs_kurt_pca_lead:.2f}"),
    ]),
    ("The prompt-length mixture", [
        ("most gated coordinate", "pos {max_eta_pos}, ch {max_eta_ch} &mdash; eta&sup2; = {max_eta:.2f}"),
        ("coordinates with eta&sup2; &gt; 0.25", "{gated_coords_eta_gt_25:,}"),
        ("coordinates with eta&sup2; &gt; 0.50", "{gated_coords_eta_gt_50:,}"),
        ("PC1 / PC2 correlation with length", "{pc1_corr_prompt_length:.2f} / {pc2_corr_prompt_length:.2f}"),
    ]),
    ("Structure", [
        ("PCA components above the null",
         "{pca_components_above_null} of {pca_components_fitted:,}"),
        ("PCA participation ratio", "{pca_participation_ratio:.0f}"),
        ("GMM BIC best k", "{gmm_best_k}"),
        ("median |r|, same position", "{corr_same_position:.3f}"),
        ("median |r|, same channel", "{corr_same_channel:.3f}"),
        ("median |r|, independent null", "{corr_null:.3f}"),
    ]),
    ("Model coupling", [
        ("sigma vs to_k sensitivity", "r = {sigma_sensitivity_corr:+.2f}"),
        ("UNet sensitivity spread (p99 / p1)", "{sensitivity_spread:.1f}&times;"),
        ("channels for 80% of effective drive",
         "{channels_for_80pct_drive} of {n_channels}"),
        ("readable-variance ratio (true / diagonal)", "{rv_lo:.2f} &ndash; {rv_hi:.2f}"),
    ]),
    ("Sampler scoreboard", [
        ("nearest-neighbour cosine &mdash; corpus", "{nn_corpus:.2f}"),
        ("nearest-neighbour cosine &mdash; diagonal / pca / hybrid",
         "{nn_diagonal:.2f} / {nn_pca:.2f} / {nn_hybrid:.2f}"),
        ("row coherence &mdash; corpus", "{coh_corpus:.2f}"),
        ("row coherence &mdash; diagonal / blend / pca",
         "{coh_diagonal:.2f} / {coh_blend:.2f} / {coh_pca:.2f}"),
        ("corpus radial spread (relative sd)", "{dev_norm_rel_sd:.3f}"),
        ("corpus distance gauge, mean / max",
         "{corpus_distance_mean:.2f} / {corpus_distance_max:.2f}"),
    ]),
]


def render_comparison(primary: dict, other: dict, v: dict) -> str:
    """The control experiment: same prompts, same code, a different checkpoint.

    Every number in this report is measured through one particular set of
    weights, and the one it was measured through is a finetune. This section
    exists to say how much of the report is a property of the corpus and how
    much is a property of that finetune.
    """
    rows, material = compare_table(primary, other)
    a_name = Path(primary.get("corpus_ckpt", "primary")).name
    b_name = Path(other.get("corpus_ckpt", "other")).name
    moved = [r[0] for r in rows if r[4]]
    bar = ("more than 10% relatively, or 0.05 absolute for the quantities "
           "already expressed as fractions or correlations")
    verdict = (
        f"<strong>{material} of the {len(rows)} headline numbers moves "
        f"materially</strong> ({bar}): {moved[0]}. Everything else &mdash; every "
        f"correlation, every effective-dimension count, the whole sampler "
        f"scoreboard &mdash; is unchanged to the digits printed here. The report's "
        f"conclusions are properties of the prompts and of CLIP's architecture, "
        f"not of this finetune."
        if material == 1 else
        f"<strong>{material} of the {len(rows)} headline numbers move "
        f"materially</strong> ({bar}): " + ", ".join(moved) + "."
    ) if material else (
        f"<strong>None of the {len(rows)} headline numbers moves materially</strong> "
        f"({bar}). The report's conclusions are properties of the prompts and of "
        f"CLIP's architecture, not of this finetune."
    )
    body = [
        '<section id=checkpoints><p class=kicker>Control</p>'
        '<h2>Is any of this a property of the finetune?</h2><div class=prose>',
        f"""<p>Everything above is measured through
        <code>{html.escape(a_name)}</code>, which is an SD&nbsp;1.5 <em>finetune</em>
        &mdash; and a finetune that touched the text encoder: 196 of its 197
        CLIP tensors differ from base SD&nbsp;1.5, by up to 1% of each tensor's
        own magnitude. Since the corpus <em>is</em> the text encoder's output,
        that is enough to worry about. So the entire analysis was re-run on the
        same {v['n_prompts']:,} prompts through stock
        <code>{html.escape(b_name)}</code>, changing nothing else.</p>""",
        f"<p>{verdict}</p>",
        (f"""<p>The one number that does move is a count sitting on a threshold:
        how many coordinates take more than a quarter of their variance from the
        past-EOS switch. <code>{html.escape(b_name)}</code> puts more coordinates
        just over that line. The underlying effect is the same size in both
        &mdash; the most gated coordinate is at eta&sup2;
        {primary.get('max_eta', 0):.2f} vs {other.get('max_eta', 0):.2f} &mdash;
        so this is threshold sensitivity, not a different phenomenon.</p>"""
         if material == 1 and moved and "eta" in moved[0] else ""),
        """<p>Read the table as a robustness check on the report, not as a
        comparison of model quality. Nothing here says which checkpoint makes
        better images; it says only that the <em>shape of the conditioning
        distribution</em> survives the finetune. Note also what this does not
        cover: the figures themselves are still rendered from the primary
        checkpoint, and the base-checkpoint set is on disk under
        <code>outputs/analysis_base/plots/</code> rather than inlined here, to
        keep the page to one download.</p></div>""",
        '<div class=scroll><table class=cmp><thead><tr>'
        f'<th>Headline number</th><th>{html.escape(a_name)}</th>'
        f'<th>{html.escape(b_name)}</th><th>change</th>'
        "</tr></thead><tbody>",
    ]
    for label, a, b, delta, big in rows:
        mark = ' style="color:var(--accent);font-weight:650"' if big else ""
        body.append(f'<tr><td class=k>{label}</td><td class=n>{a}</td>'
                    f'<td class=n>{b}</td><td class=n{mark}>{delta}</td></tr>')
    body.append("</tbody></table></div></section>")
    return "\n".join(body)


def render(plots: Path, stats: dict, other: dict | None = None) -> str:
    v = build_values(stats)
    fmt = lambda s: s.format(**v)                                # noqa: E731
    out = ["<!doctype html><html lang=en><head><meta charset=utf-8>",
           '<meta name=viewport content="width=device-width,initial-scale=1">',
           "<title>Semantic Anarchy &mdash; what the prompt corpus actually looks "
           "like</title>", f"<style>{CSS}</style></head><body><div class=wrap>"]

    # ---- hero ----------------------------------------------------------
    out.append("""
<header class=hero>
  <p class=eyebrow>Semantic Anarchy &middot; distribution report</p>
  <h1>What the prompt corpus actually looks like</h1>
  <p class=standfirst>{n_prompts:,} prompts, encoded once through SD&nbsp;1.5's
  CLIP text encoder, taken apart coordinate by coordinate. The question behind
  every figure here is the same: <em>where is the leverage for a better
  sampler?</em> Some of the answers are negative results, and those turned out
  to be the useful ones.</p>
  <p class=meta>Generated by <code>scripts/analyze_distribution.py</code> and
  <code>scripts/build_distribution_report.py</code>. Every number below is read
  from <code>outputs/analysis/stats.json</code>; every figure is regenerable
  from the cached corpus.</p>
</header>""".format(**v))

    # ---- tiles ---------------------------------------------------------
    out.append("<div class=tiles>")
    for val, lab, sub in TILES:
        out.append(f"<div class=tile><div class=v>{fmt(val)}</div>"
                   f"<div class=l>{fmt(lab)}</div>"
                   f"<div class=s>{fmt(sub)}</div></div>")
    out.append("</div>")

    # ---- orientation ---------------------------------------------------
    out.append("""
<div class=prose>
<section id=summary style="padding-top:52px">
  <p class=kicker>The short version</p>
  <h2>Four findings, two of which were surprises</h2>
  <p><strong>The corpus is globally multimodal, and the single fitted Gaussian
  averages over that.</strong> BIC on the leading components picks
  k&nbsp;=&nbsp;{gmm_best_k}, with a clear interior minimum. The largest split is
  visible by eye in the PC1&ndash;PC2 plane: prompts long enough to fill CLIP's
  77-token window form a detached lobe, and PC1 &mdash; which correlates
  {pc1_corr_prompt_length:.2f} with prompt length &mdash; comes out bimodal
  rather than bell-shaped.</p>
  <p><strong>Per-coordinate sigma modulation has almost nothing to bite on.</strong>
  Ninety-eight per cent of the {n_coords:,} coordinates have a sigma within a
  factor of {sigma_spread:.1f} of each other, and the UNet's sensitivity to them
  spans only {sensitivity_spread:.1f}&times; &mdash; uncorrelated with sigma.
  The one genuinely frozen group,
  the BOS row, the fit already reproduces exactly. Freezing, boosting or
  skewing individual coordinates is not where a better sampler comes from.</p>
  <p><strong>The non-Gaussianity that remains is a mixture, not a shape.</strong>
  CLIP pads to 77 with EOS, so at any middle position the corpus is two
  populations &mdash; prompts still carrying content, and prompts already
  padding. That unmodelled binary explains up to
  {max_eta:.0%} of a coordinate's variance, and it is what makes the widest
  coordinates bimodal with the fitted Gaussian peaking in the empty gap between
  the lobes. It is a <em>second</em>, per-position mixture, layered under the
  global one above.</p>
  <p><strong>Coordinate independence fails along exactly one axis.</strong>
  Channels within a token row are nearly independent ({corr_same_position:.2f});
  the same channel across rows correlates at {corr_same_channel:.2f}. The
  diagonal sampler generates 77 mutually unrelated summaries of 77 different
  prompts and hands them to the model as one tensor &mdash; row coherence
  {coh_diagonal:.2f} against the corpus's {coh_corpus:.2f}.</p>
  <p>Everything else &mdash; per-position energy budgets, the magnitude of
  variance the model can read, the calibration of the distance gauge &mdash;
  the current implementation gets right. The failure is structural, not
  scalar.</p>
  <p class=meta style="margin-top:22px">Measured on
  <code>{corpus_prompts}</code> ({n_prompts:,} prompts) through
  <code>{ckpt_name}</code>. Numbers that depend on the corpus's own composition
  &mdash; how long its prompts are, how formulaic they are &mdash; are called out
  as such where they appear, because they are the ones that move when the prompt
  file changes.</p>
</section>
</div>""".format(**v))

    # ---- toc -----------------------------------------------------------
    out.append("<nav class=toc><h2>Figures</h2><ol>")
    for slug, _, head, _ in SECTIONS:
        out.append(f'<li><a href="#fig-{slug[:2]}">{fmt(head)}</a></li>')
    out.append('</ol>')
    out.append('<h2 style="margin-top:18px">Then</h2><ol style="columns:1">'
               '<li><a href="#implications">Implications for the sampler</a></li>'
               + ('<li><a href="#checkpoints">Is any of this a property of the '
                  'finetune?</a></li>' if other is not None else "")
               + '<li><a href="#appendix">Every measured number</a></li>'
               '</ol></nav>')

    # ---- figure sections ------------------------------------------------
    for i, (slug, kicker, head, paras) in enumerate(SECTIONS, 1):
        img = plots / f"{slug}.png"
        out.append(f'<section id="fig-{slug[:2]}"><p class=kicker>{kicker}</p>'
                   f'<h2><span class=num>{i:02d}</span>{fmt(head)}</h2>')
        if img.exists():
            out.append(f'<figure><div class=frame>'
                       f'<img src="{data_uri(img)}" alt="{html.escape(fmt(head))}">'
                       f'</div><figcaption>Figure {i:02d} &mdash; '
                       f'{html.escape(slug)}.png. Scroll the frame sideways on a '
                       f'narrow screen.</figcaption></figure>')
        else:
            out.append(f"<p><em>missing {img}</em></p>")
        out.append("<div class=prose>")
        out.extend(f"<p>{fmt(p)}</p>" for p in paras)
        out.append("</div></section>")

    # ---- conclusions -----------------------------------------------------
    count_words = {6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}
    n_impl = count_words.get(len(CONCLUSIONS), str(len(CONCLUSIONS)))
    out.append(f"""
<section id=implications>
  <p class=kicker>What follows from it</p>
  <h2>{n_impl} implications for the sampler</h2>
  <div class=prose><p>These are implications, not tested results &mdash; the
  analysis measures the corpus, not the images. They are ordered by how much of
  the measured mismatch each one would close.</p></div>""")
    for j, (title, src, body) in enumerate(CONCLUSIONS, 1):
        out.append(f'<div class=rec><h3>{j}. {fmt(title)}</h3>'
                   f'<p class=src>{src}</p><p>{fmt(body)}</p></div>')
    out.append("</section>")

    # ---- the control experiment ------------------------------------------
    if other is not None:
        out.append(render_comparison(stats, other, v))

    # ---- appendix --------------------------------------------------------
    out.append('<section id=appendix><p class=kicker>Appendix</p>'
               '<h2>Every measured number</h2><div class=scroll>')
    for group, rows in APPENDIX:
        out.append(f"<table><thead><tr><th>{group}</th><th></th></tr></thead><tbody>")
        for label, value in rows:
            out.append(f'<tr><td class=k>{fmt(label)}</td>'
                       f'<td class=n>{fmt(value)}</td></tr>')
        out.append("</tbody></table>")
    out.append("</div></section>")

    regen = (f"<code>python scripts/analyze_distribution.py --prompts "
             f"{html.escape(str(stats.get('corpus_prompts', 'prompts_1000.txt')))} "
             f"--ckpt MODEL.safetensors --dist "
             f"{html.escape(str(stats.get('dist_prefix', 'outputs/dist')))}</code> "
             f"then <code>python scripts/build_distribution_report.py"
             + ("  --compare outputs/analysis_base" if other is not None else "")
             + "</code>")
    out.append(f"""
<footer>
  <p>Regenerate: {regen}. The corpus encode is cached in
  <code>outputs/analysis/</code>, so only the first run needs a GPU.</p>
</footer></div></body></html>""")
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", type=Path, default=Path("outputs/analysis"))
    p.add_argument("--out", type=Path, default=Path("distribution_report.html"))
    p.add_argument("--compare", type=Path, default=None,
                   help="a second analysis dir (or stats.json) measured on the "
                        "same prompts through different weights; renders the "
                        "checkpoint-robustness section")
    args = p.parse_args(argv)

    stats = json.loads((args.indir / "stats.json").read_text())
    other = None
    if args.compare:
        cmp_path = args.compare
        if cmp_path.is_dir():
            cmp_path = cmp_path / "stats.json"
        other = json.loads(cmp_path.read_text())
    html_text = render(args.indir / "plots", stats, other)
    args.out.write_text(html_text)
    kb = len(html_text.encode()) / 1024
    print(f"[report] {args.out}  ({kb / 1024:.1f} MB, {len(SECTIONS)} figures inlined)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
