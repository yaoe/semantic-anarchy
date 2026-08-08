/**
 * The sidebar, as data.
 *
 * One object per knob. Rendering (ParamPanel/Field), form state (store.ts) and
 * the /api/run payload (buildRunRequest) are all derived from this array —
 * adding a control means adding one entry here and nothing else.
 *
 * Cross-checked against webui/app.py `RunRequest` / `build_argv` and
 * semantic_anarchy/cli_args.py `add_backend_args`, so every flag the backend
 * accepts has a home. `id` is the RunRequest field name; the handful of knobs
 * that are client-side only (aspect) carry `send: false`.
 */
import type { Config, RunRequest, TasteBand } from '../api/types'
import { evalExpr } from '../lib/calc'

/* ------------------------------------------------------------------ types */

export type Values = Record<string, string>

export type FieldType = 'segmented' | 'select' | 'number' | 'text' | 'textarea' | 'note'

export interface Option {
  value: string
  label: string
  title?: string
}

export interface Ctx {
  values: Values
  config?: Config
  tasteband?: TasteBand
}

export type Dyn<T> = T | ((ctx: Ctx) => T)

/** A hint line: plain text plus a tone, so schema.ts stays markup-free. */
export interface Hint {
  text: string
  tone?: 'dim' | 'warn'
}

export type GroupId =
  | 'action' | 'model' | 'sampler' | 'reach' | 'image' | 'sweep' | 'advanced'

export interface ParamField {
  id: string
  type: FieldType
  group: GroupId
  label?: Dyn<string>
  /** Visibility predicate: every key must match one of its listed values. */
  when?: Record<string, string[]>
  /**
   * Extra visibility predicate, ANDed with `when`. `when` is a conjunction of
   * value tests and cannot express "A or B" — which several knobs genuinely
   * need, because whether a flag is read depends on the action AND the sampler
   * in ways that don't factor (sampler_sweep forwards `--coherence` but no
   * other sampler knob; hybrid reads `temperature` even in shell mode).
   */
  visible?: (values: Values) => boolean
  options?: Dyn<Option[]>
  placeholder?: Dyn<string>
  step?: number
  min?: number
  tooltip?: Dyn<string>
  hint?: Dyn<Hint | string | null>
  /** Initial form value. '' means "left blank -> the script's own default". */
  default?: string
  /**
   * What `buildRunRequest` sends while the knob is hidden. Only needed when
   * `default` is an *active* value (one that changes what the run does): a
   * hidden knob must be a no-op, so the two part ways there.
   */
  hidden?: string
  cast?: 'number' | 'string' | 'bool'
  /** '' becomes null in the payload (optional string flags). */
  emptyAsNull?: boolean
  /** false = never sent to /api/run (pure UI affordance). */
  send?: boolean
  /** Grid columns inside the group (the sidebar is a 2-column grid). */
  span?: 1 | 2
  /**
   * Accept arithmetic: `1024+256` becomes `1280` on Enter or blur. Implies a
   * text input (a `type=number` box refuses to hold the expression while it is
   * being typed), so it is opt-in per knob.
   */
  expr?: boolean
  /** textarea only: visible rows. */
  rows?: number
  /**
   * Show `placeholder` as real, editable text the first time the box is
   * focused while empty. For knobs whose default is a *value you want to
   * tweak* (the negative prompt) rather than a value you want to replace —
   * a placeholder alone can be read but not edited. Clearing the box still
   * means "left blank -> the script's own default".
   */
  seedFromPlaceholder?: boolean
}

export interface Group {
  id: GroupId
  title?: string
  /** Rendered inside a <details> like the legacy "Advanced" block. */
  collapsible?: boolean
}

export const GROUPS: Group[] = [
  { id: 'action', title: 'Action' },
  { id: 'model', title: 'Model' },
  { id: 'sampler', title: 'Sampler' },
  { id: 'reach', title: 'Reach' },
  { id: 'image' },
  { id: 'sweep' },
  { id: 'advanced', title: 'Advanced', collapsible: true },
]

/* -------------------------------------------------------------- constants */

/** Actions that expose the sampler block (legacy `.genonly` = gen || sweep). */
const GEN_OR_SWEEP = ['generate', 'temp_sweep', 'sampler_sweep']
/** Actions that render actual images with per-image knobs (legacy `.genimg`). */
const GEN_ONLY = ['generate']

/* --------------------------------------------- what each action really reads */
/*
 * `build_argv` in webui/app.py does NOT forward the same flags to every script,
 * and the difference is invisible from the sidebar unless it is encoded here.
 * Two rules cover it:
 *
 *   sampler_sweep  renders its own diagonal/blend/pca rows and forwards only
 *                  --temperature/--coherence/--seeds. Every other sampler knob
 *                  (rho, length, comp-lo, equalize, truncation, empirical-head,
 *                  temp-on/off, and --sampler itself) is dropped on the floor.
 *   temp_sweep     forwards the whole sampler set but owns temperature itself
 *                  (--temps), and takes none of the reach flags.
 *
 * Only `generate` carries --target-distance / --radius-band / --min-distance;
 * the sweep scripts do not even define those arguments.
 */

/** Does this action forward the shared `--sampler ...` knob set? */
function usesSamplerKnobs(v: Values): boolean {
  return v.action === 'generate' || v.action === 'temp_sweep'
}

/**
 * Which knob owns "how far from the corpus centre" for the current form state.
 *
 * Temperature, a target shell and the corpus radius band are three spellings of
 * ONE quantity, and they are not additive: `retarget()` rescales each sample to
 * an exact distance, so the temperature factor cancels out of the result
 * algebraically (verified: identical tensors at T=1.0 and T=3.7 under a shell).
 * Presenting them as a mode instead of three coexisting boxes is what keeps a
 * live temperature from silently meaning nothing.
 *
 * Anything other than `generate` is always on plain temperature, because that
 * is the only action whose argv can carry the other two.
 */
export function reachMode(v: Values): 'temperature' | 'shell' | 'band' {
  if (v.action !== 'generate') return 'temperature'
  return v.reach === 'shell' || v.reach === 'band' ? v.reach : 'temperature'
}

/** Native resolution per backend — drives the aspect-ratio presets. */
export const NATIVE_RES: Record<string, number> = {
  sd15: 512,
  sd2: 768,
  sdxl: 1024,
  flux2: 1024,
  krea2: 1024,
}

/** SDXL per-model step/guidance defaults, mirrored from SDXL_MODEL_DEFAULTS. */
const SDXL_MODEL_DEFAULTS: Record<string, { steps: number; guidance: number }> = {
  'sdxl-base-1.0': { steps: 30, guidance: 7 },
  'sdxl-turbo': { steps: 1, guidance: 0 },
}

/**
 * Per-backend step/guidance defaults, mirrored from `BACKEND_DEFAULTS` in
 * semantic_anarchy/backend.py — the numbers `resolve_gen_defaults` fills in
 * when the flag is absent. No `sdxl` entry on purpose: app.py's `_gen_flags`
 * never lets sdxl reach those (it backfills from SDXL_MODEL_DEFAULTS above so
 * base can't fall through to turbo's 1-step/no-CFG), so `genDefault` routes
 * sdxl through the checkpoint table instead.
 *
 * These exist so a blank box still SHOWS the number the run will use — the
 * sidebar never says "auto".
 */
const BACKEND_GEN_DEFAULTS: Record<string, { steps: number; guidance: number }> = {
  sd15: { steps: 30, guidance: 7.5 },
  sd2: { steps: 30, guidance: 9 },
  flux2: { steps: 28, guidance: 4 },
  krea2: { steps: 28, guidance: 4.5 },
}

/** The steps/guidance a blank box resolves to, for the current backend+ckpt. */
function genDefault(values: Values, key: 'steps' | 'guidance'): string {
  const d =
    values.backend === 'sdxl'
      ? (SDXL_MODEL_DEFAULTS[values.model] ?? SDXL_MODEL_DEFAULTS['sdxl-base-1.0'])
      : BACKEND_GEN_DEFAULTS[values.backend]
  return d ? String(d[key]) : ''
}

/** neg-mode a blank picker resolves to, per `resolve_gen_defaults`. */
const NEG_AUTO: Record<string, string> = { sd15: 'text', sdxl: 'mean' }

const NEG_LABELS: Record<string, string> = {
  text: 'text (house negative prompt)',
  mean: 'mean (corpus mean)',
  empty: 'empty (empty prompt)',
  zeros: 'zeros (zero tensor)',
}

/* ----------------------------------------------------------- the schema  */

export const PARAM_SCHEMA: ParamField[] = [
  {
    id: 'action',
    type: 'segmented',
    group: 'action',
    default: 'generate',
    span: 2,
    options: [
      { value: 'generate', label: 'Generate', title: 'sample conditioning and render images' },
      { value: 'temp_sweep', label: 'Temp sweep', title: 'contact sheet across temperatures' },
      { value: 'sampler_sweep', label: 'Sampler sweep', title: 'contact sheet across samplers' },
      { value: 'mine', label: 'Mine', title: '(re)build this backend’s distribution from the prompt corpus' },
    ],
  },
  {
    // Experiment identity: the string that turns "I ran a batch" into "I
    // measured a cell". It reaches every .json sidecar, every label record and
    // outputs/experiments/<id>.json, so one id is enough to reconstruct and
    // re-judge a batch months later.
    id: 'experiment',
    type: 'text',
    group: 'action',
    label: 'Experiment id',
    placeholder: 'E01-length (blank = untagged)',
    default: '',
    emptyAsNull: true,
    span: 2,
    when: { action: GEN_ONLY },
    tooltip:
      'tags every image of this batch. Non-alphanumerics are slugged, so ' +
      '“E07 · negatives” becomes “E07-negatives”.',
  },
  {
    id: 'hypothesis',
    type: 'text',
    group: 'action',
    label: 'Hypothesis',
    placeholder: 'one sentence, falsifiable by eye',
    default: '',
    emptyAsNull: true,
    span: 2,
    when: { action: GEN_ONLY },
    tooltip:
      'stored once in the experiment manifest. A later batch with the same id ' +
      'and no hypothesis does not erase it.',
    hint: ({ values }) =>
      values.experiment?.trim()
        ? null
        : { text: 'needs an experiment id to be recorded', tone: 'dim' },
  },
  {
    id: 'backend',
    type: 'segmented',
    group: 'model',
    default: 'sd15',
    span: 2,
    options: [
      { value: 'sd15', label: 'SD 1.5' },
      { value: 'sd2', label: 'SD 2.1' },
      { value: 'sdxl', label: 'SDXL' },
      { value: 'flux2', label: 'FLUX.2' },
      { value: 'krea2', label: 'Krea 2' },
    ],
  },
  {
    id: 'model',
    type: 'select',
    group: 'model',
    label: 'SDXL checkpoint',
    default: 'sdxl-base-1.0',
    span: 2,
    when: { backend: ['sdxl'] },
    options: [
      { value: 'sdxl-base-1.0', label: 'sdxl-base-1.0 (30 steps, CFG) — recommended' },
      { value: 'sdxl-turbo', label: 'sdxl-turbo (1 step, no CFG — fast preview, generic)' },
    ],
    hint: ({ config }) =>
      config?.picked_models?.sdxl
        ? { text: 'ignored — a checkpoint is hand-picked below', tone: 'warn' }
        : null,
  },
  {
    // The legacy #ckptHint line: what --ckpt/--model the backend will resolve to.
    id: 'ckpt_note',
    type: 'note',
    group: 'model',
    span: 2,
    hint: ({ values, config }) => {
      const b = values.backend
      // A checkpoint hand-picked in the ModelPicker replaces whatever the env
      // vars point at, so the "missing checkpoint" warnings no longer apply.
      if (config?.picked_models?.[b as keyof typeof config.picked_models])
        return { text: 'using the checkpoint picked below' }
      if (b === 'sd15') {
        return config?.sd15_ckpt && !config.sd15_ckpt_exists
          ? { text: `SD1.5 checkpoint missing: ${config.sd15_ckpt}`, tone: 'warn' }
          : { text: 'single-file ckpt → --ckpt (512²)' }
      }
      if (b === 'sd2') {
        return config?.sd2_ckpt && !config.sd2_ckpt_exists
          ? { text: `SD2.1 checkpoint missing: ${config.sd2_ckpt}`, tone: 'warn' }
          : { text: 'single-file 768 v-pred ckpt → --ckpt (768²)' }
      }
      if (b === 'flux2')
        return { text: 'FLUX.2 klein (flow model, Qwen3 encoder) — mine first, then generate' }
      if (b === 'krea2')
        return {
          text:
            'Krea 2 Raw — use sampler diagonal (T 1.0–1.3) or blend λ0.6–0.7; ' +
            'pure pca looks washed (256-comp mine). Slow.',
        }
      return { text: 'cached HF repo → --model (1024²)' }
    },
  },

  /* ------------------------------------------------------------- sampler */
  // Which distribution to sample is NOT a form value — it's server-side state
  // (config.json -> "dists", per backend) driven by features/distribution/
  // DistPicker, which App slots under the Model group right below the
  // checkpoint picker (the two pick a matched pair). Nothing here sends
  // `dist`, so /api/run falls through to that persisted choice.
  {
    id: 'sampler',
    type: 'select',
    group: 'sampler',
    default: 'diagonal',
    span: 2,
    when: { action: GEN_OR_SWEEP },
    options: [
      { value: 'diagonal', label: 'diagonal — independent coords (raw)' },
      { value: 'pca', label: 'pca — on the corpus manifold (T>1 extrapolates)' },
      { value: 'blend', label: 'blend — interpolate diagonal/pca' },
      { value: 'hybrid', label: 'hybrid — SLERP two real concepts' },
      {
        value: 'split',
        label: 'split — separate on/off-manifold temperatures',
        title:
          'a diagonal draw cut into its PCA-subspace projection and the orthogonal ' +
          'remainder, each with its own temperature',
      },
    ],
    hint: ({ values }) =>
      values.action === 'sampler_sweep'
        ? { text: 'sampler-sweep renders every sampler itself — this picker is ignored.' }
        : null,
  },
  {
    id: 'coherence',
    type: 'number',
    group: 'sampler',
    label: 'Coherence λ',
    step: 0.1,
    placeholder: '0.5',
    default: '',
    cast: 'number',
    // Also live in sampler_sweep, which labels its middle row `blend λ=…` —
    // the one sampler flag that action forwards, and it used to be hidden there.
    visible: (v) =>
      v.action === 'sampler_sweep' || (usesSamplerKnobs(v) && v.sampler === 'blend'),
    tooltip:
      'blends the two COVARIANCES: Cov = λ·Cov_pca + (1-λ)·Cov_diagonal. ' +
      'λ=1 is bit-exactly the pca sampler, λ=0 bit-exactly the diagonal one.',
    hint: ({ values }) => {
      if (values.action === 'sampler_sweep')
        return {
          text: 'labels the sweep’s middle row; its other two rows are fixed diagonal/pca.',
          tone: 'dim',
        }
      const raw = (values.coherence ?? '').trim()
      if (!raw) return null
      const lam = Number(raw)
      if (lam >= 1)
        return { text: 'λ=1 is pure pca — the diagonal half, and ρ with it, drops out.', tone: 'warn' }
      if (lam <= 0)
        return { text: 'λ=0 is pure diagonal — the PCA half drops out.', tone: 'warn' }
      return null
    },
  },
  /*
   * The corpus-autopsy corrections. Each is a measured mismatch between the
   * fitted model and the corpus (distribution_report.html), shipped as a knob
   * beside the old behaviour rather than as a replacement — the "broken"
   * settings are textures, not bugs to be removed. Defaults reproduce the
   * original samplers exactly.
   */
  {
    id: 'rho',
    type: 'number',
    group: 'sampler',
    label: 'Row coherence ρ',
    step: 0.1,
    min: 0,
    placeholder: '0 (independent rows)',
    default: '',
    cast: 'number',
    span: 2,
    // ρ shapes the DIAGONAL draw only. pca has no diagonal half at all, and
    // hybrid returns real corpus embeddings before ρ is ever consulted.
    visible: (v) =>
      usesSamplerKnobs(v) && ['diagonal', 'blend', 'split'].includes(v.sampler),
    tooltip:
      'how much of each deviation is SHARED across the 77 token positions. The ' +
      'corpus sits at ~0.65; the diagonal sampler has always sat at 0.00, i.e. ' +
      '77 unrelated prompt-summaries stapled into one tensor. Per-coordinate ' +
      'marginals are identical at every ρ — only the agreement between rows moves. ' +
      'Independent of temperature: ρ turns the deviation, T scales it.',
    hint: ({ values }) => {
      const v = Number(values.rho)
      // blend at λ=1 IS the pca sampler, which never draws a diagonal half.
      if (v && values.sampler === 'blend' && Number(values.coherence) >= 1)
        return { text: 'inactive: λ=1 is pure pca, and ρ only shapes the diagonal half.', tone: 'warn' }
      if (!v) return { text: 'ρ=0 is the historical static — a texture, not a bug.', tone: 'dim' }
      if (v >= 0.95) return { text: 'ρ→1: one deviation smeared through the whole sentence.' }
      const near = v > 0.55 && v < 0.75 ? ' — corpus-like (~0.65)' : ''
      const half = values.sampler === 'blend' ? ' (the diagonal half only)'
        : values.sampler === 'split' ? ' (before the on/off split)' : ''
      return { text: `ρ=${v}${near}${half}` }
    },
  },
  {
    id: 'temp_on',
    type: 'number',
    group: 'sampler',
    label: 'T on-manifold',
    step: 0.25,
    placeholder: '1.0',
    default: '',
    cast: 'number',
    visible: (v) => usesSamplerKnobs(v) && v.sampler === 'split',
    tooltip:
      'temperature of the half of the deviation that lies IN the corpus PCA ' +
      'subspace — the part a real prompt could have produced. In split mode ' +
      'this and T off-manifold ARE the temperature; the global one is not shown ' +
      'because it multiplies both and would only duplicate them.',
  },
  {
    id: 'temp_off',
    type: 'number',
    group: 'sampler',
    label: 'T off-manifold',
    step: 0.25,
    placeholder: '1.0',
    default: '',
    cast: 'number',
    visible: (v) => usesSamplerKnobs(v) && v.sampler === 'split',
    tooltip:
      'temperature of the half orthogonal to the corpus subspace — where no ' +
      'prompt ever goes. 1,1 is bit-exactly the diagonal sampler; on=1,off=0 ' +
      'collapses the draw onto the manifold; on=0,off=1 is pure off-manifold noise.',
  },
  {
    // The split sampler's own read-out: three numbers used to imply an
    // effective pair, and reading `T × temp_on` off the sidebar was guesswork.
    id: 'split_note',
    type: 'note',
    group: 'sampler',
    span: 2,
    visible: (v) => usesSamplerKnobs(v) && v.sampler === 'split',
    hint: ({ values }) => {
      const on = (values.temp_on ?? '').trim() === '' ? 1 : Number(values.temp_on)
      const off = (values.temp_off ?? '').trim() === '' ? 1 : Number(values.temp_off)
      if (values.action === 'temp_sweep')
        return { text: `each swept temperature multiplies both: T·${on} on-manifold, T·${off} off.`, tone: 'dim' }
      if (reachMode(values) !== 'temperature')
        return {
          text: `radius is pinned below, so only the RATIO matters here — ${on}:${off} on/off-manifold.`,
          tone: 'dim',
        }
      if (on === off)
        return { text: `on=off=${on}: identical to the diagonal sampler at temperature ${on}.`, tone: 'dim' }
      return { text: `deviation = ${on}× on-manifold + ${off}× off-manifold.`, tone: 'dim' }
    },
  },
  {
    id: 'length_mode',
    type: 'select',
    group: 'sampler',
    label: 'Prompt length',
    default: 'corpus',
    // 'corpus' is an active default, so hiding the knob must fall back to the
    // inert 'off' rather than to the default (hybrid would otherwise carry a
    // length mode it never reads into argv and into the sidecar).
    hidden: 'off',
    span: 2,
    // hybrid returns SLERPed corpus embeddings before length conditioning is
    // ever consulted, so the mode would be a no-op there.
    visible: (v) => usesSamplerKnobs(v) && v.sampler !== 'hybrid',
    tooltip:
      'CLIP pads to 77 with EOS, so every middle position is really two ' +
      'populations (content vs padding) and the single fitted Gaussian peaks in ' +
      'the gap between them. Conditioning on a length draws from one lobe or the ' +
      'other. Needs a distribution mined since the length split existed.',
    options: [
      { value: 'off', label: 'off — one pooled Gaussian per position' },
      { value: 'corpus', label: 'corpus — draw each length from the histogram' },
      { value: 'fixed', label: 'fixed — pin every sample to one length' },
    ],
    hint: ({ values }) =>
      values.length_mode === 'off'
        ? null
        : {
            text: 'Prompt length was PC1/PC2 in disguise (|r| 0.69/0.64) — the corpus’s biggest semantic dial.',
            tone: 'dim',
          },
  },
  {
    id: 'length',
    type: 'number',
    group: 'sampler',
    label: 'Length (tokens)',
    placeholder: '30',
    default: '30',
    cast: 'number',
    min: 1,
    span: 2,
    when: { length_mode: ['fixed'] },
    visible: (v) => usesSamplerKnobs(v) && v.sampler !== 'hybrid',
    tooltip: '“sample me a 60-token image”. 1–77; the corpus median is ~23.',
  },
  {
    id: 'scheduler',
    type: 'select',
    group: 'sampler',
    label: 'Sampler / scheduler',
    default: 'default',
    span: 2,
    when: { action: GEN_OR_SWEEP },
    options: [
      { value: 'default', label: 'default scheduler' },
      { value: 'ddim', label: 'DDIM (smooth, for high-step renders)' },
      { value: 'euler', label: 'Euler' },
      { value: 'euler_a', label: 'Euler ancestral' },
      { value: 'dpm', label: 'DPM++ 2M' },
    ],
  },

  /* --------------------------------------------------------------- reach */
  /*
   * One quantity, three spellings, and they do NOT compose: `retarget()`
   * rescales a sample to an exact distance, which divides the temperature
   * factor straight back out. Setting a temperature AND a target distance used
   * to look like two controls cooperating; it was one control plus a dead box.
   * So the sidebar picks a mode, and only that mode's knob is shown — and,
   * because `buildRunRequest` sends a hidden field at its default, only that
   * mode's flag can reach argv.
   */
  {
    id: 'reach',
    type: 'segmented',
    group: 'reach',
    span: 2,
    default: 'temperature',
    send: false, // pure UI: it selects WHICH reach flag build_argv receives
    when: { action: GEN_ONLY },
    options: [
      {
        value: 'temperature',
        label: 'Temperature',
        title:
          'No radius pinning. Scale the whole deviation and let the draw land where it lands ' +
          '(a diagonal sample ends up at distance ≈ T).',
      },
      {
        value: 'shell',
        label: 'Shell',
        title:
          'ONE radius, typed by you. Every image in the batch is rescaled onto that exact ' +
          'distance — same sphere, different directions. No radius variety at all.',
      },
      {
        value: 'band',
        label: 'Corpus band',
        title:
          'Same rescaling, but a DIFFERENT radius per image, drawn at random from the ' +
          'distances real corpus prompts actually have. You don’t type a radius — the ' +
          'corpus supplies them, and they vary across the batch.',
      },
    ],
  },
  {
    id: 'reach_note',
    type: 'note',
    group: 'reach',
    span: 2,
    when: { action: GEN_ONLY },
    hint: ({ values }) => {
      const m = reachMode(values)
      if (m === 'temperature')
        return { text: 'Distance from the corpus centre is left to the draw itself.', tone: 'dim' }
      // hybrid's T is a jitter weight, not a scale factor, so it is the one
      // sampler a pinned radius does not cancel — the note must not claim it does.
      const tail =
        values.sampler === 'hybrid'
          ? 'Temperature stays live because hybrid uses it as jitter, not as reach.'
          : 'Temperature would cancel out exactly, so it is not sent.'
      // Shell and band are the SAME operation (retarget: keep the direction,
      // overwrite the radius) and differ only in where the radius comes from —
      // one number you type vs. one resampled per image from the corpus. Say
      // that out loud in both branches, or the two modes look interchangeable.
      return {
        text:
          m === 'shell'
            ? `Shell: one radius for the whole batch, the number you type — every image sits ` +
              `on the same sphere. Switch to Corpus band to get a different radius per image ` +
              `instead. ${tail}`
            : `Corpus band: same rescaling as Shell, but the radius is re-drawn per image from ` +
              `the corpus’s own distances, so the batch spreads over a range instead of one ` +
              `sphere. ${tail} Needs a fit mined with the radius band.`,
        tone: 'dim',
      }
    },
  },
  {
    id: 'temperature',
    type: 'number',
    group: 'reach',
    label: 'Temperature',
    step: 0.1,
    placeholder: '1.0',
    default: '',
    cast: 'number',
    span: 2,
    // Hidden — and therefore not sent — wherever it cannot mean what it says:
    // temp_sweep owns it via --temps, split expresses it as temp_on/temp_off,
    // and a pinned radius divides it back out. hybrid is the exception: there T
    // is a jitter weight, which survives a shell.
    visible: (v) =>
      GEN_OR_SWEEP.includes(v.action) &&
      v.action !== 'temp_sweep' &&
      !(usesSamplerKnobs(v) && v.sampler === 'split') &&
      (reachMode(v) === 'temperature' || v.sampler === 'hybrid'),
    tooltip: ({ values }) =>
      values.sampler === 'hybrid'
        ? 'hybrid does not scale its reach with T at all: it SLERPs two REAL ' +
          'corpus embeddings, so a sample sits at corpus distance (~1.0) whatever ' +
          'T says. T scales only the gaussian jitter added on top.'
        : 'how wide to draw, as a multiple of the corpus spread. Scales the whole ' +
          'deviation from the corpus centre, so a diagonal sample lands at ' +
          'distance ≈ T: 1.0 = as far out as a real prompt, >1 pushes outward into ' +
          'wilder / less legible territory, <1 hugs the bland centre.',
    hint: ({ values }) => {
      if (values.sampler === 'hybrid' && usesSamplerKnobs(values))
        return {
          text: 'hybrid: T is jitter, not reach — distance stays ≈1.0 whatever you set.',
          tone: 'warn',
        }
      if (values.action === 'sampler_sweep')
        return { text: 'one fixed T for all three rows of the sheet.', tone: 'dim' }
      if (['pca', 'blend'].includes(values.sampler))
        return {
          text: 'pca draws inside the retained subspace only, so the distance gauge lands below T.',
          tone: 'dim',
        }
      return null
    },
  },
  {
    id: 'target_distance',
    type: 'number',
    group: 'reach',
    label: 'Target distance',
    step: 0.1,
    placeholder: '1.0',
    default: '',
    cast: 'number',
    span: 2,
    // Only generate.py defines --target-distance; the sweep scripts do not take
    // it, and build_argv never forwarded it there — the box just looked live.
    when: { action: GEN_ONLY },
    visible: (v) => reachMode(v) === 'shell',
    tooltip:
      'shell sampling: after drawing, every sample is rescaled so its distance gauge ' +
      'lands exactly here. The sampler still chooses the direction, this pins the radius ' +
      '— which is why a temperature would cancel and is not sent. One number for the ' +
      'whole batch, so the batch has no radius spread: use this to hold the radius fixed ' +
      'while you vary something else, or to re-shoot the exact distance your ★ keepers ' +
      'live at. For a natural spread instead, use Corpus band.',
    hint: ({ values, tasteband }) => {
      if (usesSamplerKnobs(values) && values.sampler === 'split')
        return {
          text: 'split + shell: only the temp_on : temp_off RATIO changes anything — the radius is already fixed.',
          tone: 'dim',
        }
      if (values.sampler === 'hybrid')
        return {
          text: 'hybrid + shell: T still tilts the jitter/SLERP mix, so it stays live above.',
          tone: 'dim',
        }
      return tasteband?.count
        ? {
            text: `your ★ keepers: d≈${tasteband.mean} (band ${tasteband.p25}–${tasteband.p75}, n=${tasteband.count}) — try that as target`,
          }
        : null
    },
  },
  {
    id: 'radius_scale',
    type: 'number',
    group: 'reach',
    label: 'Band scale',
    step: 0.1,
    placeholder: '1.0',
    default: '',
    cast: 'number',
    span: 2,
    when: { action: GEN_ONLY },
    visible: (v) => reachMode(v) === 'band',
    tooltip:
      'Corpus band = shell sampling with a radius drawn per image instead of typed once: ' +
      'each sample gets its own target, picked at random (with replacement) from the list ' +
      'of distances the real corpus prompts sit at, then rescaled onto it exactly as Shell ' +
      'does. The point is the spread — a real corpus occupies a band of radii (sd15: ≈0.89–' +
      '1.11 around 0.99) while every sampler left alone collapses to a ~9× tighter spike ' +
      'around one value. This knob shifts that whole band: 1.0 = the corpus’s own radii, ' +
      '>1 = the same shape of spread, further out.',
  },
  {
    id: 'min_distance',
    type: 'number',
    group: 'reach',
    label: 'Min distance',
    step: 0.1,
    placeholder: 'off',
    default: '',
    cast: 'number',
    span: 2,
    when: { action: GEN_ONLY },
    tooltip:
      'floor, applied LAST: any sample still closer to the corpus centre than ' +
      'this is scaled up onto it. Meant to keep the bland core out of a ' +
      'temperature draw — it runs after a shell or a band too, so a floor above ' +
      'them silently replaces them.',
    hint: ({ values }) => {
      const floor = Number((values.min_distance ?? '').trim())
      if (!floor) return null
      const m = reachMode(values)
      if (m === 'shell') {
        const t = Number((values.target_distance ?? '').trim())
        return t && floor >= t
          ? {
              text: `floor ≥ shell: every sample is pushed to ${floor}, so the ${t} shell does nothing.`,
              tone: 'warn',
            }
          : { text: 'below the shell — no sample can reach it, so this is inert.', tone: 'dim' }
      }
      if (m === 'band')
        return { text: 'clips the bottom of the corpus band; the top is untouched.', tone: 'dim' }
      return null
    },
  },

  /* --------------------------------------------------------------- image */
  {
    id: 'n',
    type: 'number',
    group: 'image',
    label: 'Images (n)',
    placeholder: '8',
    default: '',
    cast: 'number',
    when: { action: GEN_ONLY },
  },
  {
    id: 'seed',
    type: 'number',
    group: 'image',
    label: 'Seed',
    placeholder: 'random',
    default: '',
    cast: 'number',
    when: { action: GEN_ONLY },
    tooltip:
      'image i is rendered with seed + i, so “seed 1000, n 16” IS the fixed ' +
      'seed panel — the same 16 noise draws every comparative batch uses.',
    // Comparative batches (A/B on one variable) must be PAIRED, or the noise
    // difference swamps the idea. The panel is one flag pair, so the sidebar
    // says so rather than leaving it to memory.
    hint: ({ values, config }) => {
      const panel = config?.seed_panel
      if (!panel) return null
      const onPanel =
        Number(values.seed) === panel.seed && Number(values.n || 0) >= panel.n
      return onPanel
        ? { text: `⧉ seed panel: image seeds ${panel.seed}–${panel.seed + panel.n - 1}` }
        : {
            text: `comparative batch? use seed ${panel.seed} + n ${panel.n} (the fixed panel)`,
            tone: 'dim',
          }
    },
  },
  {
    id: 'steps',
    type: 'number',
    group: 'image',
    label: 'Steps',
    default: '',
    cast: 'number',
    when: { action: GEN_ONLY },
    seedFromPlaceholder: true,
    placeholder: ({ values }) => genDefault(values, 'steps'),
    tooltip: 'denoising steps. Blank = the shown default; click to edit it.',
  },
  {
    id: 'guidance',
    type: 'number',
    group: 'image',
    label: 'Guidance',
    step: 0.5,
    default: '',
    cast: 'number',
    when: { action: GEN_ONLY },
    seedFromPlaceholder: true,
    placeholder: ({ values }) => genDefault(values, 'guidance'),
    tooltip: 'CFG scale. Blank = the shown default; click to edit it.',
  },
  {
    id: 'aspect',
    type: 'select',
    group: 'image',
    label: 'Aspect ratio',
    default: '',
    span: 2,
    send: false, // client-only: it just fills width/height for the backend
    when: { action: GEN_ONLY },
    options: [
      { value: '', label: 'default (square)' },
      { value: '1:1', label: 'square 1:1' },
      { value: '3:2', label: 'landscape 3:2' },
      { value: '2:3', label: 'portrait 2:3' },
      { value: '4:3', label: 'landscape 4:3' },
      { value: '3:4', label: 'portrait 3:4' },
      { value: '16:9', label: 'wide 16:9' },
      { value: '9:16', label: 'tall 9:16' },
      { value: '21:9', label: 'cinematic 21:9' },
    ],
    hint: ({ values }) => {
      const d = aspectDims(values)
      if (!d.width || !d.height) return null
      return {
        text: `${d.width}×${d.height} (${values.backend} native ${NATIVE_RES[values.backend] ?? 1024})`,
      }
    },
  },
  {
    id: 'width',
    type: 'number',
    group: 'image',
    label: 'Width',
    placeholder: ({ values }) => String(NATIVE_RES[values.backend] ?? 1024),
    tooltip: 'the backend’s native width unless set. Accepts arithmetic: 1024+256 ⏎ → 1280',
    default: '',
    cast: 'number',
    expr: true,
    seedFromPlaceholder: true,
    when: { action: GEN_ONLY },
  },
  {
    id: 'height',
    type: 'number',
    group: 'image',
    label: 'Height',
    placeholder: ({ values }) => String(NATIVE_RES[values.backend] ?? 1024),
    tooltip: 'the backend’s native height unless set. Accepts arithmetic: 1024+256 ⏎ → 1280',
    default: '',
    cast: 'number',
    expr: true,
    seedFromPlaceholder: true,
    when: { action: GEN_ONLY },
  },
  {
    id: 'init_folder',
    type: 'select',
    group: 'image',
    label: 'Init folder',
    tooltip: 'start from a random good init image',
    default: 'off',
    when: { action: GEN_ONLY },
    options: ({ config }) => {
      const opts: Option[] = [{ value: 'off', label: 'off' }]
      const folders = config?.init_folders ?? []
      if (folders.length)
        opts.push({ value: '__any__', label: `any folder (${config?.init_count ?? 0})` })
      for (const f of folders)
        opts.push({ value: f.path, label: `${f.name} (${f.count})` })
      return opts
    },
  },
  {
    id: 'init_mode',
    type: 'select',
    group: 'image',
    label: 'Init mode',
    default: 'img2img',
    when: { action: GEN_ONLY },
    options: [
      { value: 'img2img', label: 'img2img (structure)' },
      { value: 'embedding', label: 'image-embedding (content)' },
    ],
  },
  {
    id: 'init_strength',
    type: 'number',
    group: 'image',
    label: 'Strength / scale',
    step: 0.05,
    placeholder: '0.7',
    default: '',
    cast: 'number',
    tooltip: 'img2img denoise strength, or IP-Adapter scale in embedding mode',
    when: { action: GEN_ONLY },
  },
  {
    id: 'init_note',
    type: 'note',
    group: 'image',
    span: 2,
    when: { action: GEN_ONLY },
    hint: ({ config }) =>
      config && config.init_count > 0
        ? {
            text: `${config.init_count} init image(s) across ${config.init_folders.length} folder(s) in init_images/`,
          }
        : {
            text: `no init images yet — drop folders/images in ${config?.init_dir ?? 'init_images/'}`,
            tone: 'warn',
          },
  },

  /* --------------------------------------------------------------- sweep */
  {
    id: 'temps',
    type: 'text',
    group: 'sweep',
    label: 'Temperatures (csv)',
    placeholder: '0.5,1.0,1.5,2.0',
    default: '',
    emptyAsNull: true,
    span: 2,
    when: { action: ['temp_sweep'] },
  },
  {
    id: 'seeds',
    type: 'text',
    group: 'sweep',
    label: 'Seeds (csv)',
    placeholder: '0,1,2',
    default: '',
    emptyAsNull: true,
    span: 2,
    when: { action: ['temp_sweep', 'sampler_sweep'] },
  },

  /* ------------------------------------------------------------ advanced */
  {
    // Two different meanings on one flag: the PCA rank to FIT when mining, the
    // axis count to SAMPLE otherwise. Both are real, so the label says which.
    id: 'components',
    type: 'number',
    group: 'advanced',
    label: ({ values }) => (values.action === 'mine' ? 'Components (PCA rank)' : 'Components'),
    placeholder: ({ values }) => (values.action === 'mine' ? '512' : 'all fitted axes'),
    default: '',
    cast: 'number',
    visible: (v) =>
      v.action === 'mine' ||
      (usesSamplerKnobs(v) && ['pca', 'blend', 'split'].includes(v.sampler)),
    tooltip: ({ values }) =>
      values.action === 'mine'
        ? 'how many principal axes the fit KEEPS. Each costs a full feature row ' +
          'on disk and past ~400 the spectrum is noise.'
        : 'use N principal axes starting at comp-lo. On split this sets which ' +
          'axes count as “on-manifold”, so a low N widens the off-manifold half.',
  },
  {
    id: 'truncation',
    type: 'number',
    group: 'advanced',
    label: 'Truncation σ',
    step: 0.5,
    placeholder: 'off',
    default: '',
    cast: 'number',
    // hybrid never draws gaussian coefficients to truncate.
    visible: (v) => usesSamplerKnobs(v) && v.sampler !== 'hybrid',
    tooltip:
      'resample coords beyond this many sigma (typical-set trick). Shapes the ' +
      'draw rather than scaling it, so unlike temperature it still bites under ' +
      'a shell or a band. Off by default and meant to stay there — the real ' +
      'corpus has 5–8 sigma events.',
  },
  {
    id: 'comp_lo',
    type: 'number',
    group: 'advanced',
    label: 'Comp-lo (weird axis)',
    placeholder: '0',
    default: '',
    cast: 'number',
    visible: (v) => usesSamplerKnobs(v) && ['pca', 'blend', 'split'].includes(v.sampler),
    tooltip: 'skip the dominant/standard PCA axes; higher = stranger subjects',
  },
  {
    id: 'equalize',
    type: 'select',
    group: 'advanced',
    label: 'Equalize',
    default: '',
    cast: 'bool',
    // Lives in _pca_dev only: split projects a diagonal draw and never rescales
    // per axis, so equalize would be a no-op there.
    visible: (v) => usesSamplerKnobs(v) && ['pca', 'blend'].includes(v.sampler),
    tooltip: 'express every selected axis at equal RMS strength',
    options: [
      { value: '', label: 'off' },
      { value: '1', label: 'on (express minor axes)' },
    ],
  },
  {
    id: 'empirical_head',
    type: 'number',
    group: 'advanced',
    label: 'Empirical PCA head',
    placeholder: '0 (all Gaussian)',
    default: '',
    cast: 'number',
    min: 0,
    visible: (v) => usesSamplerKnobs(v) && ['pca', 'blend'].includes(v.sampler),
    tooltip:
      'draw the first K principal coefficients from the corpus’s own CDF instead ' +
      'of N(0,1). PC1 is two lobes, so a Gaussian puts its densest mass in a gap ' +
      'no prompt occupies. K=2 covers both length-correlated axes.',
    hint: ({ values }) => {
      const k = Number((values.empirical_head ?? '').trim())
      const lo = Number((values.comp_lo ?? '').trim())
      return k && lo >= k
        ? { text: `comp-lo ${lo} skips past all K=${k} axes, so this does nothing.`, tone: 'warn' }
        : null
    },
  },
  {
    id: 'advanced_note',
    type: 'note',
    group: 'advanced',
    span: 2,
    visible: (v) => v.action !== 'mine',
    hint: {
      text: 'For non-standard subjects: sampler pca, comp-lo ~40–200, equalize on, temp ~1.1–1.4.',
    },
  },
  {
    id: 'neg_mode',
    type: 'select',
    group: 'advanced',
    label: 'neg-mode',
    default: 'mean',
    // Hidden (the non-image actions) means "whatever the backend does by
    // itself" — the blank entry — not this default.
    hidden: '',
    emptyAsNull: true,
    span: 2,
    when: { action: GEN_OR_SWEEP },
    tooltip:
      'what CFG pushes away from. The first entry is this backend’s own ' +
      'default and names which mode that is (text on sd15, mean on sdxl, ' +
      'empty elsewhere).',
    // The blank entry spells out which mode it resolves to for the current
    // backend, so the picker never just says "auto".
    options: ({ values }) => {
      const auto = NEG_AUTO[values.backend] ?? 'empty'
      return [
        { value: '', label: `${NEG_LABELS[auto]} — ${values.backend} default` },
        ...(['text', 'mean', 'empty', 'zeros'] as const).map((m) => ({
          value: m,
          label: NEG_LABELS[m],
        })),
      ]
    },
  },
  {
    // The one human-written string a "promptless" run still uses, so it is
    // shown in full rather than named. Hidden for the backends whose negative
    // is a tensor (sdxl) or absent (flux2/krea2), and for the tensor neg-modes.
    id: 'negative',
    type: 'textarea',
    group: 'advanced',
    label: 'negative prompt',
    default: '',
    rows: 4,
    emptyAsNull: true,
    seedFromPlaceholder: true,
    span: 2,
    when: { backend: ['sd15', 'sd2'], neg_mode: ['', 'text'], action: GEN_OR_SWEEP },
    // sd2 has no house negative of its own, so nothing to prefill there.
    placeholder: (c) => (c.values.backend === 'sd2' ? '' : (c.config?.sd15_negative ?? '')),
    tooltip: 'The CFG negative branch, in words. Blank = the house SD1.5 negative.',
    hint: (c) =>
      c.values.backend === 'sd2'
        ? { text: 'sd2 defaults to no negative text — set neg-mode to text to use this.', tone: 'warn' }
        : { text: 'Click to edit. Clear the box to restore the default.', tone: 'dim' },
  },
]

/* -------------------------------------------------------------- helpers  */

export const DEFAULT_VALUES: Values = Object.fromEntries(
  PARAM_SCHEMA.filter((f) => f.type !== 'note').map((f) => [f.id, f.default ?? '']),
)

export function resolve<T>(v: Dyn<T> | undefined, ctx: Ctx): T | undefined {
  return typeof v === 'function' ? (v as (c: Ctx) => T)(ctx) : v
}

/** Every key in `when` must match one of its listed values, and `visible` pass. */
export function isVisible(f: ParamField, values: Values): boolean {
  if (f.when && !Object.entries(f.when).every(([k, allowed]) => allowed.includes(values[k])))
    return false
  return f.visible ? f.visible(values) : true
}

export function visibleFields(values: Values): ParamField[] {
  return PARAM_SCHEMA.filter((f) => isVisible(f, values))
}

/**
 * Aspect preset -> width/height at the backend's native pixel budget, rounded
 * to multiples of 64 (identical maths to the legacy `applyAspect`).
 */
export function aspectDims(values: Values): { width: string; height: string } {
  const v = values.aspect
  if (!v) return { width: '', height: '' }
  const base = NATIVE_RES[values.backend] ?? 1024
  const area = base * base
  const [rw, rh] = v.split(':').map(Number)
  if (!rw || !rh) return { width: '', height: '' }
  const r = rw / rh
  const round64 = (x: number) => Math.max(64, Math.round(x / 64) * 64)
  return {
    width: String(round64(Math.sqrt(area * r))),
    height: String(round64(Math.sqrt(area / r))),
  }
}

function castValue(f: ParamField, raw: string): unknown {
  const v = (raw ?? '').trim()
  if (f.cast === 'number') {
    if (v === '') return null
    const n = Number(v)
    // An expr box submitted without ever losing focus still holds "1024+256".
    if (Number.isNaN(n) && f.expr) {
      const e = evalExpr(v)
      return e === null ? null : Number.isInteger(f.step ?? 1) ? Math.round(e) : e
    }
    return n
  }
  if (f.cast === 'bool') return v === '1'
  return f.emptyAsNull && v === '' ? null : v
}

/**
 * Form state -> the /api/run body. Generic over the schema; the short tail
 * covers the two derived knobs the legacy JS also computed by hand (the init
 * toggle, and init_strength doubling as ip_scale in embedding mode).
 */
export function buildRunRequest(values: Values): RunRequest {
  const out: Record<string, unknown> = {}
  for (const f of PARAM_SCHEMA) {
    if (f.type === 'note' || f.send === false) continue
    // A knob the sidebar is hiding is sent at its inert value (`hidden`, or the
    // default when that value IS the default), not at whatever the form still
    // remembers. The `when`/`visible` predicates encode which flags
    // each action and sampler actually read, so this is what makes them binding
    // rather than decorative: a temperature typed before switching to a shell,
    // or a ρ left over from a diagonal run, can no longer reach argv and turn
    // up in the sidecar of an image it had no effect on. Falling back to the
    // default (rather than null) also keeps the non-optional server fields —
    // `sampler`, `init_mode` — valid when their control is hidden.
    const raw = isVisible(f, values) ? (values[f.id] ?? '') : (f.hidden ?? f.default ?? '')
    out[f.id] = castValue(f, raw)
  }
  const folder = values.init_folder ?? 'off'
  const strength =
    (values.init_strength ?? '').trim() === '' ? 0.7 : Number(values.init_strength)
  out.init = folder !== 'off'
  out.init_folder = folder === 'off' ? null : folder
  out.init_strength = strength
  out.ip_scale = strength
  // The corpus radius band is no longer its own toggle — it IS a reach mode, so
  // the flag is derived from the picker instead of being a second way to say it.
  out.radius_band = reachMode(values) === 'band'
  return out as unknown as RunRequest
}

/** Label for the submit button. */
export function runButtonLabel(values: Values): string {
  return values.action === 'mine' ? 'Mine ▶' : 'Run ▶'
}
