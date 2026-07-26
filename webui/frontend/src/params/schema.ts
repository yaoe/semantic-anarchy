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

export type FieldType = 'segmented' | 'select' | 'number' | 'text' | 'note'

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

export type GroupId = 'action' | 'model' | 'sampler' | 'image' | 'sweep' | 'advanced'

export interface ParamField {
  id: string
  type: FieldType
  group: GroupId
  label?: string
  /** Visibility predicate: every key must match one of its listed values. */
  when?: Record<string, string[]>
  options?: Dyn<Option[]>
  placeholder?: Dyn<string>
  step?: number
  min?: number
  tooltip?: string
  hint?: Dyn<Hint | string | null>
  /** Initial form value. '' means "left blank -> the script's own default". */
  default?: string
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
  { id: 'image' },
  { id: 'sweep' },
  { id: 'advanced', title: 'Advanced', collapsible: true },
]

/* -------------------------------------------------------------- constants */

/** Actions that expose the sampler block (legacy `.genonly` = gen || sweep). */
const GEN_OR_SWEEP = ['generate', 'temp_sweep', 'sampler_sweep']
/** Actions that render actual images with per-image knobs (legacy `.genimg`). */
const GEN_ONLY = ['generate']

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
  {
    id: 'dist',
    type: 'select',
    group: 'sampler',
    label: 'Distribution',
    tooltip: 'which learned distribution to sample from',
    default: 'base',
    span: 2,
    options: [
      { value: 'base', label: 'base corpus' },
      { value: 'evolved', label: 'evolved ★ branch (from 🧪)' },
    ],
  },
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
    ],
    hint: ({ values }) =>
      values.action === 'sampler_sweep'
        ? { text: 'sampler-sweep renders every sampler itself — this picker is ignored.' }
        : null,
  },
  {
    id: 'temperature',
    type: 'number',
    group: 'sampler',
    label: 'Temperature',
    step: 0.1,
    placeholder: '1.0',
    default: '',
    cast: 'number',
    tooltip:
      'how wide to draw, as a multiple of the corpus spread. Scales the whole ' +
      'deviation from the corpus center, so a sample typically lands at ' +
      'distance ≈ T: 1.0 = as far out as a real prompt, >1 pushes outward into ' +
      'wilder / less legible territory, <1 hugs the bland center. ' +
      'Ignored for reach when Target distance is set.',
    when: { action: GEN_OR_SWEEP },
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
    tooltip: 'blend lambda in [0,1]: 1 ≡ pure pca, 0 ≡ pure diagonal',
    when: { action: GEN_OR_SWEEP, sampler: ['blend'] },
  },
  {
    id: 'target_distance',
    type: 'number',
    group: 'sampler',
    label: 'Target distance',
    step: 0.1,
    placeholder: 'off',
    default: '',
    cast: 'number',
    span: 2,
    tooltip:
      'shell sampling: exact distance instead of a spread. After drawing, every ' +
      "sample is rescaled so its distance gauge lands exactly here — the sampler " +
      'still picks the direction, this pins the radius. Overrides how far ' +
      'Temperature reaches (T only shapes the draw, not where it ends up). ' +
      'Blank = off, temperature decides.',
    when: { action: GEN_OR_SWEEP },
    hint: ({ tasteband }) =>
      tasteband?.count
        ? {
            text: `your ★ keepers: d≈${tasteband.mean} (band ${tasteband.p25}–${tasteband.p75}, n=${tasteband.count}) — try that as target`,
          }
        : null,
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
  },
  {
    id: 'steps',
    type: 'number',
    group: 'image',
    label: 'Steps',
    default: '',
    cast: 'number',
    when: { action: GEN_ONLY },
    placeholder: ({ values }) =>
      values.backend === 'sdxl'
        ? `default ${SDXL_MODEL_DEFAULTS[values.model]?.steps ?? 30}`
        : 'auto (try 50)',
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
    placeholder: ({ values }) =>
      values.backend === 'sdxl'
        ? `default ${SDXL_MODEL_DEFAULTS[values.model]?.guidance ?? 7}`
        : 'auto',
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
    placeholder: 'auto',
    tooltip: 'accepts arithmetic: 1024+256 ⏎ → 1280',
    default: '',
    cast: 'number',
    expr: true,
    when: { action: GEN_ONLY },
  },
  {
    id: 'height',
    type: 'number',
    group: 'image',
    label: 'Height',
    placeholder: 'auto',
    tooltip: 'accepts arithmetic: 1024+256 ⏎ → 1280',
    default: '',
    cast: 'number',
    expr: true,
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
    id: 'components',
    type: 'number',
    group: 'advanced',
    label: 'Components',
    placeholder: 'all',
    default: '',
    cast: 'number',
    tooltip: 'pca/blend: use N principal axes starting at comp-lo (mine: PCA rank)',
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
    tooltip: 'resample coords beyond this many sigma (typical-set trick)',
  },
  {
    id: 'comp_lo',
    type: 'number',
    group: 'advanced',
    label: 'Comp-lo (weird axis)',
    placeholder: '0',
    default: '',
    cast: 'number',
    tooltip: 'skip the dominant/standard PCA axes; higher = stranger subjects',
  },
  {
    id: 'equalize',
    type: 'select',
    group: 'advanced',
    label: 'Equalize',
    default: '',
    cast: 'bool',
    tooltip: 'express every selected axis at equal RMS strength',
    options: [
      { value: '', label: 'off' },
      { value: '1', label: 'on (express minor axes)' },
    ],
  },
  {
    id: 'min_distance',
    type: 'number',
    group: 'advanced',
    label: 'Min distance',
    step: 0.1,
    placeholder: 'off',
    default: '',
    cast: 'number',
    tooltip: 'floor: never sample closer to the corpus centre than this',
    when: { action: GEN_ONLY },
  },
  {
    id: 'advanced_note',
    type: 'note',
    group: 'advanced',
    span: 2,
    hint: {
      text: 'For non-standard subjects: sampler pca, comp-lo ~40–200, equalize on, temp ~1.1–1.4.',
    },
  },
  {
    id: 'neg_mode',
    type: 'select',
    group: 'advanced',
    label: 'SDXL neg-mode',
    default: '',
    emptyAsNull: true,
    span: 2,
    tooltip: '(sdxl CFG) negative conditioning. mean = push away from the corpus average',
    options: [
      { value: '', label: 'auto' },
      { value: 'mean', label: 'mean' },
      { value: 'empty', label: 'empty' },
      { value: 'zeros', label: 'zeros' },
    ],
  },
]

/* -------------------------------------------------------------- helpers  */

export const DEFAULT_VALUES: Values = Object.fromEntries(
  PARAM_SCHEMA.filter((f) => f.type !== 'note').map((f) => [f.id, f.default ?? '']),
)

export function resolve<T>(v: Dyn<T> | undefined, ctx: Ctx): T | undefined {
  return typeof v === 'function' ? (v as (c: Ctx) => T)(ctx) : v
}

/** Every key in `when` must match one of its listed values. */
export function isVisible(f: ParamField, values: Values): boolean {
  if (!f.when) return true
  return Object.entries(f.when).every(([k, allowed]) => allowed.includes(values[k]))
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
    out[f.id] = castValue(f, values[f.id] ?? '')
  }
  const folder = values.init_folder ?? 'off'
  const strength =
    (values.init_strength ?? '').trim() === '' ? 0.7 : Number(values.init_strength)
  out.init = folder !== 'off'
  out.init_folder = folder === 'off' ? null : folder
  out.init_strength = strength
  out.ip_scale = strength
  return out as unknown as RunRequest
}

/** Label for the submit button. */
export function runButtonLabel(values: Values): string {
  return values.action === 'mine' ? 'Mine ▶' : 'Run ▶'
}
