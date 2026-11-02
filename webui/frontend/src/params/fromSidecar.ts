/**
 * An image's sidecar -> the sidebar values that would sample like it.
 *
 * The inverse of `buildRunRequest`, for the knobs a sidecar records. A knob it
 * does not record was not in play (older sidecars predate it), so it goes to its
 * inert value rather than keeping whatever the form held: "reset" has to mean no
 * leftover ρ quietly shaping the next batch.
 *
 * Deliberately left alone:
 *   seed        Cleared, so Run draws NEW images under these conditions. The
 *               recorded seed can't re-make this image anyway: length and sample
 *               draws share one rng sized by `n`, and `n` isn't recorded. (The
 *               .npz holds the exact tensor; explore/upscale use that.)
 *   n           How many you want now, not a condition of the image.
 *   experiment  A new batch doesn't belong to the old one's experiment by default.
 */
import { DEFAULT_VALUES, PARAM_SCHEMA, type Option, type Values } from './schema'

type Sidecar = Record<string, unknown>

const str = (v: unknown): string => (v === null || v === undefined ? '' : String(v))

/** Recorded value if the sidebar offers it, else the default plus a warning. */
function pick(id: string, recorded: unknown, warnings: string[]): string {
  const value = str(recorded)
  const fallback = DEFAULT_VALUES[id] ?? ''
  if (!value) return fallback
  const options = PARAM_SCHEMA.find((f) => f.id === id)?.options as Option[]
  if (options.some((o) => o.value === value)) return value
  warnings.push(`${id} “${value}” isn’t in the sidebar — using “${fallback}”`)
  return fallback
}

export function paramsFromSidecar(
  m: Sidecar,
  sdxlModel: string | null,
): { patch: Values; warnings: string[] } {
  const warnings: string[] = []
  const sampler = pick('sampler', m.sampler, warnings)
  const lengthMode = str(m.length_mode) || 'off'

  // The sidebar hides the global temperature under split (hidden = sent as 1.0),
  // so a recorded T has to move into the two it multiplies to stay equivalent.
  const t = m.temperature == null ? 1 : Number(m.temperature)
  const scaled = (v: unknown) =>
    v == null ? '' : String(Number((Number(v) * t).toPrecision(6)))

  const reach =
    m.target_distance != null ? 'shell'
    : m.radius_band != null || m.radius_scale != null ? 'band'
    : 'temperature'

  // Which file an image started from is random per image, so what can come back
  // is the folder. Inside a chosen folder the record is folder-relative, which
  // loses the name: then the whole pool is the honest answer.
  const initImage = str(m.init_image)
  const initDir = initImage.includes('/') ? initImage.split('/')[0] : ''
  if (initImage && !initDir)
    warnings.push(`started from init image ${initImage}; its folder isn’t recorded, so init is set to any folder`)

  const patch: Values = {
    action: 'generate',
    backend: str(m.backend) || DEFAULT_VALUES.backend,
    sampler,
    scheduler: pick('scheduler', m.scheduler, warnings),
    reach,
    temperature: sampler === 'split' ? '' : str(m.temperature),
    temp_on: sampler === 'split' ? scaled(m.temp_on) : '',
    temp_off: sampler === 'split' ? scaled(m.temp_off) : '',
    target_distance: str(m.target_distance),
    radius_scale: str(m.radius_scale),
    min_distance: str(m.min_distance),
    coherence: str(m.coherence),
    rho: str(m.rho),
    length_mode: lengthMode,
    // In corpus mode `length` is this one image's draw, not a setting.
    length: lengthMode === 'fixed' ? str(m.length) : DEFAULT_VALUES.length,
    components: str(m.components),
    comp_lo: m.comp_lo ? str(m.comp_lo) : '',
    truncation: str(m.truncation),
    equalize: m.equalize ? '1' : '',
    empirical_head: str(m.empirical_head),
    steps: str(m.steps),
    guidance: str(m.guidance),
    neg_mode: str(m.neg_mode),
    negative: str(m.negative),
    aspect: '',
    width: str(m.width),
    height: str(m.height),
    init_folder: !initImage ? 'off' : initDir || '__any__',
    init_mode: str(m.init_mode) || DEFAULT_VALUES.init_mode,
    init_strength: str(m.init_strength ?? m.ip_scale),
    seed: '',
  }
  if (sdxlModel) patch.model = sdxlModel
  return { patch, warnings }
}
