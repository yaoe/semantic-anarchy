import { Slider } from '../../components/ui/Slider'
import { Tool } from '../../components/ui/Tool'
import { useUI } from '../../store'

/** hires: how far the enlarge goes, and how much of the original schedule re-runs. */
export const HIRES_FACTOR = 2.0
export const HIRES_DENOISE = 0.3

/**
 * Settings for the ⤴ Upscale action on gallery cards (POST /api/refine).
 * Nothing here submits — the card does, reading these values from the store.
 *
 * Three engines, and each one hides the knobs the others own:
 *  · hires — the same model re-rendering its own latents (two sliders, nothing else;
 *            steps/guidance/scheduler/seed all come from the source's sidecar)
 *  · flux  — a different model regenerating from the image as a reference
 *  · sd    — the general img2img refine, tiled or single-pass
 */
export function RefineBar() {
  const engine = useUI((s) => s.tools.rfEngine)
  const promptSel = useUI((s) => s.tools.rfPromptSel)

  return (
    <>
      <Tool
        id="rfEngine"
        label="Engine"
        tooltip="Same-latent hires re-renders the picture the model already drew, at more pixels."
        width="w-[170px]"
        options={[
          { value: 'hires', label: 'Same-latent hires' },
          { value: 'flux', label: 'FLUX klein' },
          { value: 'sd', label: 'SD img2img' },
        ]}
      />

      {engine === 'hires' ? (
        <>
          <Slider
            id="rfFactor"
            label="Upscale"
            tooltip="Target = source × this, snapped to a multiple of 16 px."
            min={1}
            max={3}
            step={0.05}
            fallback={HIRES_FACTOR}
            format={(v) => `×${v.toFixed(2)}`}
          />
          <Slider
            id="rfDenoise"
            label="Denoise"
            tooltip={
              'How much of the ORIGINAL schedule to re-run on the enlarged image. ' +
              '0.3 = its last 30% of steps, same conditioning, same seed — detail without drift.'
            }
            min={0.05}
            max={1}
            step={0.05}
            fallback={HIRES_DENOISE}
            width="w-[165px]"
            format={(v) => `${v.toFixed(2)} · last ${Math.round(v * 100)}%`}
          />
        </>
      ) : null}

      {engine !== 'hires' ? (
        <>
          <Tool
            id="rfScale"
            label="Upscale ×"
            width="w-[90px]"
            options={[
              { value: '1.25', label: '1.25' },
              { value: '1.5', label: '1.5' },
              { value: '2.0', label: '2.0' },
            ]}
          />
          <Tool id="rfSteps" label="Steps" type="number" placeholder="40" width="w-[80px]" />
        </>
      ) : null}

      {engine === 'sd' ? (
        <>
          <Tool
            id="rfMode"
            label="SD mode"
            width="w-[140px]"
            options={[
              { value: 'tiled', label: 'Detail (tiled)' },
              { value: 'single', label: 'Standard (1 pass)' },
            ]}
          />
          <Tool
            id="rfStrength"
            label="Denoise"
            type="number"
            step={0.05}
            placeholder="0.45"
            width="w-[90px]"
          />
          <Tool
            id="rfSched"
            label="Scheduler"
            width="w-[120px]"
            options={[
              { value: 'ddim', label: 'DDIM' },
              { value: 'default', label: 'default' },
              { value: 'dpm', label: 'DPM++ 2M' },
            ]}
          />
        </>
      ) : null}

      {engine === 'flux' ? (
        <>
          <Tool
            id="rfPromptSel"
            label="Flux prompt"
            tooltip="FLUX engine: instruction given alongside the reference image"
            width="w-[170px]"
            options={[
              { value: 'faithful', label: 'faithful upscale' },
              { value: 'recreate', label: 'creative re-render' },
              { value: 'custom', label: 'custom…' },
            ]}
          />
          {promptSel === 'custom' ? (
            <Tool
              id="rfPrompt"
              label="Custom prompt"
              placeholder="your instruction"
              width="w-[220px]"
            />
          ) : null}
        </>
      ) : null}
    </>
  )
}

/** The creative preset — the legacy "Recreate" instruction, verbatim. */
export const RECREATE_PROMPT =
  'Recreate this exact image at higher resolution with maximum fine detail and ' +
  'texture fidelity. Keep the composition, colors, style and every element identical.'

/** faithful -> null (refine_flux.py's own default); custom -> the typed text. */
export function refinePromptFor(tools: Record<string, string>): string | null {
  if (tools.rfPromptSel === 'recreate') return RECREATE_PROMPT
  if (tools.rfPromptSel === 'custom') return (tools.rfPrompt ?? '').trim() || null
  return null
}
