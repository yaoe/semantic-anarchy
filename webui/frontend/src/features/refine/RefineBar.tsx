import { Tool } from '../../components/ui/Tool'
import { useUI } from '../../store'

/**
 * Settings for the ⤴ Upscale action on gallery cards (POST /api/refine).
 * Nothing here submits — the card does, reading these values from the store.
 */
export function RefineBar() {
  const engine = useUI((s) => s.tools.rfEngine)
  const promptSel = useUI((s) => s.tools.rfPromptSel)

  return (
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
      <Tool
        id="rfEngine"
        label="Engine"
        width="w-[150px]"
        options={[
          { value: 'flux', label: 'FLUX klein (best)' },
          { value: 'sd', label: 'SD img2img' },
        ]}
      />
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
      ) : (
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
      )}
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
