import { Select } from '../../components/ui/Select'
import { Tip } from '../../components/ui/Tip'
import { Tool } from '../../components/ui/Tool'
import { basename } from '../../lib/utils'
import { useUI } from '../../store'

/** Settings + status line for the 🧭 / 🚶 / 🧬 card actions (POST /api/explore). */
const WALK_DIRS = [
  { value: 'outward', label: 'outward' },
  { value: 'random', label: 'random' },
  { value: 'drift', label: 'drift' },
]

export function ExploreBar() {
  const breedParent = useUI((s) => s.breedParent)

  return (
    <>
      <Tool
        id="exRadius"
        label="Explore radius"
        tooltip="perturbation size for 🧭 Explore (fraction of the corpus spread)"
        type="number"
        step={0.05}
        placeholder="0.3"
        width="w-[110px]"
      />
      <Tool id="exN" label="Children" type="number" placeholder="6" width="w-[70px]" />
      <WalkTool />
      {breedParent ? (
        <span className="sa-hint mb-1.5 min-w-[220px] max-w-[520px] flex-1 basis-0">
          🧬 parent A = <b className="text-ink">{basename(breedParent)}</b> — click 🧬 on a
          second image (same backend) to breed, or click it again to cancel.
        </span>
      ) : (
        <Tip
          side="top"
          content={
            <>
              <b>🧭 Explore</b> samples around an image's latent point (hill-climb on taste);{' '}
              <b>🧬 Breed</b>: click two images to SLERP-cross them.
            </>
          }
        >
          <span className="sa-hint mb-2 cursor-help select-none text-[15px]" aria-label="help">
            ⓘ
          </span>
        </Tip>
      )}
    </>
  )
}

/** 🚶 walk knobs under one label: step · heading · turn (turn only for drift). */
function WalkTool() {
  const tools = useUI((s) => s.tools)
  const setTool = useUI((s) => s.setTool)
  const drift = tools.wkDir === 'drift'

  return (
    <div>
      <Tip
        side="top"
        content={
          <>
            🚶 <b>step</b>: distance growth per frame (0.15 = +15% further out each step) ·{' '}
            <b>heading</b>: outward = straight to the periphery, random = straight in a random
            direction, drift = the heading turns every frame (a smooth curve) · <b>turn</b>: drift's
            turn per frame in radians (0.35 ≈ 20°, 0.8 = twisty)
          </>
        }
      >
        <label className="sa-label !mt-0" htmlFor="t-wkStep">
          Walk
        </label>
      </Tip>
      <div className="flex gap-1">
        <input
          id="t-wkStep"
          className="sa-input w-[48px]"
          type="number"
          step={0.05}
          placeholder="0.15"
          title="step"
          value={tools.wkStep ?? ''}
          onChange={(e) => setTool('wkStep', e.target.value)}
        />
        <Select
          value={tools.wkDir ?? ''}
          onChange={(v) => setTool('wkDir', v)}
          options={WALK_DIRS}
          ariaLabel="Walk heading"
          className="w-[96px]"
        />
        {drift ? (
          <input
            className="sa-input w-[48px]"
            type="number"
            step={0.05}
            placeholder="0.35"
            title="turn (radians per frame)"
            aria-label="Turn"
            value={tools.wkTurn ?? ''}
            onChange={(e) => setTool('wkTurn', e.target.value)}
          />
        ) : null}
      </div>
    </div>
  )
}
