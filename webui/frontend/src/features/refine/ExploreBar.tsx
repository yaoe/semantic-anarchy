import { Tool } from '../../components/ui/Tool'
import { basename } from '../../lib/utils'
import { useUI } from '../../store'

/** Settings + status line for the 🧭 / 🚶 / 🧬 card actions (POST /api/explore). */
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
      <Tool
        id="wkStep"
        label="Walk step"
        tooltip="🚶 walk: distance growth per frame (0.15 = +15% further out each step)"
        type="number"
        step={0.05}
        placeholder="0.15"
        width="w-[80px]"
      />
      <span className="sa-hint mb-1.5 max-w-[520px] flex-1">
        {breedParent ? (
          <>
            🧬 parent A = <b className="text-ink">{basename(breedParent)}</b> — click 🧬 on a
            second image (same backend) to breed, or click it again to cancel.
          </>
        ) : (
          <>
            <b className="text-ink">🧭 Explore</b> samples around an image's latent point
            (hill-climb on taste); <b className="text-ink">🧬 Breed</b>: click two images to
            SLERP-cross them.
          </>
        )}
      </span>
    </>
  )
}
