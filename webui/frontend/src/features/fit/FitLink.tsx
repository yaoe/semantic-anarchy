import { useFitList } from '../../api/queries'
import type { BackendId } from '../../api/types'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'

/**
 * 🧬 Fit ★ — the one-hop version of the fit loop, for the case it started as:
 * "make a distribution out of my starred images".
 *
 * It doesn't fit anything itself. It sets the ★-only filter and opens the 🧬 Fit
 * tab, where the selection is visible and editable before any file is written —
 * which is the whole difference from the old 🧪 Evolve ★ button, whose selection
 * was invisible and whose result was a branch that still sampled the corpus's
 * geometry (see semantic_anarchy/selection_fit.py).
 */
export function FitLink() {
  const setTab = useUI((s) => s.setTab)
  const setFit = useUI((s) => s.setFit)
  const backend = useUI((s) => s.params.backend) as BackendId
  const tab = useUI((s) => s.tab)
  const { data: fits } = useFitList(backend)

  return (
    <button
      className={cn('sa-btn sa-btn-sm', tab === 'fit' && 'sa-btn-sel')}
      title="fit a distribution to your starred images (opens the 🧬 Fit tab)"
      onClick={() => {
        setFit('starred', '1')
        setTab('fit')
      }}
    >
      🧬 Fit ★
      {/* No pick count here: next to the ★ it read as "90 starred", and the
          number that matters (what gets fitted) depends on the tab's filter. */}
      {fits?.length ? <span className="ml-1 text-dim">· {fits.length} saved</span> : null}
    </button>
  )
}
