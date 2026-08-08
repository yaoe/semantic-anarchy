import { useState } from 'react'

import { useEvolve } from '../../api/queries'
import { Confirm } from '../../components/ui/Confirm'

/**
 * 🧪 Evolve ★ — refit a distribution branch around the starred latents and
 * sample it (scripts/evolve_favorites.py, via POST /api/evolve).
 */
export function EvolveBar() {
  const [open, setOpen] = useState(false)
  const evolve = useEvolve()

  return (
    <>
      <button
        className="sa-btn sa-btn-sm"
        title="refit a distribution branch around your ★ favorites and sample it"
        disabled={evolve.isPending}
        onClick={() => setOpen(true)}
      >
        🧪 Evolve ★
      </button>
      <Confirm
        open={open}
        onOpenChange={setOpen}
        title="Evolve from your ★ favorites?"
        body={
          <>
            Refits the distribution around your starred latents (grafting the corpus PCA
            axes back on) and samples 8 images from the new branch. Writes
            <code className="mx-1">outputs/dist_evolved*</code>.
            {evolve.isError ? (
              <div className="mt-2 text-err">{(evolve.error as Error).message}</div>
            ) : null}
          </>
        }
        confirmLabel="Evolve"
        onConfirm={() => evolve.mutate({ n: 8 })}
      />
      {evolve.isError ? (
        <span className="sa-hint text-err self-center">
          {(evolve.error as Error).message}
        </span>
      ) : null}
    </>
  )
}
