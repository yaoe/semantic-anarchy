/**
 * The fits you have already made. Each row is a distribution on disk: select it
 * as the base every sample is drawn from, or delete it.
 *
 * A fit is cheap to remake (the latents never move), so deleting is a normal
 * part of iterating rather than a destructive act — it still confirms once,
 * because a fit is also the only record of a selection you assembled by hand.
 */
import { useState } from 'react'

import { useDeleteFit, useDistConfig, useFitList, useSetDist } from '../../api/queries'
import type { BackendId, SavedFit } from '../../api/types'
import { Confirm } from '../../components/ui/Confirm'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'

function when(created: number | null): string {
  if (!created) return ''
  return new Date(created * 1000).toLocaleString([], {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

export function SavedFits() {
  const backend = useUI((s) => s.params.backend) as BackendId
  const modelKey = useUI((s) => s.params.model)
  const { data: fits } = useFitList(backend)
  const { data: current } = useDistConfig(backend, backend === 'sdxl' ? modelKey : null)
  const setDist = useSetDist()
  const del = useDeleteFit()
  const [pendingDelete, setPendingDelete] = useState<SavedFit | null>(null)

  const rows = fits ?? []

  return (
    <section className="sa-panel shrink-0 px-3 py-2.5">
      <div className="flex items-baseline gap-2">
        <h2 className="m-0 text-[12px] uppercase tracking-[0.8px] text-dim">Saved fits</h2>
        <span className="text-[11px] text-dim">outputs/dist_fits · {rows.length}</span>
        {del.isError ? (
          <span className="text-[11px] text-err">{(del.error as Error).message}</span>
        ) : null}
      </div>

      {!rows.length ? (
        <p className="sa-hint">
          none yet — pick images above and fit one. It then shows up here and in the
          sidebar’s “Select base distribution…”.
        </p>
      ) : (
        <div className="mt-1.5 flex flex-col gap-1">
          {rows.map((f) => {
            const active = current?.base === f.base || current?.path === f.files[0]
            return (
              <div
                key={f.base}
                className={cn(
                  'flex flex-wrap items-center gap-2 rounded-md border px-2 py-1.5 text-[12px]',
                  active ? 'border-run bg-panel2' : 'border-line',
                )}
              >
                <span className="shrink-0">🧬</span>
                <span className="min-w-0 flex-1 truncate text-[13px] text-ink" title={f.base}>
                  {f.name}
                </span>
                <span className="shrink-0 text-dim tabular-nums">
                  {f.n_samples ?? '?'} images
                </span>
                {f.backend ? <span className="shrink-0 text-dim">{f.backend}</span> : null}
                {f.meta ? (
                  <span className="shrink-0 text-dim">
                    {f.meta.feature_shape.join('×')}
                  </span>
                ) : null}
                <span className="shrink-0 text-dim">{when(f.created)}</span>
                {f.note ? (
                  <span className="min-w-0 basis-full truncate text-dim" title={f.note}>
                    {f.note}
                  </span>
                ) : null}
                {!f.ready ? (
                  <span className="shrink-0 text-err">files missing</span>
                ) : active ? (
                  <span className="shrink-0 text-run">in use</span>
                ) : (
                  <button
                    className="sa-btn sa-btn-sm shrink-0"
                    disabled={setDist.isPending || f.backend !== backend}
                    title={
                      f.backend !== backend
                        ? `fitted on ${f.backend} — switch the backend to use it`
                        : 'sample from this fit'
                    }
                    onClick={() =>
                      setDist.mutate({
                        backend,
                        kind: 'file',
                        path: f.files[0],
                        model: backend === 'sdxl' ? modelKey : null,
                      })
                    }
                  >
                    use
                  </button>
                )}
                <button
                  className="sa-btn sa-btn-sm shrink-0"
                  title="delete this fit"
                  disabled={del.isPending}
                  onClick={() => setPendingDelete(f)}
                >
                  🗑
                </button>
              </div>
            )
          })}
        </div>
      )}

      <Confirm
        open={pendingDelete != null}
        onOpenChange={(v) => !v && setPendingDelete(null)}
        title={`Delete “${pendingDelete?.name}”?`}
        body={
          <>
            Removes the fitted <code>.npz</code> and its manifest. The images it was
            fitted from are untouched — but the list of exactly which ones they were
            goes with it.
          </>
        }
        confirmLabel="Delete"
        danger
        onConfirm={() => {
          if (pendingDelete) del.mutate(pendingDelete.name)
          setPendingDelete(null)
        }}
      />
    </section>
  )
}
