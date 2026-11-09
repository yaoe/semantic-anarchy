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

  // One wrapping row of pills: the name is the "use" button, details live in
  // the tooltip, 🗑 deletes. The list only needs to answer "which fits exist,
  // which is in use".
  return (
    <section className="flex shrink-0 flex-wrap items-center gap-1.5 px-1 text-[12px]">
      <span className="text-dim">Saved fits</span>
      {!rows.length ? (
        <span className="text-dim">— none yet; a fit shows up here and in the sidebar’s distribution picker</span>
      ) : null}
      {rows.map((f) => {
        const active = current?.base === f.base || current?.path === f.files[0]
        const other = f.backend !== backend
        const info =
          `${f.n_samples ?? '?'} images · ${f.backend ?? '?'}` +
          (f.meta ? ` · ${f.meta.feature_shape.join('×')}` : '') +
          ` · ${when(f.created)}` +
          (f.note ? `\n${f.note}` : '')
        return (
          <span
            key={f.base}
            className={cn(
              'group inline-flex items-center overflow-hidden rounded-full border',
              active ? 'border-run bg-panel2' : 'border-line',
            )}
          >
            <button
              className={cn('px-2 py-[2px]', active ? 'text-run' : 'text-ink hover:text-run',
                (!f.ready || other) && 'cursor-not-allowed opacity-60')}
              disabled={active || !f.ready || other || setDist.isPending}
              title={
                !f.ready ? `files missing — ${info}`
                  : other ? `fitted on ${f.backend} — switch the backend to use it\n${info}`
                    : active ? `in use — ${info}` : `sample from this fit — ${info}`
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
              🧬 {f.name} <span className="tabular-nums text-dim">{f.n_samples ?? '?'}</span>
              {active ? <span className="ml-1">· in use</span> : null}
            </button>
            <button
              className="border-l border-line px-1.5 py-[2px] text-dim hover:text-err"
              title="delete this fit"
              disabled={del.isPending}
              onClick={() => setPendingDelete(f)}
            >
              ×
            </button>
          </span>
        )
      })}
      {del.isError ? <span className="text-err">{(del.error as Error).message}</span> : null}

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
