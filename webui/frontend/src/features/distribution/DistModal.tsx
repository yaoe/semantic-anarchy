import { useEffect, useState } from 'react'
import * as D from '@radix-ui/react-dialog'
import { useQueryClient } from '@tanstack/react-query'

import {
  useDistProbe,
  useEncodeDist,
  useFitList,
  useFs,
  useJobWatcher,
  useSetDist,
} from '../../api/queries'
import type { BackendId, DistKind, DistRow, FsEntry } from '../../api/types'
import { cn, fmtSize } from '../../lib/utils'

/** The two built-in bases, offered as one-click rows above the browser. */
const PRESETS: { kind: DistKind; label: string; note: string }[] = [
  { kind: 'base', label: 'base corpus', note: 'outputs/dist — the repo’s mined prompts_1000' },
  {
    kind: 'evolved',
    label: 'evolved ★ branch',
    note: 'outputs/dist_evolved — the older refit-around-your-stars branch, kept for '
      + 'fits already on disk. The 🧬 Fit tab is what makes new ones.',
  },
]

function icon(e: FsEntry): string {
  if (e.dir) return '📁'
  return e.kind === 'prompts' ? '📝' : '🎯'
}

/** "4144 prompts · 77×768" — the fit's shape, straight off its .meta.json. */
function metaLine(row?: DistRow | null): string | null {
  if (!row?.meta) return null
  return `${row.meta.n_samples} prompts · ${row.meta.feature_shape.join('×')}`
}

/**
 * Pick the distribution every sample is drawn from.
 *
 * A prompt corpus (.txt) anywhere on the machine can be selected, but only once
 * its latents exist: the encode pass writes them *beside* the .txt, tagged with
 * the checkpoint that produced them, so one corpus can hold a separate fit per
 * model and switching checkpoints switches which fit is in play. A saved .npz
 * (an earlier mine, an evolved branch) can also be picked directly.
 *
 * Browsing is server-side (`/api/fs?pick=dist`), so this works from a tailnet
 * device that has never seen the GPU box's filesystem.
 */
export function DistModal({
  open,
  onOpenChange,
  backend,
  modelKey,
  current,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  backend: BackendId
  modelKey?: string | null
  current?: DistRow
}) {
  const [dir, setDir] = useState<string | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [manual, setManual] = useState('')
  const [note, setNote] = useState<string | null>(null)
  const [encoding, setEncoding] = useState<number | null>(null)
  // Re-encoding overwrites a fit that already exists and costs GPU minutes, so
  // it asks once before queueing.
  const [confirmRe, setConfirmRe] = useState(false)

  const listing = useFs(dir, open, { pick: 'dist', backend, model: modelKey })
  const fits = useFitList(backend, open)
  const probe = useDistProbe(backend, open ? sel : null, modelKey)
  const setDist = useSetDist()
  const encode = useEncodeDist()
  const watchJob = useJobWatcher()
  const qc = useQueryClient()

  // Reopening lands on the current selection's folder with it highlighted,
  // rather than wherever the last browse happened to stop.
  useEffect(() => {
    if (!open) return
    setNote(null)
    setManual('')
    setSel(current?.path ?? null)
    setDir(current?.path ? current.path.slice(0, current.path.lastIndexOf('/')) : null)
  }, [open, current?.path])

  // A pending "overwrite?" must never carry over to a different corpus.
  useEffect(() => setConfirmRe(false), [sel])

  const row = probe.data
  const probeError = probe.error as Error | undefined
  const isCorpus = row?.kind === 'prompts'
  const ready = !!row?.ready
  const busy = setDist.isPending || encode.isPending || encoding != null

  function commit(kind: DistKind, path?: string | null) {
    setNote(null)
    setDist.mutate(
      { backend, kind, path: path ?? null, model: modelKey },
      { onSuccess: () => onOpenChange(false), onError: (e) => setNote((e as Error).message) },
    )
  }

  /** Queue the encode pass, then re-probe when the job lands. */
  function runEncode() {
    if (!sel) return
    setNote(null)
    setConfirmRe(false)
    encode.mutate(
      { backend, path: sel, model: modelKey },
      {
        onSuccess: (res) => {
          setEncoding(res.job_id)
          watchJob(res.job_id, (status) => {
            setEncoding(null)
            // The probe drives the footer button; the listing draws the
            // per-corpus "✓ encoded" badge; the sidebar shows the fit's shape
            // (and whether it has length stats). A re-encode restates all three.
            probe.refetch()
            qc.invalidateQueries({ queryKey: ['fs'] })
            qc.invalidateQueries({ queryKey: ['dist'] })
            if (status !== 'done') setNote(`encoding ${status} — see the job log`)
          })
        },
        onError: (e) => setNote((e as Error).message),
      },
    )
  }

  return (
    <D.Root open={open} onOpenChange={onOpenChange}>
      <D.Portal>
        <D.Overlay className="fixed inset-0 z-[95] bg-black/70" />
        <D.Content
          className="fixed left-1/2 top-1/2 z-[96] flex h-[min(660px,90vh)] w-[min(740px,94vw)]
                     -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border border-line
                     bg-panel shadow-2xl"
        >
          <div className="border-b border-line px-4 py-3">
            <D.Title className="text-[15px] font-semibold">Base distribution</D.Title>
            <D.Description className="mt-0.5 text-[12px] text-dim">
              A prompt corpus (<code>.txt</code>) to sample the conditioning of, or a saved
              fit (<code>.npz</code>) to use as-is.
            </D.Description>
            <p className="mt-1.5 flex min-w-0 items-baseline gap-1.5 text-[12px]">
              <span className="shrink-0">🧠</span>
              <span className="truncate text-ink" title={current?.model.path}>
                {current?.model.name ?? '—'}
              </span>
              <span className="shrink-0 text-dim">
                — latents will be encoded using this model
              </span>
            </p>
          </div>

          {/* the two built-in bases */}
          <div className="flex flex-wrap items-center gap-1.5 border-b border-line px-4 py-2">
            {PRESETS.map((p) => (
              <button
                key={p.kind}
                className={cn(
                  'sa-btn sa-btn-sm',
                  current?.kind === p.kind && !current?.path && 'sa-btn-sel',
                )}
                disabled={busy}
                title={p.note}
                onClick={() => commit(p.kind)}
              >
                {p.label}
              </button>
            ))}
            <span className="ml-auto text-[11px] text-dim">
              {listing.isFetching ? 'loading…' : `${listing.data?.entries.length ?? 0} items`}
            </span>
          </div>

          {/* fits made from picked images (🧬 Fit tab) — the common case now,
              so they get named rows rather than a hunt through outputs/ */}
          {(fits.data ?? []).filter((f) => f.ready && f.backend === backend).length ? (
            <div className="border-b border-line px-4 py-2">
              <p className="mb-1 text-[11px] uppercase tracking-[0.8px] text-dim">
                fitted from images
              </p>
              <div className="flex flex-wrap gap-1.5">
                {(fits.data ?? [])
                  .filter((f) => f.ready && f.backend === backend)
                  .map((f) => (
                    <button
                      key={f.base}
                      className={cn(
                        'sa-btn sa-btn-sm',
                        current?.path === f.files[0] && 'sa-btn-sel',
                      )}
                      disabled={busy}
                      title={`${f.n_samples ?? '?'} images${f.note ? ` — ${f.note}` : ''}`}
                      onClick={() => commit('file', f.files[0])}
                    >
                      🧬 {f.name}
                      <span className="ml-1 text-dim tabular-nums">{f.n_samples}</span>
                    </button>
                  ))}
              </div>
            </div>
          ) : null}

          {/* roots + breadcrumb */}
          <div className="flex flex-wrap items-center gap-1.5 border-b border-line px-4 py-2">
            {(listing.data?.roots ?? []).map((r) => (
              <button
                key={r.path}
                className={cn('sa-btn sa-btn-sm', listing.data?.path === r.path && 'sa-btn-sel')}
                onClick={() => setDir(r.path)}
              >
                {r.name}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2 border-b border-line px-4 py-2">
            <button
              className="sa-btn sa-btn-sm"
              disabled={!listing.data?.parent}
              onClick={() => setDir(listing.data?.parent ?? null)}
              title="parent folder"
            >
              ↑ up
            </button>
            <span
              className="min-w-0 flex-1 truncate font-mono text-[11px] text-dim"
              title={listing.data?.path}
            >
              {listing.data?.path ?? ''}
            </span>
          </div>

          {/* listing */}
          <div className="min-h-0 flex-1 overflow-y-auto px-2 py-1.5">
            {listing.error ? (
              <p className="px-2 py-3 text-[12px] text-danger">
                {(listing.error as Error).message}
              </p>
            ) : null}
            {(listing.data?.entries ?? []).map((e) => (
              <button
                key={e.path}
                className={cn(
                  'flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[13px]',
                  sel === e.path ? 'bg-panel2' : 'hover:bg-panel2',
                )}
                onClick={() => (e.dir ? setDir(e.path) : (setSel(e.path), setNote(null)))}
                onDoubleClick={() => {
                  if (!e.dir && e.kind === 'npz') commit('file', e.path)
                  if (!e.dir && e.kind === 'prompts' && e.ready) commit('prompts', e.path)
                }}
              >
                <span className="w-4 shrink-0 text-center">{icon(e)}</span>
                <span className="min-w-0 flex-1 truncate">{e.name}</span>
                {e.kind === 'prompts' ? (
                  <span
                    className={cn('shrink-0 text-[10px]', e.ready ? 'text-ok' : 'text-dim')}
                    title={
                      e.ready
                        ? 'already encoded with the active checkpoint'
                        : 'not encoded for the active checkpoint yet'
                    }
                  >
                    {e.ready ? '✓ encoded' : 'needs encoding'}
                  </span>
                ) : null}
                <span className="shrink-0 font-mono text-[11px] text-dim">
                  {e.size != null ? fmtSize(e.size) : ''}
                </span>
              </button>
            ))}
            {!listing.isFetching && !listing.error && !listing.data?.entries.length ? (
              <p className="px-2 py-3 text-[12px] text-dim">
                No prompt files or distributions here.
              </p>
            ) : null}
          </div>

          {/* what the current pick means */}
          <div className="border-t border-line px-4 py-2.5">
            {sel ? (
              <>
                <p className="truncate font-mono text-[11px] text-ink" title={sel}>
                  {sel}
                </p>
                {probeError ? (
                  <p className="sa-hint text-danger">{probeError.message}</p>
                ) : ready ? (
                  <p className="sa-hint text-ok">
                    ✓ encoded{metaLine(row) ? ` — ${metaLine(row)}` : ''}
                    {row?.files.length ? ` · ${row.files.length} tensor file(s)` : ''}
                  </p>
                ) : isCorpus ? (
                  <p className="sa-hint">
                    not encoded with <b>{row?.model.name}</b> yet — will be written to{' '}
                    <span className="font-mono">{row?.files[0]?.path.split('/').pop()}</span>
                  </p>
                ) : (
                  <p className="sa-hint text-danger">
                    missing:{' '}
                    {(row?.files ?? [])
                      .filter((f) => !f.exists)
                      .map((f) => f.path.split('/').pop())
                      .join(', ')}
                  </p>
                )}
              </>
            ) : (
              <p className="sa-hint">Pick a .txt corpus or a .npz distribution.</p>
            )}
            {encoding != null ? (
              <p className="sa-hint text-run">
                encoding… job #{encoding} — this runs the text encoder over the whole corpus
              </p>
            ) : null}
            {note ? <p className="sa-hint text-danger">{note}</p> : null}
          </div>

          {/* escape hatch + actions */}
          <div className="flex items-center gap-2 border-t border-line px-4 py-3">
            <input
              className="sa-input min-w-0 flex-1 font-mono text-[12px]"
              placeholder="…or paste an absolute path"
              value={manual}
              onChange={(ev) => setManual(ev.target.value)}
              onKeyDown={(ev) => {
                if (ev.key === 'Enter' && manual.trim()) setSel(manual.trim())
              }}
            />
            {isCorpus && !ready ? (
              <button
                className="sa-btn sa-btn-sel shrink-0"
                disabled={busy}
                onClick={runEncode}
                title="encode this corpus through the active checkpoint’s text encoder"
              >
                ⚙ Encode prompt distribution
              </button>
            ) : (
              <>
                {/* An already-encoded corpus can be mined again — the fit is
                    overwritten in place, which is how you pick up sampler
                    corrections a stale .npz was mined before. */}
                {isCorpus ? (
                  <button
                    className={cn('sa-btn shrink-0', confirmRe && 'sa-btn-sel')}
                    disabled={busy}
                    onClick={() => (confirmRe ? runEncode() : setConfirmRe(true))}
                    title={`re-run the text encoder over this corpus with ${
                      row?.model.name ?? 'the active checkpoint'
                    } and overwrite its existing fit`}
                  >
                    {confirmRe ? '↻ Overwrite fit?' : '↻ Re-encode'}
                  </button>
                ) : null}
                <button
                  className="sa-btn sa-btn-sel shrink-0"
                  disabled={busy || !sel || !ready}
                  onClick={() => commit(row?.kind ?? 'file', sel)}
                >
                  Select distribution
                </button>
              </>
            )}
            <D.Close className="sa-btn shrink-0">Cancel</D.Close>
          </div>
        </D.Content>
      </D.Portal>
    </D.Root>
  )
}
