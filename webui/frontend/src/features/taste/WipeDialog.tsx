import { useState } from 'react'
import * as D from '@radix-ui/react-dialog'

import { useLabelFacets, useWipe, useWipePreview } from '../../api/queries'
import type { WipeFilters, WipeResult } from '../../api/types'
import { Confirm } from '../../components/ui/Confirm'
import { Select } from '../../components/ui/Select'
import { cn } from '../../lib/utils'
import type { Option } from '../../params/schema'
import { facetOptions, fromLocalInput, toLocalInput, WINDOWS } from '../fit/FitFilters'

const DEFAULTS: WipeFilters = {
  cutoff: 5,
  include_unlabeled: false,
  include_starred: false,
  backend: '',
  ckpt: '',
  since: '',
  until: '',
}

const CUTOFFS: Option[] = [
  { value: '0', label: 'keep all labeled images' },
  ...Array.from({ length: 9 }, (_, i) => ({
    value: String(i + 1),
    label: i === 0 ? 'scored 0' : `scored below ${i + 1}`,
  })),
  { value: '10', label: 'every score (0–9)' },
]

const plural = (n: number) => `${n} image${n === 1 ? '' : 's'}`

/**
 * The form is mounted only while the dialog is open, so every open starts from
 * the defaults: a past "delete everything" must not still be armed next week.
 */
function WipeForm({ onDone }: { onDone: (res: WipeResult) => void }) {
  const [f, setF] = useState<WipeFilters>(DEFAULTS)
  const [confirming, setConfirming] = useState(false)
  const [windowLabel, setWindowLabel] = useState<string | null>('all time')
  const set = <K extends keyof WipeFilters>(k: K, v: WipeFilters[K]) =>
    setF((old) => ({ ...old, [k]: v }))

  const { data: facets } = useLabelFacets()
  const { data: preview, isError, error } = useWipePreview(f)
  const wipe = useWipe()
  const count = preview?.count ?? 0

  // Frozen to an absolute instant on click, like the other time-window pickers.
  const pickWindow = (label: string, hours: number | null) => {
    setWindowLabel(label)
    setF((old) => ({
      ...old,
      since: hours == null ? '' : String(Math.floor(Date.now() / 1000 - hours * 3600)),
      until: '',
    }))
  }
  const setCustomTime = (k: 'since' | 'until', v: string) => {
    setWindowLabel(null)
    set(k, v)
  }

  const run = () =>
    wipe.mutate(
      // No end time set means "up to what the count covered", not "up to whatever
      // has finished rendering by the time this request lands".
      { ...f, until: f.until || String(preview?.as_of ?? '') },
      { onSuccess: onDone },
    )

  return (
    <>
      <div className="border-b border-line px-4 py-3">
        <D.Title className="text-[15px] font-semibold">🧹 Wipe images</D.Title>
        <D.Description className="mt-0.5 text-[12px] text-dim">
          Permanently delete images and their <code>.json</code>/<code>.npz</code> files. Your
          scores stay in the labels file.
        </D.Description>
      </div>

      <div className="flex flex-col gap-3 px-4 py-3">
        <div>
          <span className="sa-label !mt-0">Labeled images to delete</span>
          <Select
            value={String(f.cutoff)}
            onChange={(v) => set('cutoff', Number(v))}
            options={CUTOFFS}
            ariaLabel="Label score cutoff"
          />
        </div>

        <div className="flex flex-col gap-1.5 text-[13px]">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={f.include_unlabeled}
              onChange={(e) => set('include_unlabeled', e.target.checked)}
            />
            Also delete images I haven’t labeled
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={f.include_starred}
              onChange={(e) => set('include_starred', e.target.checked)}
            />
            Also delete starred images
          </label>
        </div>

        <div className="grid grid-cols-2 gap-3 border-t border-line pt-3">
          <div className="min-w-0">
            <span className="sa-label !mt-0">Backend</span>
            <Select
              value={f.backend}
              onChange={(v) => set('backend', v)}
              options={facetOptions(facets?.facets?.backend, 'any backend')}
              ariaLabel="Backend"
            />
          </div>
          <div className="min-w-0">
            <span className="sa-label !mt-0">Model</span>
            <Select
              value={f.ckpt}
              onChange={(v) => set('ckpt', v)}
              options={facetOptions(facets?.facets?.ckpt, 'any model')}
              ariaLabel="Model"
            />
          </div>
        </div>

        <div className="border-t border-line pt-3">
          <span className="sa-label !mt-0">Created</span>
          <div className="flex flex-wrap gap-1.5">
            {WINDOWS.map((w) => (
              <button
                key={w.label}
                className={cn('sa-btn sa-btn-sm', windowLabel === w.label && 'sa-btn-sel')}
                onClick={() => pickWindow(w.label, w.hours)}
              >
                {w.hours == null ? w.label : `last ${w.label}`}
              </button>
            ))}
          </div>
          <div className="mt-2 flex items-center gap-2">
            <input
              type="datetime-local"
              aria-label="Created from"
              className="sa-input min-w-0 flex-1 !py-[4px] !text-[12px]"
              value={toLocalInput(f.since)}
              onChange={(e) => setCustomTime('since', fromLocalInput(e.target.value))}
            />
            <span className="text-[12px] text-dim">to</span>
            <input
              type="datetime-local"
              aria-label="Created until"
              className="sa-input min-w-0 flex-1 !py-[4px] !text-[12px]"
              value={toLocalInput(f.until)}
              onChange={(e) => setCustomTime('until', fromLocalInput(e.target.value))}
            />
          </div>
        </div>
      </div>

      <div className="border-t border-line px-4 py-3">
        <div className="min-h-[38px] text-[12px] leading-relaxed text-dim">
          {isError ? (
            <span className="text-err">{(error as Error).message}</span>
          ) : preview ? (
            <>
              <span className="text-ink">
                {plural(count)} will be deleted
              </span>
              {count ? ` (${preview.labeled} labeled, ${preview.unlabeled} unlabeled)` : ''}.
              {preview.kept_starred ? <div>{plural(preview.kept_starred)} kept: starred.</div> : null}
              {preview.kept_lineage ? (
                <div>
                  {plural(preview.kept_lineage)} kept: images you’re keeping were made from them.
                </div>
              ) : null}
            </>
          ) : (
            'counting…'
          )}
          {wipe.isError ? <div className="text-err">{(wipe.error as Error).message}</div> : null}
        </div>
        <div className="mt-2 flex justify-end gap-2">
          <D.Close className="sa-btn">Cancel</D.Close>
          <button
            className="sa-btn border-err text-danger hover:border-err"
            disabled={!count || wipe.isPending}
            onClick={() => setConfirming(true)}
          >
            {wipe.isPending ? 'deleting…' : `Delete ${plural(count)}…`}
          </button>
        </div>
      </div>

      <Confirm
        open={confirming}
        onOpenChange={setConfirming}
        title={`Permanently delete ${plural(count)}?`}
        danger
        confirmLabel={`Delete ${count}`}
        body="This cannot be undone."
        onConfirm={run}
      />
    </>
  )
}

/** 🧹 Wipe — bulk-delete images chosen by label score, model and creation time. */
export function WipeDialog() {
  const [open, setOpen] = useState(false)
  const [done, setDone] = useState<string | null>(null)

  return (
    <>
      <button
        className="sa-btn sa-btn-sm text-danger"
        title="bulk-delete images by label score, model and creation time"
        onClick={() => {
          setDone(null)
          setOpen(true)
        }}
      >
        🧹 Wipe…
      </button>
      {done ? <span className="sa-hint self-center">{done}</span> : null}
      <D.Root open={open} onOpenChange={setOpen}>
        <D.Portal>
          <D.Overlay className="fixed inset-0 z-[95] bg-black/70" />
          <D.Content
            className="fixed left-1/2 top-1/2 z-[96] max-h-[92vh] w-[min(520px,94vw)]
                       -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-xl border
                       border-line bg-panel shadow-2xl"
          >
            <WipeForm
              onDone={(res) => {
                setDone(`deleted ${res.deleted} images (${res.files} files)`)
                setOpen(false)
              }}
            />
          </D.Content>
        </D.Portal>
      </D.Root>
    </>
  )
}
