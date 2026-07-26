import { useEffect, useState } from 'react'
import * as D from '@radix-ui/react-dialog'

import { useFs } from '../../api/queries'
import type { FsEntry } from '../../api/types'
import { cn } from '../../lib/utils'

function gb(bytes: number | null): string {
  if (bytes == null) return ''
  return bytes >= 1 << 30
    ? `${(bytes / (1 << 30)).toFixed(1)} GB`
    : `${Math.round(bytes / (1 << 20))} MB`
}

/** A row is pickable if it's a single-file checkpoint or a diffusers folder. */
function pickable(e: FsEntry): boolean {
  return e.kind === 'ckpt' || e.kind === 'diffusers'
}

/**
 * The fallback picker: a server-side file browser, for when the dashboard is
 * open on a *different* device than the one running the models (a native OS
 * dialog would then open on a screen nobody is looking at). Sandboxed
 * server-side to `browse_roots()`.
 */
export function FileBrowser({
  open,
  onOpenChange,
  start,
  onPick,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  start?: string | null
  onPick: (path: string) => void
}) {
  const [path, setPath] = useState<string | null>(start ?? null)
  const [manual, setManual] = useState('')
  const { data, isFetching, error } = useFs(path, open)

  // Reopening lands back where the current checkpoint lives, not where the
  // last browse session happened to end.
  useEffect(() => {
    if (open) {
      setPath(start ?? null)
      setManual('')
    }
  }, [open, start])

  const here = data?.path ?? ''
  const selfPickable = data?.kind === 'diffusers'

  return (
    <D.Root open={open} onOpenChange={onOpenChange}>
      <D.Portal>
        <D.Overlay className="fixed inset-0 z-[95] bg-black/70" />
        <D.Content
          className="fixed left-1/2 top-1/2 z-[96] flex h-[min(640px,88vh)] w-[min(720px,94vw)]
                     -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border border-line
                     bg-panel shadow-2xl"
        >
          <div className="border-b border-line px-4 py-3">
            <D.Title className="text-[15px] font-semibold">Pick a model</D.Title>
            <D.Description className="mt-0.5 text-[12px] text-dim">
              A <code>.safetensors</code>/<code>.ckpt</code> file, or a diffusers folder
              (one containing <code>model_index.json</code>).
            </D.Description>
          </div>

          {/* roots + breadcrumb */}
          <div className="flex flex-wrap items-center gap-1.5 border-b border-line px-4 py-2">
            {(data?.roots ?? []).map((r) => (
              <button
                key={r.path}
                className={cn('sa-btn sa-btn-sm', here === r.path && 'sa-btn-sel')}
                onClick={() => setPath(r.path)}
              >
                {r.name}
              </button>
            ))}
            <span className="ml-auto text-[11px] text-dim">
              {isFetching ? 'loading…' : `${data?.entries.length ?? 0} items`}
            </span>
          </div>
          <div className="flex items-center gap-2 border-b border-line px-4 py-2">
            <button
              className="sa-btn sa-btn-sm"
              disabled={!data?.parent}
              onClick={() => setPath(data?.parent ?? null)}
              title="parent folder"
            >
              ↑ up
            </button>
            <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-dim" title={here}>
              {here}
            </span>
            {selfPickable ? (
              <button className="sa-btn sa-btn-sm sa-btn-sel" onClick={() => onPick(here)}>
                use this folder
              </button>
            ) : null}
          </div>

          {/* listing */}
          <div className="min-h-0 flex-1 overflow-y-auto px-2 py-1.5">
            {error ? (
              <p className="px-2 py-3 text-[12px] text-danger">{(error as Error).message}</p>
            ) : null}
            {(data?.entries ?? []).map((e) => (
              <button
                key={e.path}
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left
                           text-[13px] hover:bg-panel2"
                onDoubleClick={() => {
                  if (pickable(e) && !e.dir) onPick(e.path)
                }}
                onClick={() => (e.dir ? setPath(e.path) : onPick(e.path))}
              >
                <span className="w-4 shrink-0 text-center">
                  {e.kind === 'diffusers' ? '📦' : e.dir ? '📁' : '🧠'}
                </span>
                <span className={cn('min-w-0 flex-1 truncate', pickable(e) && 'text-ink')}>
                  {e.name}
                </span>
                {e.kind === 'diffusers' ? (
                  <span className="shrink-0 text-[10px] text-dim">diffusers</span>
                ) : null}
                <span className="shrink-0 font-mono text-[11px] text-dim">{gb(e.size)}</span>
              </button>
            ))}
            {!isFetching && !error && !data?.entries.length ? (
              <p className="px-2 py-3 text-[12px] text-dim">
                No checkpoints or subfolders here.
              </p>
            ) : null}
          </div>

          {/* escape hatch: paste an absolute path (e.g. a drive outside the roots) */}
          <div className="flex items-center gap-2 border-t border-line px-4 py-3">
            <input
              className="sa-input flex-1 font-mono text-[12px]"
              placeholder="…or paste an absolute path"
              value={manual}
              onChange={(ev) => setManual(ev.target.value)}
              onKeyDown={(ev) => {
                if (ev.key === 'Enter' && manual.trim()) onPick(manual.trim())
              }}
            />
            <button
              className="sa-btn"
              disabled={!manual.trim()}
              onClick={() => onPick(manual.trim())}
            >
              Use
            </button>
            <D.Close className="sa-btn">Cancel</D.Close>
          </div>
        </D.Content>
      </D.Portal>
    </D.Root>
  )
}
