import { useState } from 'react'

import { useModelConfig, useNativePick, useSetModel } from '../../api/queries'
import type { BackendId, ModelRow } from '../../api/types'
import { Tip } from '../../components/ui/Tip'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { FileBrowser } from './FileBrowser'

/** Directory a browse should start in: where the current checkpoint lives. */
function startDir(row?: ModelRow): string | null {
  const p = row?.selected ?? (row?.kind === 'repo' ? null : row?.effective ?? null)
  if (!p) return null
  return row?.kind === 'diffusers' ? p : p.slice(0, p.lastIndexOf('/')) || null
}

/**
 * The checkpoint picker, slotted under the sidebar's Model group — hand-pick
 * the checkpoint this backend loads from, without editing run.sh.
 *
 * Clicking a browse button asks the *server host* to open its own OS file
 * dialog (zenity/kdialog/osascript). That only helps when you're sitting at the
 * machine, so if no dialog is available — or it fails — we fall back to the
 * in-browser server-side file browser, which works from any tailnet device.
 * The choice persists in webui/model_config.json, per backend.
 */
export function ModelPicker() {
  const backend = useUI((s) => s.params.backend) as BackendId
  const { data } = useModelConfig()
  const setModel = useSetModel()
  const nativePick = useNativePick()
  const [browsing, setBrowsing] = useState(false)
  const [note, setNote] = useState<string | null>(null)

  const row = data?.backends?.[backend]
  const native = data?.native_picker ?? null
  const overridden = !!row?.selected
  const missing = row?.exists === false

  function commit(path: string) {
    setNote(null)
    setBrowsing(false)
    setModel.mutate(
      { backend, path },
      { onError: (e) => setNote((e as Error).message) },
    )
  }

  /** Native dialog first; fall back to the in-browser browser on any failure. */
  function choose(mode: 'file' | 'folder') {
    setNote(null)
    if (!native) {
      setBrowsing(true)
      return
    }
    nativePick.mutate(
      { mode, start: startDir(row) },
      {
        onSuccess: (res) => {
          if (res.path) commit(res.path)
        },
        onError: (e) => {
          setNote(`${(e as Error).message} — browsing on the server instead`)
          setBrowsing(true)
        },
      },
    )
  }

  const busy = nativePick.isPending || setModel.isPending

  return (
    <section className="sa-panel mt-2 px-3 py-2.5">
      <div className="flex items-baseline gap-2">
        <h2 className="m-0 text-[12px] uppercase tracking-[0.8px] text-dim">Checkpoint</h2>
        <span className="text-[11px] text-dim">{backend}</span>
        {overridden ? (
          <Tip content="forget this pick and go back to the env-var / HF default">
            <button
              className="ml-auto text-[11px] text-dim underline decoration-dotted hover:text-ink"
              disabled={busy}
              onClick={() => {
                setNote(null)
                setModel.mutate({ backend, path: null })
              }}
            >
              reset
            </button>
          </Tip>
        ) : (
          <span className="ml-auto text-[10px] uppercase tracking-wide text-dim">default</span>
        )}
      </div>

      <Tip content={row?.effective} side="bottom">
        <div className="mt-1.5 flex min-w-0 items-center gap-1.5">
          <span className="shrink-0">
            {row?.kind === 'diffusers' ? '📦' : row?.kind === 'repo' ? '☁️' : '🧠'}
          </span>
          <span
            className={cn(
              'min-w-0 flex-1 truncate text-[13px]',
              missing ? 'text-danger' : 'text-ink',
            )}
          >
            {row?.name ?? '—'}
          </span>
        </div>
      </Tip>
      <p className="truncate font-mono text-[10px] text-dim" title={row?.effective}>
        {row?.effective ?? ''}
      </p>
      {missing ? (
        <p className="sa-hint text-danger">checkpoint not found on disk</p>
      ) : null}

      <div className="mt-2 flex items-center gap-1.5">
        <button className="sa-btn sa-btn-sm flex-1" disabled={busy} onClick={() => choose('file')}>
          {nativePick.isPending ? '📂 dialog open…' : '📂 Choose…'}
        </button>
        <Tip content="pick a diffusers model *folder* (one with model_index.json)">
          <button className="sa-btn sa-btn-sm" disabled={busy} onClick={() => choose('folder')}>
            📁
          </button>
        </Tip>
        {native ? (
          <Tip content="browse the server's filesystem in this page — use this when you're on a different device than the GPU box">
            <button
              className="sa-btn sa-btn-sm"
              disabled={busy}
              onClick={() => setBrowsing(true)}
            >
              🔎
            </button>
          </Tip>
        ) : null}
      </div>
      <p className="sa-hint">
        {nativePick.isPending
          ? `answer the ${native} dialog on the machine running this server`
          : native
            ? `OS dialog opens on the server host (${native})`
            : 'no OS dialog on the server host — browsing in-page'}
      </p>
      {note ? <p className="sa-hint text-danger">{note}</p> : null}

      <FileBrowser
        open={browsing}
        onOpenChange={setBrowsing}
        start={startDir(row)}
        onPick={commit}
      />
    </section>
  )
}
