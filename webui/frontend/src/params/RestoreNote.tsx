import { basename } from '../lib/utils'
import { useUI } from '../store'

/**
 * Says the sidebar just changed under you, and what could not come back. Lives
 * beside the Run button because the change spans the whole scrolling form, and
 * because a silently kept checkpoint is exactly what you must see before Run.
 */
export function RestoreNote() {
  const note = useUI((s) => s.restoreNote)
  const clear = useUI((s) => s.clearRestoreNote)
  if (!note) return null
  return (
    <div className="mt-2 flex items-start gap-2 text-[12px] leading-snug text-dim">
      <div className="min-w-0 flex-1 break-words">
        settings from <span className="text-ink">{basename(note.rel)}</span>
        {note.source !== note.rel ? <> (sampled as {basename(note.source)})</> : null}
        , with a new random seed
        {note.warnings.map((w) => (
          <div key={w} className="mt-1 text-err">⚠ {w}</div>
        ))}
      </div>
      <button className="text-dim hover:text-ink" title="dismiss" onClick={clear}>✕</button>
    </div>
  )
}
