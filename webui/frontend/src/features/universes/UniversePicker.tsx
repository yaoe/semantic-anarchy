/**
 * "Add to universe…" for one image or many: every universe with how many of
 * these already wear it, click to put them all in (or, when all already are,
 * take them all out), or type a new name. Other universes the images wear are
 * kept — `useBulkTag` sends each image's complete set.
 */
import { useState } from 'react'

import { useBulkTag } from '../../api/queries'
import { cn } from '../../lib/utils'
import { useUniverses, wears } from './universes'

export function UniversePicker({
  rels,
  onDone,
  tone = 'app',
}: {
  rels: string[]
  /** after a change lands (e.g. close the popover) */
  onDone?: (tag: string, on: boolean) => void
  tone?: 'app' | 'map'
}) {
  const { list, byRel } = useUniverses()
  const bulk = useBulkTag()
  const [draft, setDraft] = useState('')

  const apply = (tag: string, on: boolean) =>
    bulk.mutate({ rels, tag, on }, { onSuccess: () => onDone?.(tag, on) })

  const commit = () => {
    const text = draft.trim()
    if (!text) return
    const exact = list.find((u) => u.tag.toLocaleLowerCase() === text.toLocaleLowerCase())
    apply(exact?.tag ?? text, true)
    setDraft('')
  }

  const map = tone === 'map'
  return (
    <div className="flex flex-col gap-1.5">
      <div className={cn('text-[10px] uppercase tracking-[2px]', map ? 'text-[#e9dccb]/60' : 'text-dim')}>
        {rels.length === 1 ? 'universes of this image' : `put these ${rels.length} into`}
      </div>
      <div className="flex max-h-[180px] flex-wrap gap-1 overflow-y-auto">
        {list.length === 0 ? (
          <span className="text-[11px] text-dim">no universes yet — name the first one below</span>
        ) : null}
        {list.map((u) => {
          const have = rels.filter((r) => wears(byRel.get(r), u.tag)).length
          const all = have === rels.length && have > 0
          return (
            <button
              key={u.tag}
              type="button"
              disabled={bulk.isPending || !rels.length}
              onClick={() => apply(u.tag, !all)}
              title={
                all
                  ? `all ${have} wear “${u.tag}” — click to take them out`
                  : `${have ? `${have} of ${rels.length} already` : 'none yet'} — click to add ${rels.length === 1 ? 'it' : 'all'}`
              }
              className={cn(
                'flex items-center gap-1 rounded-full border px-2 py-[2px] text-[11.5px] disabled:opacity-50',
                all
                  ? 'border-transparent text-[#111]'
                  : map
                    ? 'border-white/15 bg-black/30 text-[#efe6da]/85 hover:border-white/40'
                    : 'border-line bg-panel2 text-dim hover:border-line2 hover:text-ink',
              )}
              style={all ? { background: u.color } : undefined}
            >
              <span className="h-2 w-2 rounded-full" style={{ background: u.color }} />
              {u.tag}
              {have > 0 && !all ? <span className="opacity-60">{have}</span> : null}
            </button>
          )
        })}
      </div>
      <div className="flex gap-1">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            e.stopPropagation()
            if (e.key === 'Enter') commit()
          }}
          placeholder="new universe… ⏎"
          className={cn(
            'min-w-0 flex-1 rounded-[6px] border px-2 py-1 text-[12px] focus:outline-none',
            map
              ? 'border-white/15 bg-black/40 text-[#efe6da] placeholder:text-[#e9dccb]/40 focus:border-[#6fffd0]/60'
              : 'border-line bg-panel2 text-ink placeholder:text-dim focus:border-accent',
          )}
        />
        <button
          type="button"
          disabled={!draft.trim() || bulk.isPending}
          onClick={commit}
          className={cn(
            'rounded-[6px] border px-2 text-[12px] disabled:opacity-40',
            map ? 'border-[#6fffd0]/50 text-[#6fffd0]' : 'border-accent text-accent',
          )}
        >
          + add
        </button>
      </div>
      {bulk.isPending ? <div className="text-[11px] text-dim">tagging {rels.length}…</div> : null}
      {bulk.isError ? <div className="text-[11px] text-err">{bulk.error.message}</div> : null}
    </div>
  )
}
