/**
 * 🏷 The tag strip — which aesthetic universe the image in front of you belongs to.
 *
 * A score says how GOOD an image is; a tag says what KIND it is. Both are
 * assigned from the same keyboard, which is why this sits directly under the
 * score strip rather than behind a dialog: sorting is a judgement made at the
 * same glance as scoring, and anything that costs a mouse trip won't get done
 * across ten thousand images.
 *
 * Keys (owned by `LabelPage`, listed here because they define the layout):
 * `t` focuses the box · `Shift+1…9` toggles the first nine universes · Enter in
 * the box assigns the typed one, creating it if it is new · Escape leaves.
 *
 * The vocabulary is derived server-side from what images actually wear, so
 * "create" is not a separate act — assigning a name that nobody uses yet IS
 * creating it, and the last image to drop a name retires it.
 */
import { useMemo, useState } from 'react'
import type { RefObject } from 'react'

import type { TagCell } from '../../api/types'
import { cn } from '../../lib/utils'

/** How many universes get a `Shift+<n>` shortcut. Beyond that, type the name. */
const SHORTCUTS = 9

export function TagStrip({
  tags,
  vocabulary,
  onSet,
  inputRef,
  disabled,
}: {
  /** The universes this image currently wears. */
  tags: string[]
  /** Every universe in use, commonest first — the server's own ordering. */
  vocabulary: TagCell[]
  onSet: (tags: string[]) => void
  inputRef: RefObject<HTMLInputElement | null>
  disabled?: boolean
}) {
  const [draft, setDraft] = useState('')

  const worn = useMemo(
    () => new Set(tags.map((t) => t.toLocaleLowerCase())),
    [tags],
  )
  const has = (tag: string) => worn.has(tag.toLocaleLowerCase())

  /**
   * Chips shown, in a STABLE order: the shortcut-bearing ones must not move
   * under the fingers when a count changes, so the vocabulary's own order is
   * kept and only the typed filter narrows it. Universes this image wears are
   * always shown, even when filtered out, so you can always take one off.
   */
  const shown = useMemo(() => {
    const needle = draft.trim().toLocaleLowerCase()
    if (!needle) return vocabulary
    return vocabulary.filter(
      (c) => c.tag.toLocaleLowerCase().includes(needle) || has(c.tag),
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vocabulary, draft, worn])

  const toggle = (tag: string) => {
    if (disabled) return
    onSet(has(tag) ? tags.filter((t) => t.toLocaleLowerCase() !== tag.toLocaleLowerCase()) : [...tags, tag])
  }

  /** Enter: assign what was typed — an exact existing universe, or a new one. */
  const commit = () => {
    const text = draft.trim()
    if (!text || disabled) return
    const exact = vocabulary.find((c) => c.tag.toLocaleLowerCase() === text.toLocaleLowerCase())
    const tag = exact?.tag ?? text
    if (!has(tag)) onSet([...tags, tag])
    setDraft('')
  }

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-2">
      <input
        ref={inputRef}
        value={draft}
        disabled={disabled}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          // The window handler ignores inputs, so the keys that matter while
          // typing are handled here and nowhere else.
          if (e.key === 'Enter') {
            e.preventDefault()
            commit()
          } else if (e.key === 'Escape') {
            e.preventDefault()
            setDraft('')
            e.currentTarget.blur()
          }
        }}
        placeholder="tag (t) — filter or name a new universe, ⏎ to assign"
        className={cn(
          'h-8 w-[280px] rounded-md border border-line bg-panel2 px-2',
          'text-[12px] text-ink placeholder:text-dim',
          'focus:border-accent focus:outline-none disabled:opacity-40',
        )}
      />

      {draft.trim() &&
      !vocabulary.some((c) => c.tag.toLocaleLowerCase() === draft.trim().toLocaleLowerCase()) ? (
        <button
          onClick={commit}
          disabled={disabled}
          title="create this universe and assign it"
          className="h-8 rounded-md border border-accent bg-panel2 px-2 text-[12px] text-accent"
        >
          + {draft.trim()}
        </button>
      ) : null}

      <div className="flex flex-1 flex-wrap items-center gap-1.5">
        {shown.length === 0 && !draft ? (
          <span className="text-[11px] text-dim">
            no universes yet — type a name and press ⏎ to make the first one
          </span>
        ) : null}
        {shown.map((cell, i) => {
          const on = has(cell.tag)
          return (
            <button
              key={cell.tag}
              onClick={() => toggle(cell.tag)}
              disabled={disabled}
              title={
                `${cell.count} image${cell.count === 1 ? '' : 's'}` +
                (i < SHORTCUTS ? ` · shift+${i + 1}` : '')
              }
              className={cn(
                'h-7 rounded-full border px-2.5 text-[12px] transition-colors disabled:opacity-40',
                on
                  ? 'border-accent bg-accent text-white'
                  : 'border-line bg-panel2 text-dim hover:border-line2 hover:text-ink',
              )}
            >
              {i < SHORTCUTS ? (
                <span className={cn('mr-1 tabular-nums', on ? 'opacity-70' : 'opacity-50')}>
                  {i + 1}
                </span>
              ) : null}
              {cell.tag}
            </button>
          )
        })}
      </div>
    </div>
  )
}
