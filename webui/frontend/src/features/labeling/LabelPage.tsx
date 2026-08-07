/**
 * 🏷 The labeling surface — the referee of the exploration loop.
 *
 * One image, one keypress, next image. Everything here serves that: the queue
 * is fetched once and frozen (see `useLabelQueue`), the next images are
 * prefetched while you look at the current one, and the knob readout is hidden
 * by default so the score stays perceptual rather than analytical.
 *
 * Keys: 0–9 score + advance · ←/→ navigate (relabeling allowed, latest wins) ·
 * Space skip · s star · k knobs.
 *
 * WHICH images are queued is `QueueBar`'s job, and the cursor/progress readout
 * is `LabelApp`'s single status line; this component is the score strip, the
 * image, and the keyboard.
 */
import { useCallback, useEffect, useMemo, useRef } from 'react'

import { useConfig, useFavorite, useLabelQueue, useSubmitLabel } from '../../api/queries'
import type { LabelRow } from '../../api/types'
import { cn, imgSrc } from '../../lib/utils'
import { useUI } from '../../store'
import { useLabelSelection } from './selection'

/** Score → ink, so a past call reads at a glance without being read. */
function scoreTone(s: number): string {
  if (s >= 7) return 'text-score'
  if (s >= 4) return 'text-dim'
  return 'text-danger'
}

/**
 * The 0–9 strip, spread across the full width directly above the image: the
 * hand and the eye stay in the same place, and each target is wide enough to
 * hit without aiming. Clickable for the mouse, but the keyboard is the point.
 */
function ScoreStrip({
  row,
  onPick,
  onStar,
}: {
  row?: LabelRow
  onPick: (s: number) => void
  onStar: () => void
}) {
  return (
    <div className="flex shrink-0 items-stretch gap-2">
      <div className="grid flex-1 grid-cols-10 gap-2">
        {Array.from({ length: 10 }, (_, s) => (
          <button
            key={s}
            disabled={!row}
            onClick={() => onPick(s)}
            title={`score ${s}`}
            className={cn(
              'h-9 rounded-md border text-[14px] tabular-nums transition-colors',
              'disabled:opacity-40',
              row?.score === s
                ? 'border-accent bg-accent text-white'
                : 'border-line bg-panel2 text-ink hover:border-line2',
            )}
          >
            {s}
          </button>
        ))}
      </div>
      {row?.score != null ? (
        <span
          className={cn(
            'flex w-[84px] items-center justify-center text-[12px] tabular-nums',
            scoreTone(row.score),
          )}
        >
          scored {row.score}
        </span>
      ) : (
        <span className="w-[84px]" />
      )}
      <button
        className={cn('sa-btn w-[104px] !py-0 text-[13px]', row?.fav && 'sa-btn-sel')}
        disabled={!row}
        onClick={onStar}
        title="star (s) — keeps the existing taste loop fed"
      >
        {row?.fav ? '★ starred' : '☆ star'}
      </button>
    </div>
  )
}

export function LabelPage() {
  const label = useUI((s) => s.label)
  const setLabel = useUI((s) => s.setLabel)
  const index = useUI((s) => s.labelIndex)
  const setIndex = useUI((s) => s.setLabelIndex)

  const selection = useLabelSelection()
  const { data: queue, isLoading } = useLabelQueue(selection)
  const { data: config } = useConfig()
  const submit = useSubmitLabel(selection)
  const favorite = useFavorite()

  const rows: LabelRow[] = useMemo(() => queue?.queue ?? [], [queue])
  const at = Math.min(index, Math.max(0, rows.length - 1))
  const row = rows[at] as LabelRow | undefined
  const showKnobs = label.knobs === '1'
  const panelSeeds = config?.seed_panel?.seeds

  // Prefetch the next few images while this one is being judged. Without it the
  // rhythm is keypress → blank → image, which is exactly the pause that turns a
  // five-minute batch into a fifteen-minute one.
  useEffect(() => {
    for (const nxt of rows.slice(at + 1, at + 4)) {
      const img = new Image()
      img.src = imgSrc(nxt.url, nxt.mtime)
    }
  }, [rows, at])

  const advance = useCallback(
    (delta: number) => setIndex(Math.max(0, Math.min(rows.length - 1, at + delta))),
    [at, rows.length, setIndex],
  )

  const score = useCallback(
    (s: number) => {
      if (!row) return
      submit.mutate({ rel: row.rel, score: s })
      // Advance immediately — the write is optimistic, so waiting on the server
      // would only ever make the loop slower, never more correct.
      if (at < rows.length - 1) setIndex(at + 1)
    },
    [row, submit, at, rows.length, setIndex],
  )

  const star = useCallback(() => {
    if (row) favorite.mutate({ rel: row.rel, on: !row.fav })
  }, [row, favorite])

  // One window-level handler. Held in a ref so the listener is attached once
  // rather than re-bound on every cursor move.
  const stateRef = useRef({ score, advance, star, showKnobs })
  stateRef.current = { score, advance, star, showKnobs }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      // A Radix Select renders its list in a portal and does typeahead on
      // printable keys — so while a queue picker is open, "4" is a search, not
      // a score. Anything with an open popup wins the keyboard.
      if (document.querySelector('[role="listbox"], [role="dialog"]')) return
      const { score: sc, advance: adv, star: st, showKnobs: kn } = stateRef.current
      if (e.key >= '0' && e.key <= '9') {
        e.preventDefault()
        sc(Number(e.key))
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        adv(-1)
      } else if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault()          // Space = skip without labeling
        adv(1)
      } else if (e.key === 's') {
        e.preventDefault()
        st()
      } else if (e.key === 'k') {
        e.preventDefault()
        setLabel('knobs', kn ? '' : '1')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [setLabel])

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <ScoreStrip row={row} onPick={score} onStar={star} />

      <div className="relative flex min-h-0 flex-1 items-center justify-center">
        {isLoading ? (
          <span className="text-dim">loading queue…</span>
        ) : !row ? (
          <span className="max-w-[52ch] text-center text-dim">
            {queue?.scope === 'unlabeled'
              ? 'nothing left unlabeled in this selection — widen the filters above, or switch the scope to “all” to go back over what you already scored.'
              : 'no images match this selection.'}
          </span>
        ) : (
          // h-full/w-full rather than max-*: the point of the one-line chrome
          // is that the image gets the room, and a 512² sd15 render marooned in
          // the middle of a 770px black field is harder to judge than the same
          // image scaled up to fill it. object-contain keeps the aspect ratio.
          <img
            key={row.rel}
            src={imgSrc(row.url, row.mtime)}
            alt={row.rel}
            className="h-full w-full object-contain"
          />
        )}
      </div>

      <div className="shrink-0 truncate font-mono text-[11px] leading-[1.6] text-dim">
        <span className="text-run">{row?.rel ?? ''}</span>
        {showKnobs && row ? (
          <>
            {row.experiment ? (
              <span className="ml-3">
                <span className="opacity-70">experiment</span> {row.experiment}
              </span>
            ) : null}
            {[
              ['ckpt', row.ckpt],
              ['size', row.size],
              ['d', row.distance],
            ].map(([k, v]) =>
              v == null ? null : (
                <span key={String(k)} className="ml-3">
                  <span className="opacity-70">{k}</span> {String(v)}
                </span>
              ),
            )}
            {row.image_seed != null ? (
              <span className="ml-3">
                <span className="opacity-70">seed</span> {row.image_seed}
                {panelSeeds?.includes(row.image_seed) ? ' ⧉' : ''}
              </span>
            ) : null}
            {Object.entries(row.knobs).map(([k, v]) => (
              <span key={k} className="ml-3">
                <span className="opacity-70">{k}</span> {String(v)}
              </span>
            ))}
          </>
        ) : null}
      </div>
    </div>
  )
}
