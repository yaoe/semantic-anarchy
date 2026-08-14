/**
 * The 0–9 score strip — one keypress-wide target per score, plus the star.
 *
 * Shared by the 🏷 label page and the gallery lightbox so a judgement is made
 * the same way wherever you happen to be looking: same widths, same colours,
 * same meaning of "scored 7". Takes plain values rather than a queue row
 * precisely so the lightbox (which has an `ImageItem`, not a `LabelRow`) can
 * use it unchanged.
 */
import { cn } from '../../lib/utils'

/** Score → ink, so a past call reads at a glance without being read. */
function scoreTone(s: number): string {
  if (s >= 7) return 'text-score'
  if (s >= 4) return 'text-dim'
  return 'text-danger'
}

export function ScoreStrip({
  score,
  fav,
  onPick,
  onStar,
  disabled,
}: {
  /** The score that currently stands, or null when unjudged. */
  score: number | null
  fav: boolean
  onPick: (s: number) => void
  onStar: () => void
  disabled?: boolean
}) {
  return (
    <div className="flex shrink-0 items-stretch gap-2">
      <div className="grid flex-1 grid-cols-10 gap-2">
        {Array.from({ length: 10 }, (_, s) => (
          <button
            key={s}
            disabled={disabled}
            onClick={() => onPick(s)}
            title={`score ${s}`}
            className={cn(
              'h-9 rounded-md border text-[14px] tabular-nums transition-colors',
              'disabled:opacity-40',
              score === s
                ? 'border-accent bg-accent text-white'
                : 'border-line bg-panel2 text-ink hover:border-line2',
            )}
          >
            {s}
          </button>
        ))}
      </div>
      {score != null ? (
        <span
          className={cn(
            'flex w-[84px] items-center justify-center text-[12px] tabular-nums',
            scoreTone(score),
          )}
        >
          scored {score}
        </span>
      ) : (
        <span className="w-[84px]" />
      )}
      <button
        className={cn('sa-btn w-[104px] !py-0 text-[13px]', fav && 'sa-btn-sel')}
        disabled={disabled}
        onClick={onStar}
        title="star (s) — keeps the existing taste loop fed"
      >
        {fav ? '★ starred' : '☆ star'}
      </button>
    </div>
  )
}
