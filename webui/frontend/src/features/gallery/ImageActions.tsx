/**
 * Renders the per-image action set from `useImageActions` — the gallery card's
 * icon row and the fullscreen viewer's labelled toolbar draw the SAME defs.
 */
import { Popover } from '../../components/ui/Popover'
import { Tip } from '../../components/ui/Tip'
import { basename, cn } from '../../lib/utils'
import { UniversePicker } from '../universes/UniversePicker'
import type { ActionFeedback, ImageAction } from './useImageActions'

/* --------------------------------------------------------------- render --- */

/** A square-ish icon button that shrinks to share one line with its siblings. */
const CARD_BTN =
  'sa-btn sa-btn-sm min-w-0 max-w-[34px] flex-1 overflow-hidden whitespace-nowrap px-0 py-0 ' +
  'inline-flex h-[22px] items-center justify-center'

const TONE_ON: Record<ImageAction['tone'], string> = {
  star: 'text-star border-star',
  run: 'border-run text-run',
  sel: 'sa-btn-sel',
  danger: '',
}

/** `on` wears the action's own tone; a just-queued job is always "run" green,
 *  so a finished breed never reads as "parent A is still picked". */
function toneClass(a: ImageAction): string {
  return cn(
    a.on ? TONE_ON[a.tone] : a.done && TONE_ON.run,
    a.tone === 'danger' && 'hover:border-err hover:text-err',
  )
}

/**
 * The action set for one image.
 *  - `card`: one line of icon-only buttons (the name lives in the tooltip).
 *  - `viewer`: a labelled toolbar with hotkeys in the tooltips, plus the last
 *    job it queued, so a submit never needs the viewer closed to confirm.
 */
export function ImageActions({
  actions,
  feedback,
  error,
  variant,
  current,
  rel,
}: {
  actions: ImageAction[]
  feedback?: ActionFeedback | null
  error?: string | null
  variant: 'card' | 'viewer'
  /** The viewer's open image, so feedback about another one is named. */
  current?: string | null
  /** The card's image — what a menu action (universe picker) acts on. */
  rel?: string
}) {
  if (variant === 'card') {
    return (
      <>
        <div className="flex items-center gap-px p-px">
          {actions.map((a) =>
            a.menu === 'universe' ? (
              rel ? (
                <Popover
                  key={a.id}
                  title={a.title}
                  width={270}
                  triggerClassName={cn(CARD_BTN, toneClass(a))}
                  trigger={<a.Icon size={14} />}
                >
                  {() => <UniversePicker rels={[rel]} />}
                </Popover>
              ) : null
            ) : (
            <button
              key={a.id}
              className={cn(CARD_BTN, toneClass(a), a.id === 'delete' && 'ml-auto')}
              title={a.done && feedback ? `${feedback.text} — click to queue another` : a.title}
              disabled={a.disabled}
              onClick={a.run}
            >
              <a.Icon size={14} fill={a.id === 'fav' && a.on ? 'currentColor' : 'none'} />
            </button>
            ),
          )}
        </div>
        {error ? (
          <span className="truncate px-1.5 text-[11px] text-err" title={error}>
            {error}
          </span>
        ) : null}
      </>
    )
  }

  const about = feedback && current && feedback.rel !== current ? ` (${basename(feedback.rel)})` : ''
  return (
    <div className="flex flex-wrap items-center gap-1.5 font-sans">
      {actions.filter((a) => !a.menu).map((a) => (
        <Tip
          key={a.id}
          side="top"
          content={
            <>
              {a.done && feedback ? `${feedback.text} — again to queue another` : a.title}
              {a.hotkey ? <kbd className="ml-1.5 rounded border border-line2 px-1 text-[11px]">{a.hotkey}</kbd> : null}
            </>
          }
        >
          <button
            className={cn(
              'sa-btn sa-btn-sm inline-flex items-center gap-1.5',
              toneClass(a),
              a.id === 'delete' && 'ml-auto',
            )}
            disabled={a.disabled}
            onClick={a.run}
            aria-keyshortcuts={a.hotkey}
          >
            <a.Icon size={13} fill={a.id === 'fav' && a.on ? 'currentColor' : 'none'} />
            {a.label}
            {a.hotkey ? <span className="text-[10px] text-dim">{a.hotkey}</span> : null}
          </button>
        </Tip>
      ))}
      {feedback ? (
        <span
          className={cn('basis-full text-[12px]', feedback.ok ? 'text-run' : 'text-err')}
          role="status"
        >
          {feedback.ok ? '✓ ' : '✕ '}
          {feedback.text}
          {about}
        </span>
      ) : null}
    </div>
  )
}
