/**
 * A small click-to-open panel anchored to its trigger, portalled to <body> so a
 * card's `overflow-hidden` or the map's stacking can't clip it. Closes on an
 * outside click, Escape, or when the content calls `close()`.
 *
 * Hand-rolled because the app ships no Radix popover; the needs are tiny.
 */
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

import { cn } from '../../lib/utils'

export function Popover({
  trigger,
  children,
  className,
  width = 260,
  title,
  triggerClassName,
  disabled,
}: {
  /** The button's content. */
  trigger: ReactNode
  children: (close: () => void) => ReactNode
  className?: string
  width?: number
  title?: string
  triggerClassName?: string
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const btn = useRef<HTMLButtonElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)

  useLayoutEffect(() => {
    if (!open || !btn.current) return
    const r = btn.current.getBoundingClientRect()
    const h = panel.current?.offsetHeight ?? 200
    const below = r.bottom + 6 + h < window.innerHeight
    setPos({
      left: Math.max(8, Math.min(r.left, window.innerWidth - width - 8)),
      top: below ? r.bottom + 6 : Math.max(8, r.top - 6 - h),
    })
  }, [open, width])

  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node
      if (!panel.current?.contains(t) && !btn.current?.contains(t)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        setOpen(false)
      }
    }
    window.addEventListener('pointerdown', onDown, true)
    window.addEventListener('keydown', onKey, true)
    return () => {
      window.removeEventListener('pointerdown', onDown, true)
      window.removeEventListener('keydown', onKey, true)
    }
  }, [open])

  return (
    <>
      <button
        ref={btn}
        type="button"
        title={title}
        disabled={disabled}
        aria-expanded={open}
        className={triggerClassName}
        onClick={(e) => {
          e.stopPropagation()
          setOpen((v) => !v)
        }}
      >
        {trigger}
      </button>
      {open
        ? createPortal(
            <div
              ref={panel}
              className={cn(
                'fixed z-[70] rounded-[10px] border border-line2 bg-panel p-2.5 text-[12px] text-ink shadow-2xl',
                className,
              )}
              style={{ width, left: pos?.left ?? -9999, top: pos?.top ?? -9999 }}
              onClick={(e) => e.stopPropagation()}
              onPointerDown={(e) => e.stopPropagation()}
            >
              {children(() => setOpen(false))}
            </div>,
            document.body,
          )
        : null}
    </>
  )
}
