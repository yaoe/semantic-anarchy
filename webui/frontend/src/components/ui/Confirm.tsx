import type { ReactNode } from 'react'
import * as AD from '@radix-ui/react-alert-dialog'

import { cn } from '../../lib/utils'

/**
 * Controlled confirmation dialog — replaces the legacy `confirm()` calls for
 * the destructive/expensive actions (wipe, evolve, film delete).
 */
export function Confirm({
  open,
  onOpenChange,
  title,
  body,
  confirmLabel = 'Confirm',
  danger = false,
  onConfirm,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  title: string
  body?: ReactNode
  confirmLabel?: string
  danger?: boolean
  onConfirm: () => void
}) {
  return (
    <AD.Root open={open} onOpenChange={onOpenChange}>
      <AD.Portal>
        <AD.Overlay className="fixed inset-0 z-[95] bg-black/70" />
        <AD.Content
          className="fixed left-1/2 top-1/2 z-[96] w-[min(460px,92vw)] -translate-x-1/2 -translate-y-1/2
                     rounded-xl border border-line bg-panel p-5 shadow-2xl"
        >
          <AD.Title className="text-[15px] font-semibold">{title}</AD.Title>
          {body ? (
            <AD.Description asChild>
              <div className="mt-2 text-[13px] leading-relaxed text-dim">{body}</div>
            </AD.Description>
          ) : null}
          <div className="mt-5 flex justify-end gap-2">
            <AD.Cancel className="sa-btn">Cancel</AD.Cancel>
            <AD.Action
              onClick={onConfirm}
              className={cn(
                'sa-btn',
                danger ? 'border-err text-danger hover:border-err' : 'sa-btn-sel',
              )}
            >
              {confirmLabel}
            </AD.Action>
          </div>
        </AD.Content>
      </AD.Portal>
    </AD.Root>
  )
}
