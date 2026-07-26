import type { ReactElement, ReactNode } from 'react'
import * as Tooltip from '@radix-ui/react-tooltip'

/** Wrap any element to give it a styled tooltip (replaces `title=` on labels). */
export function Tip({
  content,
  children,
  side = 'right',
}: {
  content?: ReactNode
  children: ReactElement
  side?: 'top' | 'right' | 'bottom' | 'left'
}) {
  if (!content) return children
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          side={side}
          sideOffset={6}
          collisionPadding={8}
          className="z-[90] max-w-[280px] rounded-md border border-line bg-panel2 px-2.5 py-1.5
                     text-[12px] leading-snug text-ink shadow-xl"
        >
          {content}
          <Tooltip.Arrow className="fill-panel2" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}
