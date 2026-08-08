import * as TG from '@radix-ui/react-toggle-group'

import { cn } from '../../lib/utils'
import type { Option } from '../../params/schema'

/** The legacy `.seg` button row: one exclusive choice, wrapping. */
export function Segmented({
  value,
  onChange,
  options,
  className,
  itemClassName,
  grow = true,
}: {
  value: string
  onChange: (v: string) => void
  options: Option[]
  className?: string
  itemClassName?: string
  grow?: boolean
}) {
  return (
    <TG.Root
      type="single"
      value={value}
      onValueChange={(v) => v && onChange(v)}
      className={cn('flex flex-wrap gap-1.5', className)}
    >
      {options.map((o) => (
        <TG.Item
          key={o.value}
          value={o.value}
          title={o.title}
          className={cn(
            'sa-btn text-[13px] px-2 py-[7px] whitespace-nowrap',
            grow && 'flex-1',
            value === o.value && 'sa-btn-sel',
            itemClassName,
          )}
        >
          {o.label}
        </TG.Item>
      ))}
    </TG.Root>
  )
}
