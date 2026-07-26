import * as RS from '@radix-ui/react-select'

import { cn } from '../../lib/utils'
import type { Option } from '../../params/schema'

/**
 * Radix Select styled as the legacy `<select>`. Radix forbids an empty item
 * value, but several knobs use '' to mean "auto / off", so '' is mapped to a
 * sentinel at the boundary and mapped straight back on change.
 */
const EMPTY = '__empty__'

export function Select({
  value,
  onChange,
  options,
  className,
  title,
  placeholder,
  ariaLabel,
}: {
  value: string
  onChange: (v: string) => void
  options: Option[]
  className?: string
  title?: string
  placeholder?: string
  ariaLabel?: string
}) {
  return (
    <RS.Root
      value={value === '' ? EMPTY : value}
      onValueChange={(v) => onChange(v === EMPTY ? '' : v)}
    >
      <RS.Trigger
        aria-label={ariaLabel}
        title={title}
        className={cn(
          'sa-input flex items-center justify-between gap-2 text-left cursor-pointer',
          'data-[state=open]:border-line2',
          className,
        )}
      >
        <span className="truncate">
          <RS.Value placeholder={placeholder} />
        </span>
        <RS.Icon className="shrink-0 text-dim text-[10px]">▼</RS.Icon>
      </RS.Trigger>
      <RS.Portal>
        <RS.Content
          position="popper"
          sideOffset={4}
          className="z-[80] max-h-[60vh] min-w-[var(--radix-select-trigger-width)]
                     overflow-hidden rounded-lg border border-line bg-panel2 shadow-2xl"
        >
          <RS.Viewport className="p-1">
            {options.map((o) => (
              <RS.Item
                key={o.value || EMPTY}
                value={o.value === '' ? EMPTY : o.value}
                title={o.title}
                className="cursor-pointer select-none rounded-md px-2 py-[6px] text-[13px]
                           outline-none data-[highlighted]:bg-line
                           data-[state=checked]:text-accent"
              >
                <RS.ItemText>{o.label}</RS.ItemText>
              </RS.Item>
            ))}
          </RS.Viewport>
        </RS.Content>
      </RS.Portal>
    </RS.Root>
  )
}
