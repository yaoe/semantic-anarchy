import { cn } from '../../lib/utils'
import type { JobStatus } from '../../api/types'

/** Header idle/busy indicator. */
export function StatusDot({ busy }: { busy: boolean }) {
  return (
    <span
      className={cn(
        'inline-block h-2 w-2 rounded-full',
        busy ? 'bg-run animate-pulse' : 'bg-line2',
      )}
    />
  )
}

const BADGE: Record<JobStatus, string> = {
  running: 'bg-run text-black',
  done: 'bg-done text-white',
  error: 'bg-err text-white',
  queued: 'bg-line2 text-[#cfd3dc]',
  cancelled: 'bg-[#555] text-white',
}

export function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span
      className={cn(
        'rounded-full px-[7px] py-px text-[10px] uppercase tracking-[0.5px] whitespace-nowrap',
        BADGE[status] ?? 'bg-line2 text-ink',
      )}
    >
      {status}
    </span>
  )
}
