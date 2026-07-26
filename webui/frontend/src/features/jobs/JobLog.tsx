import { useEffect, useRef } from 'react'

import type { JobSummary } from '../../api/types'
import { useUI } from '../../store'
import { useJobLog } from './useJobLog'
import { StatusBadge } from './StatusDot'

/**
 * Live stdout of the selected job. Sticks to the bottom while the user is
 * already at the bottom (same courtesy the legacy pane had), otherwise leaves
 * their scroll position alone. Collapsible — the stream keeps running while
 * folded, so re-opening shows the whole log, scrolled to the end.
 */
export function JobLog({ job }: { job: JobSummary | undefined }) {
  const { lines, streaming } = useJobLog(job?.id ?? null)
  const collapsed = useUI((s) => s.logCollapsed)
  const toggle = useUI((s) => s.toggleLog)
  const ref = useRef<HTMLPreElement>(null)
  const stick = useRef(true)

  useEffect(() => {
    const el = ref.current
    if (el && stick.current) el.scrollTop = el.scrollHeight
  }, [lines, collapsed])

  return (
    <section className="shrink-0">
      <div className="mb-2 flex items-baseline gap-3">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={!collapsed}
          title={collapsed ? 'show the job output' : 'hide the job output'}
          className="flex cursor-pointer items-baseline gap-1.5 border-0 bg-transparent p-0
                     text-dim hover:text-ink"
        >
          <span className="text-[10px] leading-none">{collapsed ? '▶' : '▼'}</span>
          <h2 className="sa-h2 !mt-0 !mb-0 text-inherit">
            Terminal output {collapsed ? '(hidden)' : ''}
          </h2>
        </button>
        {job ? (
          <>
            <span className="text-[13px] truncate">
              #{job.id} {job.label}
            </span>
            <StatusBadge status={job.status} />
            <span className="text-[10px] text-dim" title="live stream vs. polling fallback">
              {streaming ? 'streaming' : job.status === 'running' ? 'polling' : ''}
            </span>
          </>
        ) : null}
      </div>
      {collapsed ? null : (
        <pre
          ref={ref}
          onScroll={(e) => {
            const el = e.currentTarget
            stick.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 30
          }}
          className="m-0 h-[240px] overflow-auto rounded-lg border border-line bg-logbg p-3
                     font-mono text-[12px] leading-[1.5] whitespace-pre-wrap text-[#cfd3dc]"
        >
          {job == null
            ? 'select or start a job…'
            : lines.length
              ? lines.join('\n')
              : '(no output yet)'}
        </pre>
      )}
    </section>
  )
}
