import { useCancel } from '../../api/queries'
import type { JobSummary } from '../../api/types'
import { cn, fmtDuration } from '../../lib/utils'
import { useUI } from '../../store'
import { StatusBadge } from './StatusDot'

/** The sidebar queue: 12 most recent jobs, newest first (server order). */
export function JobList({ jobs }: { jobs: JobSummary[] }) {
  const selectedJob = useUI((s) => s.selectedJob)
  const setSelectedJob = useUI((s) => s.setSelectedJob)
  const cancel = useCancel()

  if (!jobs.length)
    return <div className="sa-hint">no jobs yet — hit Run.</div>

  return (
    <div className="flex flex-col gap-1.5">
      {jobs.slice(0, 12).map((j) => {
        const active = j.id === selectedJob
        const cancellable = j.status === 'queued' || j.status === 'running'
        return (
          <div
            key={j.id}
            onClick={() => setSelectedJob(j.id)}
            title={j.cmd}
            className={cn(
              'sa-panel cursor-pointer px-2.5 py-2 transition-colors hover:border-line2',
              active && 'border-accent hover:border-accent',
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <span className="text-[12px] leading-snug break-words min-w-0">
                #{j.id} {j.label}
              </span>
              <StatusBadge status={j.status} />
            </div>
            <div className="mt-1 flex items-center gap-2 text-[10px] text-dim">
              <span>{j.lines} lines</span>
              {j.started ? <span>{fmtDuration(j.started, j.ended)}</span> : null}
              {j.rc != null && j.rc !== 0 ? <span className="text-err">rc {j.rc}</span> : null}
              {cancellable ? (
                <button
                  className="ml-auto text-[10px] text-danger hover:underline"
                  onClick={(e) => {
                    e.stopPropagation()
                    cancel.mutate(j.id)
                  }}
                >
                  cancel
                </button>
              ) : null}
            </div>
          </div>
        )
      })}
    </div>
  )
}
