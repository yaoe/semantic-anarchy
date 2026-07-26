import { useCancel } from '../../api/queries'
import type { JobSummary } from '../../api/types'
import { cn } from '../../lib/utils'

/**
 * Stop whatever is on the GPU right now. Lives in the bottom-right of the
 * pinned Run box, so it is on screen while a job runs and nowhere otherwise.
 *
 * One click, no confirmation. The server SIGTERMs that one subprocess (SIGKILL
 * if it doesn't go), so the dashboard stays up and the queue moves on to the
 * next job. Whatever images the job already wrote stay on disk.
 */
export function StopButton({ job, className }: { job: JobSummary; className?: string }) {
  const cancel = useCancel()

  return (
    <button
      className={cn('sa-btn sa-btn-stop', className)}
      disabled={cancel.isPending}
      title={`stop job #${job.id} — the dashboard and the rest of the queue keep running`}
      onClick={() => cancel.mutate(job.id)}
    >
      {cancel.isPending ? 'stopping…' : '■ stop'}
    </button>
  )
}
