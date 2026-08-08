import { useJobState, useRun } from '../api/queries'
import { StopButton } from '../features/jobs/StopButton'
import { useUI } from '../store'
import { buildRunRequest, runButtonLabel } from './schema'

/**
 * The submit button, lifted out of ParamPanel so App can pin it to the top of
 * the sidebar — the form below it scrolls, this never does. It reads the same
 * Zustand form state, so it stays in sync wherever it is mounted.
 *
 * While a job is on the GPU a small Stop sits in the bottom-right of the same
 * pinned box, on the line below the button — sharing it with the queued/error
 * hint, which is short enough to never reach that corner.
 */
export function RunButton() {
  const values = useUI((s) => s.params)
  const { data: state } = useJobState()
  const run = useRun()

  const running = state?.jobs.find((j) => j.id === state.running)

  const hint = run.isError ? (
    <span className="text-err">{(run.error as Error).message}</span>
  ) : run.isSuccess ? (
    <span>queued: {run.data.label ?? `#${run.data.job_id}`}</span>
  ) : null

  return (
    <div>
      <button
        className="sa-btn sa-btn-run w-full py-[11px] font-semibold"
        onClick={() => run.mutate(buildRunRequest(values))}
        disabled={run.isPending}
      >
        {run.isPending ? 'submitting…' : runButtonLabel(values)}
      </button>
      {hint || running ? (
        <div className="mt-1.5 flex items-center gap-2">
          <div className="sa-hint mt-0 min-w-0 flex-1 truncate">{hint}</div>
          {running ? (
            <StopButton
              job={running}
              className="shrink-0 rounded-[5px] px-[6px] py-0 text-[10px] leading-[17px]"
            />
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
