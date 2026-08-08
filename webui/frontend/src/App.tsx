import { useEffect } from 'react'

import { useJobState, useRefreshOnJobFinish } from './api/queries'
import type { TabKey } from './api/types'
import { Gallery } from './features/gallery/Gallery'
import { Filters } from './features/gallery/Filters'
import { JobList } from './features/jobs/JobList'
import { ModelPicker } from './features/model/ModelPicker'
import { JobLog } from './features/jobs/JobLog'
import { StatusDot } from './features/jobs/StatusDot'
import { ExploreBar } from './features/refine/ExploreBar'
import { RefineBar } from './features/refine/RefineBar'
import { AnalyzeBar } from './features/taste/AnalyzeBar'
import { EvolveBar } from './features/taste/EvolveBar'
import { ScoreBar } from './features/taste/ScoreBar'
import { WipeDialog } from './features/taste/WipeDialog'
import { ParamPanel } from './params/ParamPanel'
import { RunButton } from './params/RunButton'
import { cn, fmtDuration } from './lib/utils'
import { useUI } from './store'

/** Tabs whose items carry conditioning, so the refine/explore bar applies. */
const ACTIONABLE: TabKey[] = ['generated', 'favorites', 'top', 'frontier']

export default function App() {
  const { data: state } = useJobState()
  const selectedJob = useUI((s) => s.selectedJob)
  const setSelectedJob = useUI((s) => s.setSelectedJob)
  const tab = useUI((s) => s.tab)

  const jobs = state?.jobs ?? []
  const busy = state?.running != null
  useRefreshOnJobFinish(state?.running)

  // Nothing selected yet -> follow the newest job, as the legacy page did.
  useEffect(() => {
    if (selectedJob == null && jobs.length) setSelectedJob(jobs[0].id)
  }, [selectedJob, jobs, setSelectedJob])

  const job = jobs.find((j) => j.id === selectedJob)

  const running = busy ? jobs.find((j) => j.id === state?.running) : undefined

  return (
    <div className="flex h-full flex-col">
      <header
        className={cn(
          'relative flex shrink-0 items-baseline gap-3.5 border-b px-5 py-3.5',
          'transition-colors duration-500',
          busy ? 'border-run/40 bg-run/[0.08]' : 'border-line',
        )}
      >
        <h1 className="m-0 text-[17px] tracking-[0.3px]">Semantic&nbsp;Anarchy</h1>
        <span className="text-[12px] text-dim">
          promptless explorer · {window.location.host}
        </span>
        <a
          href="/legacy"
          className="text-[11px] text-dim underline decoration-dotted hover:text-ink"
          title="the previous inline dashboard"
        >
          legacy UI
        </a>
        <span
          className={cn(
            'ml-auto flex min-w-0 items-center gap-2 text-[12px]',
            busy ? 'text-run' : 'text-dim',
          )}
        >
          <StatusDot busy={busy} />
          {busy ? (
            <>
              <span className="truncate">
                running #{state?.running}
                {running?.label ? ` · ${running.label}` : ''}
              </span>
              {running?.started ? (
                <span className="tabular-nums opacity-70">
                  {fmtDuration(running.started, running.ended)}
                </span>
              ) : null}
            </>
          ) : (
            'idle'
          )}
        </span>
        {/* Sweeping underline: motion in the chrome says "GPU is busy" from
            across the room, without touching the gallery's colours. */}
        {busy ? <span className="sa-runline" aria-hidden /> : null}
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-[320px_1fr]">
        <aside className="flex min-h-0 flex-col border-r border-line">
          <div
            className={cn(
              'shrink-0 border-b px-4 py-3 transition-colors duration-500',
              busy ? 'border-run/40 bg-run/[0.06]' : 'border-line bg-panel/40',
            )}
          >
            <RunButton />
          </div>
          {/* overflow-x-clip: `overflow-y-auto` alone computes overflow-x to
              auto, so any stray wide child would add a sideways scrollbar. The
              sidebar scrolls vertically only; its content wraps to fit. */}
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-clip p-4 pt-1">
            <ParamPanel after={{ model: <ModelPicker /> }} />
            <h2 className="sa-h2">Jobs</h2>
            <JobList jobs={jobs} />
          </div>
        </aside>

        <main className="flex min-h-0 flex-col gap-3 p-4 px-5">
          <JobLog job={job} />

          <Filters
            actions={
              <>
                <EvolveBar />
                <AnalyzeBar />
                <ScoreBar />
                <WipeDialog />
              </>
            }
          />

          {ACTIONABLE.includes(tab) ? (
            <div className="sa-panel flex shrink-0 flex-wrap items-end gap-2.5 px-3 py-2.5">
              <RefineBar />
              <div className="self-stretch border-l border-line" />
              <ExploreBar />
            </div>
          ) : null}

          <Gallery busy={busy} />
        </main>
      </div>
    </div>
  )
}
