import { useLabelFacets } from '../../api/queries'

/**
 * The way into the labeling page from the dashboard — a real link with
 * `target="_blank"`, so it opens as its own browser tab and stays open beside
 * the dashboard while the next batch renders.
 *
 * The badge is how much of the gallery is still unlabeled; it is the number
 * that decides whether it's worth opening.
 */
export function LabelLink() {
  const { data: facets } = useLabelFacets()
  const left = facets?.unlabeled ?? 0
  return (
    <a
      href="/label"
      target="_blank"
      rel="noopener"
      className="sa-btn sa-btn-sm no-underline border-accent/60 text-accent hover:border-accent"
      title="score images 0–9 — opens in a new tab. The labels dataset is what every later instrument is built on."
    >
      🏷 Label{left ? ` (${left})` : ''} ↗
    </a>
  )
}
