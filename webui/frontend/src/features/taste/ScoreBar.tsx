import { useScore } from '../../api/queries'

/** Score all ▶ — aesthetic pass over every generated image (POST /api/score). */
export function ScoreBar() {
  const score = useScore()
  return (
    <button
      className="sa-btn sa-btn-sm"
      title="aesthetic-score all images"
      disabled={score.isPending}
      onClick={() => score.mutate()}
    >
      {score.isPending ? 'scoring…' : 'Score all ▶'}
    </button>
  )
}
