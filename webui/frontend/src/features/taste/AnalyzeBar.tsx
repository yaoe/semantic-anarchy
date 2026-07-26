import { useResonance } from '../../api/queries'

/**
 * 🎯 Analyze — CLIP-embed new images, recompute novelty and retrain the taste
 * head from your ★ (scripts/resonance.py, via POST /api/resonance).
 */
export function AnalyzeBar() {
  const resonance = useResonance()
  return (
    <button
      className="sa-btn sa-btn-sm"
      title="embed new images, recompute novelty + retrain the taste model from your ★"
      disabled={resonance.isPending}
      onClick={() => resonance.mutate()}
    >
      {resonance.isPending ? 'analyzing…' : '🎯 Analyze'}
    </button>
  )
}
