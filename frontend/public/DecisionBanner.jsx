const decisionColor = {
  Approve: 'border-good text-good',
  'Manual Review': 'border-warn text-warn',
  Decline: 'border-bad text-bad',
}

export default function DecisionBanner({ decision }) {
  const colorClass = decisionColor[decision.decision] || 'border-muted text-muted'

  return (
    <div className={`border-l-2 pl-4 py-1 ${colorClass}`}>
      <div className="flex items-baseline justify-between">
        <span className="font-serif text-2xl">{decision.decision}</span>
        <span className="text-xs text-muted">confidence: {decision.confidence}</span>
      </div>
      <p className="text-sm text-paper mt-2 leading-relaxed">{decision.rationale}</p>
      {decision.score_gap !== undefined && (
        <p className="text-xs text-muted mt-2">Score gap between models: {decision.score_gap} points</p>
      )}
    </div>
  )
}
