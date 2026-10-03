function FactorList({ items, getLabel, getPoints }) {
  return (
    <ul className="mt-3 space-y-1.5">
      {items.map((item, i) => {
        const points = getPoints(item)
        const positive = points >= 0
        return (
          <li key={i} className="flex items-center justify-between text-xs border-b border-ink-700 pb-1.5">
            <span className="text-muted">{getLabel(item)}</span>
            <span className={`font-tabular ${positive ? 'text-good' : 'text-bad'}`}>
              {positive ? '+' : ''}{typeof points === 'number' ? points.toFixed(1) : points}
            </span>
          </li>
        )
      })}
    </ul>
  )
}

export default function ScorePanel({ mlScoring, traditionalScoring }) {
  return (
    <div className="grid grid-cols-2 gap-8">
      <div>
        <div className="flex items-baseline justify-between border-b border-ink-600 pb-2">
          <span className="text-sm text-muted">ML model</span>
          <span className="font-serif text-3xl font-tabular text-brass">{mlScoring.score}</span>
        </div>
        <p className="text-xs text-muted mt-2">baseline: {mlScoring.base_value}</p>
        <FactorList
          items={mlScoring.top_factors}
          getLabel={(f) => f.feature.replace(/^(num_pipeline__|cat_pipelines__)/, '').replace(/_/g, ' ')}
          getPoints={(f) => f.shap_value}
        />
      </div>

      <div>
        <div className="flex items-baseline justify-between border-b border-ink-600 pb-2">
          <span className="text-sm text-muted">Traditional scorecard</span>
          <span className="font-serif text-3xl font-tabular text-brass">{traditionalScoring.score}</span>
        </div>
        <p className="text-xs text-muted mt-2">rule-based, fully auditable</p>
        <FactorList
          items={traditionalScoring.breakdown.filter((b) => b.points !== 0).slice(0, 6)}
          getLabel={(b) => b.reason}
          getPoints={(b) => b.points}
        />
      </div>
    </div>
  )
}
