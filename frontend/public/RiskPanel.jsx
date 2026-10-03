const levelColor = {
  low: 'text-good border-good',
  medium: 'text-warn border-warn',
  high: 'text-bad border-bad',
}

export default function RiskPanel({ risk }) {
  const colorClass = levelColor[risk.risk_level] || 'text-muted border-muted'

  return (
    <div>
      <div className="flex items-center justify-between border-b border-ink-600 pb-2">
        <span className="text-sm text-muted">Risk review</span>
        <span className={`text-xs border px-2 py-0.5 ${colorClass}`}>{risk.risk_level}</span>
      </div>
      <p className="text-sm text-paper mt-3 leading-relaxed">{risk.summary}</p>
      {risk.flags?.length > 0 && (
        <ul className="mt-3 space-y-1.5">
          {risk.flags.map((flag, i) => (
            <li key={i} className="text-xs text-muted border-b border-ink-700 pb-1.5">
              — {flag}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
