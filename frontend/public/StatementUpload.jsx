import { useState } from 'react'

export default function StatementUpload({ onSubmit, loading }) {
  const [file, setFile] = useState(null)
  const [known, setKnown] = useState({
    gender: '',
    occupation: '',
    mortgage_status: '',
    existing_loans_count: '',
    months_credit_history: '',
    credit_card_utilization_pct: '',
    mobile_money_age_days: '',
  })

  function handleSubmit(e) {
    e.preventDefault()
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)
    Object.entries(known).forEach(([key, value]) => {
      if (value !== '') formData.append(key, value)
    })
    onSubmit(formData)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <h3 className="text-sm text-muted mb-3">Statement</h3>
        <label className="flex items-center justify-center border border-dashed border-ink-600 px-4 py-8 text-sm text-muted cursor-pointer hover:border-brass transition-colors">
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="hidden"
          />
          {file ? file.name : 'Click to choose a PDF bank statement'}
        </label>
      </div>

      <div>
        <h3 className="text-sm text-muted mb-3">
          Known fields <span className="text-ink-600">— optional, fills in what the statement can't show</span>
        </h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-3">
          <label className="block">
            <span className="block text-xs text-muted mb-1">Gender</span>
            <input
              value={known.gender}
              onChange={(e) => setKnown({ ...known, gender: e.target.value })}
              className="w-full bg-ink-700 border border-ink-600 px-2 py-1.5 text-sm text-paper focus:outline-none focus:border-brass"
            />
          </label>
          <label className="block">
            <span className="block text-xs text-muted mb-1">Occupation</span>
            <input
              value={known.occupation}
              onChange={(e) => setKnown({ ...known, occupation: e.target.value })}
              className="w-full bg-ink-700 border border-ink-600 px-2 py-1.5 text-sm text-paper focus:outline-none focus:border-brass"
            />
          </label>
          <label className="block">
            <span className="block text-xs text-muted mb-1">Mortgage status</span>
            <input
              value={known.mortgage_status}
              onChange={(e) => setKnown({ ...known, mortgage_status: e.target.value })}
              className="w-full bg-ink-700 border border-ink-600 px-2 py-1.5 text-sm text-paper focus:outline-none focus:border-brass"
            />
          </label>
          <label className="block">
            <span className="block text-xs text-muted mb-1">Existing loans</span>
            <input
              type="number"
              value={known.existing_loans_count}
              onChange={(e) => setKnown({ ...known, existing_loans_count: e.target.value })}
              className="w-full bg-ink-700 border border-ink-600 px-2 py-1.5 text-sm text-paper font-tabular focus:outline-none focus:border-brass"
            />
          </label>
          <label className="block">
            <span className="block text-xs text-muted mb-1">Credit history (months)</span>
            <input
              type="number"
              value={known.months_credit_history}
              onChange={(e) => setKnown({ ...known, months_credit_history: e.target.value })}
              className="w-full bg-ink-700 border border-ink-600 px-2 py-1.5 text-sm text-paper font-tabular focus:outline-none focus:border-brass"
            />
          </label>
          <label className="block">
            <span className="block text-xs text-muted mb-1">Card utilization (%)</span>
            <input
              type="number"
              value={known.credit_card_utilization_pct}
              onChange={(e) => setKnown({ ...known, credit_card_utilization_pct: e.target.value })}
              className="w-full bg-ink-700 border border-ink-600 px-2 py-1.5 text-sm text-paper font-tabular focus:outline-none focus:border-brass"
            />
          </label>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading || !file}
        className="w-full bg-brass text-ink-900 font-medium py-2.5 text-sm disabled:opacity-50"
      >
        {loading ? 'Analyzing statement…' : 'Analyze statement'}
      </button>
    </form>
  )
}
