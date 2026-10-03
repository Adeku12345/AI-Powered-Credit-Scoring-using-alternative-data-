import { useState } from 'react'

const initialState = {
  gender: 'Female',
  occupation: 'Professional',
  mortgage_status: 'Current',
  mobile_money_active: 'True',
  annual_salary_gbp: 42000,
  credit_card_utilization_pct: 28.5,
  on_time_payment_ratio_pct: 91,
  rent_payment_monthly_gbp: 0,
  rent_on_time_rate_pct: '',
  utility_bills_on_time_pct: 95,
  mobile_money_tx_monthly: 12,
  mobile_money_age_days: 600,
  preferred_loan_term_months: 36,
  existing_loans_count: 1,
  months_credit_history: 72,
}

const fieldGroups = [
  {
    title: 'Applicant',
    fields: [
      { key: 'gender', label: 'Gender', type: 'select', options: ['Female', 'Male', 'Other'] },
      {
        key: 'occupation', label: 'Occupation', type: 'select',
        options: ['Professional', 'Clerical', 'Skilled', 'Self-Employed', 'Unskilled', 'Unemployed'],
      },
    ],
  },
  {
    title: 'Income & credit',
    fields: [
      { key: 'annual_salary_gbp', label: 'Annual salary (GBP)', type: 'number' },
      { key: 'credit_card_utilization_pct', label: 'Card utilization (%)', type: 'number' },
      { key: 'on_time_payment_ratio_pct', label: 'On-time payment ratio (%)', type: 'number' },
      { key: 'months_credit_history', label: 'Credit history (months)', type: 'number' },
      { key: 'existing_loans_count', label: 'Existing loans', type: 'number' },
      { key: 'preferred_loan_term_months', label: 'Preferred loan term (months)', type: 'number' },
    ],
  },
  {
    title: 'Housing',
    fields: [
      {
        key: 'mortgage_status', label: 'Mortgage status', type: 'select',
        options: ['None', 'Current', 'Arrears'],
      },
      { key: 'rent_payment_monthly_gbp', label: 'Monthly rent (GBP)', type: 'number' },
      { key: 'rent_on_time_rate_pct', label: 'Rent on-time rate (%)', type: 'number' },
      { key: 'utility_bills_on_time_pct', label: 'Utilities on-time rate (%)', type: 'number' },
    ],
  },
  {
    title: 'Mobile money',
    fields: [
      { key: 'mobile_money_active', label: 'Active', type: 'select', options: ['True', 'False'] },
      { key: 'mobile_money_tx_monthly', label: 'Transactions / month', type: 'number' },
      { key: 'mobile_money_age_days', label: 'Account age (days)', type: 'number' },
    ],
  },
]

export default function ApplicantForm({ onSubmit, loading }) {
  const [form, setForm] = useState(initialState)

  function update(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    const payload = { ...form }
    Object.keys(payload).forEach((k) => {
      if (payload[k] === '') payload[k] = null
    })
    onSubmit(payload)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {fieldGroups.map((group) => (
        <div key={group.title}>
          <h3 className="text-sm text-muted mb-3">{group.title}</h3>
          <div className="grid grid-cols-2 gap-x-4 gap-y-3">
            {group.fields.map((field) => (
              <label key={field.key} className="block">
                <span className="block text-xs text-muted mb-1">{field.label}</span>
                {field.type === 'select' ? (
                  <select
                    value={form[field.key]}
                    onChange={(e) => update(field.key, e.target.value)}
                    className="w-full bg-ink-700 border border-ink-600 px-2 py-1.5 text-sm text-paper focus:outline-none focus:border-brass"
                  >
                    {field.options.map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    step="any"
                    value={form[field.key]}
                    onChange={(e) => update(field.key, e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full bg-ink-700 border border-ink-600 px-2 py-1.5 text-sm text-paper font-tabular focus:outline-none focus:border-brass"
                  />
                )}
              </label>
            ))}
          </div>
        </div>
      ))}

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-brass text-ink-900 font-medium py-2.5 text-sm disabled:opacity-50"
      >
        {loading ? 'Running assessment…' : 'Run assessment'}
      </button>
    </form>
  )
}
