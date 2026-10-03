import { useState } from 'react'
import ApplicantForm from './components/ApplicantForm.jsx'
import StatementUpload from './components/StatementUpload.jsx'
import DecisionBanner from './components/DecisionBanner.jsx'
import ScorePanel from './components/ScorePanel.jsx'
import RiskPanel from './components/RiskPanel.jsx'
import { scoreApplicant, analyzeStatement } from './api/client.js'

export default function App() {
  const [mode, setMode] = useState('manual')
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleManualSubmit(applicant) {
    setLoading(true)
    setError(null)
    try {
      const result = await scoreApplicant(applicant)
      setReport(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleStatementSubmit(formData) {
    setLoading(true)
    setError(null)
    try {
      const result = await analyzeStatement(formData)
      setReport(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-ink-700 px-8 py-6">
        <h1 className="font-serif text-xl">Credit Risk Console</h1>
        <p className="text-xs text-muted mt-1">Multi-agent assessment — ML model, scorecard, and risk review in one pass</p>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-2 gap-10 px-8 py-10 max-w-6xl mx-auto">
        <section>
          <div className="flex gap-6 border-b border-ink-700 mb-6">
            <button
              onClick={() => setMode('manual')}
              className={`pb-2 text-sm ${mode === 'manual' ? 'text-paper border-b-2 border-brass' : 'text-muted'}`}
            >
              Manual entry
            </button>
            <button
              onClick={() => setMode('upload')}
              className={`pb-2 text-sm ${mode === 'upload' ? 'text-paper border-b-2 border-brass' : 'text-muted'}`}
            >
              Upload statement
            </button>
          </div>

          {mode === 'manual' ? (
            <ApplicantForm onSubmit={handleManualSubmit} loading={loading} />
          ) : (
            <StatementUpload onSubmit={handleStatementSubmit} loading={loading} />
          )}

          {error && (
            <p className="text-sm text-bad mt-4 border-l-2 border-bad pl-3">{error}</p>
          )}
        </section>

        <section>
          {!report && !loading && (
            <div className="text-sm text-muted border border-dashed border-ink-700 px-6 py-16 text-center">
              Run an assessment to see the ML score, scorecard, risk review, and final decision here.
            </div>
          )}

          {loading && (
            <div className="text-sm text-muted px-6 py-16 text-center">
              Running the multi-agent pipeline…
            </div>
          )}

          {report && !loading && (
            <div className="space-y-10">
              <DecisionBanner decision={report.final_decision} />
              <ScorePanel mlScoring={report.ml_scoring} traditionalScoring={report.traditional_scoring} />
              <RiskPanel risk={report.risk_review} />
              {report.customer_id && (
                <p className="text-xs text-muted">Saved as {report.customer_id} in the dataset.</p>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
