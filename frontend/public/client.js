const BASE_URL = 'http://localhost:8000'

async function handleResponse(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed with status ${res.status}`)
  }
  return res.json()
}

export async function scoreApplicant(applicant) {
  const res = await fetch(`${BASE_URL}/api/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(applicant),
  })
  return handleResponse(res)
}

export async function analyzeStatement(formData) {
  const res = await fetch(`${BASE_URL}/api/analyze-statement`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(res)
}
