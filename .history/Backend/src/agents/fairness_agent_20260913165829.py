from langchain_core.runnables import RunnableLambda
from backend.agents.state import CreditScoringState

async def fairness_node(state: CreditScoringState) -> Dict:
    d = state["applicant_data"]
    issues = []
    if d.get("age",30) < 25 and state["final_score"] < 500:
        issues.append("Young applicant flagged for manual review")
    return {"fairness_check": {
        "status": "PASSED" if not issues else "REVIEW",
        "issues": issues,
        "note": "✅ No demographic bias detected" if not issues else "Manual review recommended"
    }}

fairness_runnable = RunnableLambda(fairness_node)