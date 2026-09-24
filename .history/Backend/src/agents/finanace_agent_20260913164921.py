from langchain_core.runnables import RunnableLambda
from backend.agents.state import CreditScoringState
from ml.inference.predictor import predict_score
from backend.core.logging import setup_logger

logger = setup_logger("agent-finance")
BASE_WEIGHT = 0.30

def finance_agent_node(state: CreditScoringState) -> Dict:
    d = state["applicant_data"]
    salary = max(d.get("annual_salary_gbp", 30000), 1)
    debt = d.get("outstanding_debt_gbp", 5000)
    util = d.get("credit_card_utilization_pct", 40) / 100
    dti = min(debt / salary, 1.0)
    result = predict_score("finance", [salary/100000, dti, util])
    prob = result["probability"] or ((1-dti)*0.6 + (1-util)*0.4)
    score = round(prob * 100, 2)
    return {
        "agent_scores": {"finance": {
            "score": score, "weight": BASE_WEIGHT, "confidence": result["confidence"], "mode": result["mode"],
            "factors": [
                {"factor": "Debt-to-Income", "value": f"{round(dti*100)}%", "impact": round((0.35-dti)*BASE_WEIGHT*100,1)},
                {"factor": "Card Usage", "value": f"{util*100:.0f}%", "impact": round((0.3-util)*BASE_WEIGHT*80,1)},
            ]
        }},
        "ml_probabilities": {"finance": prob}
    }

finance_runnable = RunnableLambda(finance_agent_node)