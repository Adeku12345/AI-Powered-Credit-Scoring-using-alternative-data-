from langchain_core.runnables import RunnableLambda
from backend.agents.state import CreditScoringState
from ml.inference.predictor import predict_score
from backend.core.logging import setup_logger

logger = setup_logger("agent-history")
BASE_WEIGHT = 0.20

def history_agent_node(state: CreditScoringState) -> Dict:
    d = state["applicant_data"]
    months = d.get("months_credit_history", 12)
    defaults = d.get("defaults_count", 0)
    enquiries = d.get("enquiries_last_6m", 2)
    f1, f2, f3 = min(months/240,1), min(defaults/5,1), min(enquiries/10,1)
    result = predict_score("history", [f1, f2, f3])
    prob = result["probability"] or max(0.1, min(months/60,1) - defaults*0.15 - enquiries*0.05)
    score = round(prob * 100, 2)
    return {
        "agent_scores": {"history": {
            "score": score, "weight": BASE_WEIGHT, "confidence": result["confidence"], "mode": result["mode"],
            "factors": [
                {"factor": "History Length", "value": f"{months} months", "impact": round((f1-0.25)*BASE_WEIGHT*150,1)},
                {"factor": "Past Defaults", "value": defaults, "impact": round(-defaults*BASE_WEIGHT*10,1)},
            ]
        }},
        "ml_probabilities": {"history": prob}
    }

history_runnable = RunnableLambda(history_agent_node)