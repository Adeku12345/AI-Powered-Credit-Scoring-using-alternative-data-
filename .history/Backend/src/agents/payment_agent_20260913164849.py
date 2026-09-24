from langchain_core.runnables import RunnableLambda
from backend.agents.state import CreditScoringState
from ml.inference.predictor import predict_score
from backend.core.logging import setup_logger

logger = setup_logger("agent-payment")
BASE_WEIGHT = 0.35

def payment_agent_node(state: CreditScoringState) -> Dict:
    d = state["applicant_data"]
    f1 = d.get("on_time_pct_bureau", 75) / 100
    f2 = d.get("rent_on_time_pct", 85) / 100
    result = predict_score("payment", [f1, f2])
    prob = result["probability"] or (f1*0.7 + f2*0.3)
    score = round(prob * 100, 2)
    return {
        "agent_scores": {"payment": {
            "score": score, "weight": BASE_WEIGHT, "confidence": result["confidence"], "mode": result["mode"],
            "factors": [
                {"factor": "Bureau Payment Punctuality", "value": f"{d.get('on_time_pct_bureau')}%", "impact": round((f1-0.7)*BASE_WEIGHT*50,1)},
                {"factor": "Rent Payment Consistency", "value": f"{d.get('rent_on_time_pct')}%", "impact": round((f2-0.8)*BASE_WEIGHT*50,1)},
            ]
        }},
        "ml_probabilities": {"payment": prob}
    }

payment_runnable = RunnableLambda(payment_agent_node)