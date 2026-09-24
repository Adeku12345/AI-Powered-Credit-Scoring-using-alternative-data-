from langchain_core.runnables import RunnableLambda
from backend.agents.state import CreditScoringState
from ml.inference.predictor import predict_score
from backend.core.logging import setup_logger

logger = setup_logger("agent-alternative")
BASE_WEIGHT = 0.15

def alternative_agent_node(state: CreditScoringState) -> Dict:
    d = state["applicant_data"]
    rent_pct = d.get("rent_on_time_pct", 85) / 100
    age = min(d.get("age", 30) / 65, 1.0)
    history = d.get("months_credit_history", 0)
    result = predict_score("alternative", [rent_pct, age])
    prob = result["probability"] or (rent_pct*0.8 + age*0.2)
    score = round(prob * 100, 2)
    weight = BASE_WEIGHT if history >= 12 else 0.35
    if weight == 0.35: logger.info("📱 Thin file → Alternative weight boosted")
    return {
        "agent_scores": {"alternative": {
            "score": score, "weight": weight, "confidence": result["confidence"], "mode": result["mode"],
            "factors": [{"factor": "Rent Consistency", "value": f"{rent_pct*100:.0f}%", "impact": round((rent_pct-0.8)*weight*80,1)}]
        }},
        "ml_probabilities": {"alternative": prob}
    }

alternative_runnable = RunnableLambda(alternative_agent_node)