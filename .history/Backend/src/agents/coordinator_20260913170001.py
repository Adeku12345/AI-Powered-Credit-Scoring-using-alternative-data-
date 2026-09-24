from langchain_core.runnables import RunnableLambda
from backend.agents.state import CreditScoringState
from backend.core.logging import setup_logger

logger = setup_logger("agent-coordinator")

def _grade(s): return "A+" if s>=780 else "A" if s>=720 else "B" if s>=660 else "C" if s>=600 else "D" if s>=520 else "F"
def _risk(s): return "Low Risk" if s>=720 else "Medium Risk" if s>=600 else "High Risk"

def coordinator_node(state: CreditScoringState) -> Dict:
    scores = state["agent_scores"]
    total_w = sum(s["weight"] for s in scores.values())
    base_100 = sum(s["score"]*s["weight"] for s in scores.values()) / total_w
    if state.get("fraud_flags"): base_100 = max(0, base_100 - len(state["fraud_flags"])*5)
    final = int(round(300 + (base_100/100)*550))
    final = max(300, min(850, final))
    breakdown = [
        {"agent": n.replace("_"," ").title()+" Agent", "score": round(d["score"]),
         "contribution": round(d["score"]*d["weight"]/base_100*100 if base_100 else 0),
         "mode": d.get("mode","fallback"), "factors": d.get("factors",[])}
        for n,d in scores.items()
    ]
    logger.info(f"✅ Score: {final}/850 | Grade: {_grade(final)}")
    return {
        "base_score_100": round(base_100,2), "final_score": final,
        "grade": _grade(final), "risk_category": _risk(final), "agent_breakdown": breakdown
    }

coordinator_runnable = RunnableLambda(coordinator_node)