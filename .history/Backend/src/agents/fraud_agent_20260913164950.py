from langchain_core.runnables import RunnableLambda
from backend.agents.state import CreditScoringState
from ml.inference.predictor import detect_anomaly
from backend.core.logging import setup_logger

logger = setup_logger("agent-fraud")

def fraud_agent_node(state: CreditScoringState) -> Dict:
    d = state["applicant_data"]
    features = [
        d.get("on_time_pct_bureau",70)/100, d.get("rent_on_time_pct",85)/100,
        min(d.get("annual_salary_gbp",30000)/100000,1), min(d.get("outstanding_debt_gbp",5000)/50000,1),
        d.get("credit_card_utilization_pct",40)/100, min(d.get("months_credit_history",12)/240,1),
        min(d.get("defaults_count",0)/5,1), min(d.get("enquiries_last_6m",2)/10,1),
    ]
    result = detect_anomaly(features)
    flags = []
    if result["is_anomaly"]: flags.append(f"⚠️ Statistical anomaly ({result['anomaly_score']})")
    if d.get("enquiries_last_6m",0) > 8: flags.append("Excessive credit applications")
    if d.get("age",30) < 18: flags.append("Age below minimum")
    if d.get("annual_salary_gbp",0) > 200000 and d.get("months_credit_history",0) < 6:
        flags.append("Income/history mismatch")
    return {"fraud_flags": flags, "fraud_detected": len(flags) > 0}

fraud_runnable = RunnableLambda(fraud_agent_node)