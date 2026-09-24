from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add

class CreditScoringState(TypedDict):
    applicant_data: Dict[str, Any]
    agent_scores: Annotated[Dict[str, Dict[str, Any]], lambda x,y: {**x,**y}]
    ml_probabilities: Dict[str, Optional[float]]
    fraud_flags: List[str]
    fraud_detected: bool
    final_score: int
    grade: str
    risk_category: str
    base_score_100: float
    explanation: Dict[str, Any]
    fairness_check: Dict[str, Any]
    agent_breakdown: List[Dict[str, Any]]
    errors: List[str]
    timestamp: str