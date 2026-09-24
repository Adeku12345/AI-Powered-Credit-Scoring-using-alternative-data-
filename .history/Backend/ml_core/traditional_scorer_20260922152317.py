from backend.core.logging import logger
from backend.core.exceptions import ScoringError

class BaseScorer:
    """PHASE 1: Abstract base class--OOP principle"""
    def __init__(self):
        self.score = None
        self.factors = {}
        
    def calculate(self, data: dict) -> float:
        raise NotImplementedError("Subclasses must implement calculate()")
    
    
class TraditionalScorer(BaseScorer):
    
    """Rule-based scoring - inheritance from BaseScorer""" 
    WEIGHTS = {
        "payment_history":0.35,
        "income_stability":0.25,
        "debt_ratio":0.20,
        "cerdit_utilisation":
    }       
    