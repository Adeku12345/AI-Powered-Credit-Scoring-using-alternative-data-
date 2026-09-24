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
        "credit_utilisation":0.20
    }
    
    def calculate(self,data:dict)->float:
        try:
            components = {k:data[k] * w for k, w in self.WEIGHTS.items()}
            self.score = sum(components.values()) * 1000
            self.factors = {k: round(v*1000,1) for k, v in components.items()}
            logger.info(f"Traditional score calculated: {self.score:.0f}")
            return round(self.score)
        except keyError as e:
            raise ScoringError(f"")       
    