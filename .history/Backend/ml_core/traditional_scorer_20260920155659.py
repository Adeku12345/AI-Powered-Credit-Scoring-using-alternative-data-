from backend.core.logging import logger
from backend.core.exceptions import ScoringError

class BaseScorer:
    """PHASE 1: Abstract base class--OOP principle"""
    def __init__(self):
        self.score = None
        self.factors = {}
        
    def calculate()    
    