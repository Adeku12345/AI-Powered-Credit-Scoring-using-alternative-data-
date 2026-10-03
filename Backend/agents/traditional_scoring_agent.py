import sys

from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.ml_core.traditional_credit_score import TraditionalCreditScorer


class TraditionalScoringAgent:
    """
    Responsible for the transparent, rule-based scorecard score. No LLM
    call needed here — the breakdown is self-explaining by construction.
    """

    name = "traditional_scoring_agent"

    def __init__(self):
        self.scorer = TraditionalCreditScorer()

    def run(self, applicant: dict) -> dict:
        try:
            logging.info(f"[{self.name}] scoring applicant")
            result = self.scorer.score(applicant)
            explanation = self.scorer.explain(result)

            return {
                "agent": self.name,
                "score": result["score"],
                "breakdown": result["breakdown"],
                "narrative": explanation,
            }

        except Exception as e:
            raise CustomException(e, sys)
