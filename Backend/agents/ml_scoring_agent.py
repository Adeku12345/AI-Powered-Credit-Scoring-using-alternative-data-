import sys

import pandas as pd

from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.ml_core.ml_scorer import ModelScorer
from backend.ml_core.explainer import ModelExplainer


class MLScoringAgent:
    """
    Responsible for exactly one thing: producing the ML model's prediction
    and its SHAP-based explanation for an applicant. Does not make any
    approve/decline judgment itself — that's the Decision Agent's job.
    """

    name = "ml_scoring_agent"

    def __init__(self):
        self.scorer = ModelScorer()
        self.explainer = ModelExplainer()

    def run(self, applicant: dict) -> dict:
        try:
            logging.info(f"[{self.name}] scoring applicant")
            features_df = pd.DataFrame([applicant])

            prediction = float(self.scorer.predict(features_df)[0])
            explanation = self.explainer.explain(features_df)
            narrative = self.explainer.generate_llm_explanation(explanation)

            return {
                "agent": self.name,
                "score": round(prediction),
                "base_value": round(explanation["base_value"]),
                "top_factors": explanation["top_contributions"],
                "narrative": narrative,
            }

        except Exception as e:
            raise CustomException(e, sys)
