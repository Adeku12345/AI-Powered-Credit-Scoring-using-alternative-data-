import sys

from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.agents.base_agent import BaseAgent


class RiskReviewAgent(BaseAgent):
    """
    Looks across the applicant's profile and both scorers' outputs for
    things a point-based scorecard or a regression model wouldn't
    naturally surface on their own: large disagreement between the two
    scores, internally inconsistent data, or combinations of factors that
    are individually fine but jointly concerning (e.g. very high income
    claimed alongside very high utilization and a short credit history).
    This agent does not make the final call — it flags things for the
    Decision Agent (or a human underwriter) to weigh.
    """

    name = "risk_review_agent"

    def run(self, applicant: dict, ml_result: dict, traditional_result: dict) -> dict:
        try:
            logging.info(f"[{self.name}] reviewing applicant for risk flags")

            prompt = f"""You are a credit risk reviewer. Review this applicant's data and two independent credit scores, and flag anything a risk analyst should double-check before a lending decision is made.

Applicant profile:
{applicant}

ML model score: {ml_result['score']} (top factors: {ml_result['top_factors']})
Traditional scorecard score: {traditional_result['score']}

Consider:
- Do the two scores disagree significantly (more than ~75 points apart)?
- Is there any internal inconsistency in the data (e.g. very high salary with very short credit history and high existing loan count)?
- Any single factor that looks like an outlier or data quality issue worth a human check?

Respond with ONLY a JSON object with these exact keys:
- "risk_level": one of "low", "medium", "high"
- "flags": a list of short strings, each describing one specific thing to check (empty list if none)
- "summary": one sentence summarizing the overall risk picture

No preamble, no markdown, just the raw JSON object."""

            result = self.call_llm_json(prompt, max_tokens=400)
            result["agent"] = self.name
            return result

        except Exception as e:
            raise CustomException(e, sys)
