import sys

from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.agents.base_agent import BaseAgent


class DecisionAgent(BaseAgent):
    """
    The orchestrator's final step: reconciles the ML score, the
    traditional scorecard score, and the risk review into one
    recommendation. A hard rule routes clearly risky cases to manual
    review regardless of what the LLM concludes, so the system never
    silently auto-approves something the Risk Review Agent flagged as
    high risk; the LLM's job is to handle the nuanced middle ground and
    write the rationale.
    """

    name = "decision_agent"

    def run(self, applicant: dict, ml_result: dict, traditional_result: dict, risk_result: dict) -> dict:
        try:
            logging.info(f"[{self.name}] making final decision")

            score_gap = abs(ml_result["score"] - traditional_result["score"])

            # Hard rule: never let the LLM auto-approve a high-risk case.
            if risk_result.get("risk_level") == "high":
                forced_decision = "Manual Review"
            else:
                forced_decision = None

            prompt = f"""You are making a final lending recommendation by combining three independent inputs.

ML model score: {ml_result['score']} / 850
Traditional scorecard score: {traditional_result['score']} / 850
Score gap between the two: {score_gap} points

Risk review: level={risk_result.get('risk_level')}, flags={risk_result.get('flags')}, summary="{risk_result.get('summary')}"

Based on all of this, recommend one of: "Approve", "Manual Review", or "Decline".

Guidance:
- Scores above ~650 on both methods with low risk generally support Approve.
- Scores below ~500 on both methods, or high risk flags, generally support Decline or Manual Review.
- Significant disagreement between the two scores (over ~75 points) should push toward Manual Review rather than an automatic Approve or Decline, since it suggests the two methods see this applicant differently.

Respond with ONLY a JSON object with these exact keys:
- "decision": one of "Approve", "Manual Review", "Decline"
- "confidence": one of "low", "medium", "high"
- "rationale": 2-3 sentences explaining the recommendation in plain English, referencing the actual scores and any risk flags.

No preamble, no markdown, just the raw JSON object."""

            result = self.call_llm_json(prompt, max_tokens=400)

            if forced_decision and result.get("decision") != forced_decision:
                result["decision"] = forced_decision
                result["rationale"] = (
                    f"Overridden to {forced_decision} because the risk review flagged this applicant "
                    f"as high risk, regardless of score levels. Original model rationale: {result.get('rationale', '')}"
                )

            result["agent"] = self.name
            result["score_gap"] = score_gap
            return result

        except Exception as e:
            raise CustomException(e, sys)
