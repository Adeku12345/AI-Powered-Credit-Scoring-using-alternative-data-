import sys
import json

from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.agents.ml_scoring_agent import MLScoringAgent
from backend.agents.traditional_scoring_agent import TraditionalScoringAgent
from backend.agents.risk_review_agent import RiskReviewAgent
from backend.agents.decision_agent import DecisionAgent


class CreditDecisionOrchestrator:
    """
    Coordinates the four specialized agents into one end-to-end credit
    decision pipeline:

      1. MLScoringAgent          -> model prediction + SHAP explanation
      2. TraditionalScoringAgent -> rule-based scorecard + breakdown
      3. RiskReviewAgent         -> cross-checks both scores + profile for red flags
      4. DecisionAgent           -> reconciles everything into Approve/Review/Decline

    Each agent only sees what it needs: the scoring agents see the raw
    applicant data, while the risk and decision agents see the scoring
    agents' outputs rather than re-deriving anything themselves. This
    keeps each agent's responsibility narrow and makes the pipeline easy
    to extend (e.g. add a FraudCheckAgent between steps 2 and 3 without
    touching the scoring agents at all).
    """

    def __init__(self):
        self.ml_agent = MLScoringAgent()
        self.traditional_agent = TraditionalScoringAgent()
        self.risk_agent = RiskReviewAgent()
        self.decision_agent = DecisionAgent()

    def run(self, applicant: dict) -> dict:
        try:
            logging.info("===== Multi-agent credit decision pipeline started =====")

            ml_result = self.ml_agent.run(applicant)
            logging.info(f"ML agent result: score={ml_result['score']}")

            traditional_result = self.traditional_agent.run(applicant)
            logging.info(f"Traditional agent result: score={traditional_result['score']}")

            risk_result = self.risk_agent.run(applicant, ml_result, traditional_result)
            logging.info(f"Risk agent result: level={risk_result.get('risk_level')}")

            decision_result = self.decision_agent.run(
                applicant, ml_result, traditional_result, risk_result
            )
            logging.info(f"Decision agent result: {decision_result.get('decision')}")

            logging.info("===== Multi-agent credit decision pipeline completed =====")

            return {
                "applicant": applicant,
                "ml_scoring": ml_result,
                "traditional_scoring": traditional_result,
                "risk_review": risk_result,
                "final_decision": decision_result,
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample_applicant = {
        "gender": "Female",
        "occupation": "Professional",
        "mortgage_status": "Current",
        "mobile_money_active": "True",
        "annual_salary_gbp": 42000,
        "credit_card_utilization_pct": 28.5,
        "on_time_payment_ratio_pct": 91.0,
        "rent_payment_monthly_gbp": 0,
        "rent_on_time_rate_pct": None,
        "utility_bills_on_time_pct": 95.0,
        "mobile_money_tx_monthly": 12,
        "mobile_money_age_days": 600,
        "preferred_loan_term_months": 36,
        "existing_loans_count": 1,
        "months_credit_history": 72,
    }

    orchestrator = CreditDecisionOrchestrator()
    report = orchestrator.run(sample_applicant)

    print(f"\n{'='*60}")
    print(f"ML Score: {report['ml_scoring']['score']}")
    print(f"Traditional Score: {report['traditional_scoring']['score']}")
    print(f"Risk Level: {report['risk_review']['risk_level']}")
    print(f"Risk Flags: {report['risk_review']['flags']}")
    print(f"\nFINAL DECISION: {report['final_decision']['decision']}")
    print(f"Confidence: {report['final_decision']['confidence']}")
    print(f"Rationale: {report['final_decision']['rationale']}")
    print(f"{'='*60}")
