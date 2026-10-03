import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from backend.core.exceptions import CustomException
from backend.core.logger import logging


@dataclass
class TraditionalScorerConfig:
    min_score: int = 300
    max_score: int = 850
    base_score: int = 500  # starting point before adding/subtracting rule points


class TraditionalCreditScorer:
    """
    A classic point-based credit scorecard: each feature is binned into
    ranges, each range carries a fixed point value (positive or negative),
    and the final score is the base score plus the sum of all points,
    clamped to [min_score, max_score].

    Unlike the ML model, every point is traceable to an explicit rule, so
    this needs no SHAP or LLM to explain itself — the breakdown returned
    by score() IS the explanation. This makes it useful as:
      - A transparent baseline to sanity-check the ML model against
      - A fallback scorer if the ML model or its artifacts are unavailable
      - A human-auditable reference for regulatory/compliance review
    """

    def __init__(self):
        self.config = TraditionalScorerConfig()

    def _score_annual_salary(self, value) -> (int, str):
        if pd.isna(value):
            return 0, "Salary not provided (no points applied)"
        if value < 15000:
            return -40, f"Salary under GBP 15,000 ({value:,.0f})"
        elif value < 25000:
            return -10, f"Salary GBP 15,000-25,000 ({value:,.0f})"
        elif value < 40000:
            return 20, f"Salary GBP 25,000-40,000 ({value:,.0f})"
        elif value < 60000:
            return 45, f"Salary GBP 40,000-60,000 ({value:,.0f})"
        else:
            return 65, f"Salary over GBP 60,000 ({value:,.0f})"

    def _score_credit_utilization(self, value) -> (int, str):
        if pd.isna(value):
            return 0, "Credit utilization not provided (no points applied)"
        if value < 10:
            return 60, f"Very low utilization ({value:.1f}%)"
        elif value < 30:
            return 40, f"Low utilization ({value:.1f}%)"
        elif value < 50:
            return 10, f"Moderate utilization ({value:.1f}%)"
        elif value < 75:
            return -30, f"High utilization ({value:.1f}%)"
        else:
            return -60, f"Very high utilization ({value:.1f}%)"

    def _score_on_time_payment_ratio(self, value) -> (int, str):
        if pd.isna(value):
            return 0, "Payment history not provided (no points applied)"
        if value >= 98:
            return 80, f"Excellent payment history ({value:.1f}%)"
        elif value >= 90:
            return 50, f"Good payment history ({value:.1f}%)"
        elif value >= 75:
            return 10, f"Fair payment history ({value:.1f}%)"
        elif value >= 50:
            return -40, f"Poor payment history ({value:.1f}%)"
        else:
            return -80, f"Very poor payment history ({value:.1f}%)"

    def _score_rent_on_time_rate(self, value) -> (int, str):
        if pd.isna(value):
            return 0, "Rent payment history not provided (no points applied)"
        if value >= 95:
            return 25, f"Rent almost always on time ({value:.1f}%)"
        elif value >= 80:
            return 10, f"Rent usually on time ({value:.1f}%)"
        elif value >= 60:
            return -10, f"Rent often late ({value:.1f}%)"
        else:
            return -25, f"Rent frequently late ({value:.1f}%)"

    def _score_utility_bills_on_time(self, value) -> (int, str):
        if pd.isna(value):
            return 0, "Utility payment history not provided (no points applied)"
        if value >= 95:
            return 20, f"Utilities almost always on time ({value:.1f}%)"
        elif value >= 80:
            return 8, f"Utilities usually on time ({value:.1f}%)"
        elif value >= 60:
            return -8, f"Utilities often late ({value:.1f}%)"
        else:
            return -20, f"Utilities frequently late ({value:.1f}%)"

    def _score_months_credit_history(self, value) -> (int, str):
        if pd.isna(value):
            return 0, "Credit history length not provided (no points applied)"
        if value < 12:
            return -30, f"Short credit history ({value:.0f} months)"
        elif value < 36:
            return 0, f"Moderate credit history ({value:.0f} months)"
        elif value < 84:
            return 25, f"Established credit history ({value:.0f} months)"
        else:
            return 45, f"Long credit history ({value:.0f} months)"

    def _score_existing_loans(self, value) -> (int, str):
        if pd.isna(value):
            return 0, "Existing loan count not provided (no points applied)"
        if value == 0:
            return 15, "No existing loans"
        elif value <= 2:
            return 5, f"{value:.0f} existing loan(s)"
        elif value <= 4:
            return -20, f"{value:.0f} existing loans"
        else:
            return -45, f"{value:.0f}+ existing loans (high debt load)"

    def _score_mortgage_status(self, value) -> (int, str):
        if pd.isna(value) or value is None:
            return 0, "Mortgage status not provided (no points applied)"
        value = str(value)
        if value == "Arrears":
            return -50, "Mortgage in arrears"
        elif value == "Current":
            return 15, "Mortgage current / in good standing"
        else:
            return 0, "No mortgage"

    def _score_mobile_money_activity(self, active, age_days) -> (int, str):
        if pd.isna(active) or active in (None, "nan"):
            return 0, "Mobile money activity not provided (no points applied)"
        is_active = str(active) == "True"
        if not is_active:
            return 0, "No mobile money activity (neutral)"
        if pd.isna(age_days):
            return 5, "Mobile money active (account age unknown)"
        if age_days >= 365:
            return 15, f"Mobile money active for over a year ({age_days:.0f} days)"
        else:
            return 5, f"Mobile money active, newer account ({age_days:.0f} days)"

    def score(self, applicant: dict) -> dict:
        """
        Computes the traditional scorecard score for a single applicant.
        `applicant` should be a dict (or pandas Series converted to dict)
        with the same feature names used elsewhere in the project.
        Missing features are treated as neutral (0 points), not penalized,
        since the point of this scorer is to be auditable, not punitive
        about incomplete data.
        """
        try:
            breakdown = []
            total_points = 0

            rules = [
                ("annual_salary_gbp", self._score_annual_salary(applicant.get("annual_salary_gbp"))),
                ("credit_card_utilization_pct", self._score_credit_utilization(applicant.get("credit_card_utilization_pct"))),
                ("on_time_payment_ratio_pct", self._score_on_time_payment_ratio(applicant.get("on_time_payment_ratio_pct"))),
                ("rent_on_time_rate_pct", self._score_rent_on_time_rate(applicant.get("rent_on_time_rate_pct"))),
                ("utility_bills_on_time_pct", self._score_utility_bills_on_time(applicant.get("utility_bills_on_time_pct"))),
                ("months_credit_history", self._score_months_credit_history(applicant.get("months_credit_history"))),
                ("existing_loans_count", self._score_existing_loans(applicant.get("existing_loans_count"))),
                ("mortgage_status", self._score_mortgage_status(applicant.get("mortgage_status"))),
                ("mobile_money_activity", self._score_mobile_money_activity(
                    applicant.get("mobile_money_active"), applicant.get("mobile_money_age_days")
                )),
            ]

            for feature_name, (points, reason) in rules:
                breakdown.append({"feature": feature_name, "points": points, "reason": reason})
                total_points += points

            raw_score = self.config.base_score + total_points
            final_score = max(self.config.min_score, min(self.config.max_score, raw_score))

            logging.info(f"Traditional scorecard score: {final_score} (raw: {raw_score})")

            return {
                "score": final_score,
                "base_score": self.config.base_score,
                "total_points": total_points,
                "clamped": raw_score != final_score,
                "breakdown": sorted(breakdown, key=lambda x: abs(x["points"]), reverse=True),
            }

        except Exception as e:
            raise CustomException(e, sys)

    def explain(self, result: dict) -> str:
        """
        Turns the breakdown into a short, human-readable summary —
        no LLM call needed, since every line is already a plain-English
        reason paired with a point value.
        """
        try:
            lines = [f"Traditional scorecard result: {result['score']} "
                     f"(base {result['base_score']} {'+' if result['total_points'] >= 0 else ''}"
                     f"{result['total_points']} points)"]

            if result["clamped"]:
                lines.append(f"Note: raw score was clamped to the "
                              f"[{self.config.min_score}, {self.config.max_score}] range.")

            lines.append("\nContributing factors (largest impact first):")
            for item in result["breakdown"]:
                if item["points"] == 0:
                    continue
                sign = "+" if item["points"] >= 0 else ""
                lines.append(f"  {sign}{item['points']:>4} pts — {item['reason']}")

            return "\n".join(lines)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    sample_applicant = {
        "annual_salary_gbp": 42000,
        "credit_card_utilization_pct": 28.5,
        "on_time_payment_ratio_pct": 91.0,
        "rent_on_time_rate_pct": None,
        "utility_bills_on_time_pct": 95.0,
        "mortgage_status": "Current",
        "mobile_money_active": "True",
        "mobile_money_age_days": 600,
        "preferred_loan_term_months": 36,
        "existing_loans_count": 1,
        "months_credit_history": 72,
    }

    scorer = TraditionalCreditScorer()
    result = scorer.score(sample_applicant)

    print(scorer.explain(result))
