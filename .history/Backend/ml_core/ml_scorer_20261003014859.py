import os
import sys
from dataclasses import dataclass

import pandas as pd

from backend.core.exceptions import CustomException
from backend.core.logger import logging
from backend.core.utils import load_object


@dataclass
class ModelScorerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class ModelScorer:
    """
    Loads the trained model and fitted preprocessor from artifacts/
    and scores new applicant data.
    """

    def __init__(self):
        self.config = ModelScorerConfig()

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading preprocessor and model for scoring")
            preprocessor = load_object(file_path=self.config.preprocessor_path)
            model = load_object(file_path=self.config.model_path)

            logging.info("Transforming input features")
            data_scaled = preprocessor.transform(features)

            logging.info("Generating predictions")
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Maps raw applicant inputs (e.g. from an API request or form) into the
    pandas DataFrame shape the preprocessor/model expect, in the same
    column layout used during training.
    """

    def __init__(
        self,
        gender: str,
        occupation: str,
        mortgage_status: str,
        mobile_money_active: bool,
        annual_salary_gbp: float,
        credit_card_utilization_pct: float,
        on_time_payment_ratio_pct: float,
        rent_payment_monthly_gbp: float,
        rent_on_time_rate_pct: float,
        utility_bills_on_time_pct: float,
        mobile_money_tx_monthly: int,
        mobile_money_age_days: int,
        preferred_loan_term_months: int,
        existing_loans_count: int,
        months_credit_history: int,
    ):
        self.gender = gender
        self.occupation = occupation
        self.mortgage_status = mortgage_status
        self.mobile_money_active = mobile_money_active
        self.annual_salary_gbp = annual_salary_gbp
        self.credit_card_utilization_pct = credit_card_utilization_pct
        self.on_time_payment_ratio_pct = on_time_payment_ratio_pct
        self.rent_payment_monthly_gbp = rent_payment_monthly_gbp
        self.rent_on_time_rate_pct = rent_on_time_rate_pct
        self.utility_bills_on_time_pct = utility_bills_on_time_pct
        self.mobile_money_tx_monthly = mobile_money_tx_monthly
        self.mobile_money_age_days = mobile_money_age_days
        self.preferred_loan_term_months = preferred_loan_term_months
        self.existing_loans_count = existing_loans_count
        self.months_credit_history = months_credit_history

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "occupation": [self.occupation],
                "mortgage_status": [self.mortgage_status],
                "mobile_money_active": ["mobil[str(self.mobile_money_active)],],
                "annual_salary_gbp": [self.annual_salary_gbp],
                "credit_card_utilization_pct": [self.credit_card_utilization_pct],
                "on_time_payment_ratio_pct": [self.on_time_payment_ratio_pct],
                "rent_payment_monthly_gbp": [self.rent_payment_monthly_gbp],
                "rent_on_time_rate_pct": [self.rent_on_time_rate_pct],
                "utility_bills_on_time_pct": [self.utility_bills_on_time_pct],
                "mobile_money_tx_monthly": [self.mobile_money_tx_monthly],
                "mobile_money_age_days": [self.mobile_money_age_days],
                "preferred_loan_term_months": [self.preferred_loan_term_months],
                "existing_loans_count": [self.existing_loans_count],
                "months_credit_history": [self.months_credit_history],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick manual smoke test using a plausible applicant profile.
    sample = CustomData(
        gender="Female",
        occupation="Professional",
        mortgage_status="Current",
        mobile_money_active=True,
        annual_salary_gbp=42000,
        credit_card_utilization_pct=28.5,
        on_time_payment_ratio_pct=91.0,
        rent_payment_monthly_gbp=0,
        rent_on_time_rate_pct=None,
        utility_bills_on_time_pct=95.0,
        mobile_money_tx_monthly=12,
        mobile_money_age_days=600,
        preferred_loan_term_months=36,
        existing_loans_count=1,
        months_credit_history=72,
    )

    scorer = ModelScorer()
    result = scorer.predict(sample.get_data_as_dataframe())
    print(f"Predicted credit score: {result[0]:.1f}")