import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from backend.core.exceptions import CustomException
from backend.core.logger import logging
from backend.core.utils import load_object


@dataclass
class ExplainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    background_sample_size: int = 100


class ModelExplainer:
    """
    Wraps the trained model + preprocessor to produce:
      1. SHAP feature attributions for a single applicant's prediction.
      2. A plain-English explanation of those attributions via the Claude API.
    """

    def __init__(self):
        try:
            self.config = ExplainerConfig()
            self.model = load_object(self.config.model_path)
            self.preprocessor = load_object(self.config.preprocessor_path)
            self._shap_explainer = None
        except Exception as e:
            raise CustomException(e, sys)

    def _load_background(self, target_column_name: str = "credit_score") -> np.ndarray:
        """
        Uses a random sample of the training set (transformed) as SHAP's
        background/reference distribution. A single row is not enough for
        non-tree models (e.g. linear, KNN), which need a baseline to
        compare against.
        """
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            train_df = train_df.drop(columns=[target_column_name])

            sample_size = min(self.config.background_sample_size, len(train_df))
            background_df = train_df.sample(sample_size, random_state=42)

            background_transformed = self.preprocessor.transform(background_df)
            if hasattr(background_transformed, "toarray"):
                background_transformed = background_transformed.toarray()

            return background_transformed

        except Exception as e:
            raise CustomException(e, sys)

    def _get_shap_explainer(self, background: np.ndarray):
        if self._shap_explainer is None:
            logging.info("Building SHAP explainer (this can be slow for non-tree models)")
            self._shap_explainer = shap.Explainer(self.model.predict, background)
        return self._shap_explainer

    def _feature_names(self):
        try:
            return list(self.preprocessor.get_feature_names_out())
        except Exception as e:
            raise CustomException(e, sys)

    def explain(self, features: pd.DataFrame, top_k: int = 6) -> dict:
        """
        Returns the prediction, the baseline (average) prediction over the
        background set, and the top_k features that pushed this applicant's
        score away from that baseline, ranked by absolute impact.
        """
        try:
            transformed = self.preprocessor.transform(features)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            background = self._load_background()
            explainer = self._get_shap_explainer(background)
            shap_values = explainer(transformed)

            feature_names = self._feature_names()
            values = shap_values.values[0]
            base_value = float(np.array(shap_values.base_values).reshape(-1)[0])
            prediction = float(self.model.predict(transformed)[0])

            contributions = sorted(
                zip(feature_names, values),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_k]

            return {
                "prediction": prediction,
                "base_value": base_value,
                "top_contributions": [
                    {"feature": name, "shap_value": float(val)} for name, val in contributions
                ],
            }

        except Exception as e:
            raise CustomException(e, sys)

    def generate_llm_explanation(self, explanation: dict, applicant_context: str = "") -> str:
        """
        Sends the SHAP attributions to Claude and asks for a short,
        jargon-free explanation of the score.
        """
        try:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise CustomException(
                    "ANTHROPIC_API_KEY environment variable is not set", sys
                )

            client = anthropic.Anthropic(api_key=api_key)

            contributions_text = "\n".join(
                f"- {c['feature']}: {'+' if c['shap_value'] >= 0 else ''}{c['shap_value']:.1f} points"
                for c in explanation["top_contributions"]
            )

            prompt = f"""You are explaining a credit score prediction from a machine learning model.

Predicted score: {explanation['prediction']:.0f}
Average score for a typical applicant in the training data: {explanation['base_value']:.0f}

The factors below show how much each one pushed this applicant's score up or down relative to that average (based on SHAP values from the model):
{contributions_text}

{applicant_context}

Write a short explanation (4-6 sentences) in plain English, suitable for a loan officer or the applicant themselves. Reference the 2-3 most influential factors using natural language (e.g. "a high credit card utilization rate" rather than the raw column name). Do not mention "SHAP", "feature contribution", or any modeling jargon. End with one constructive, specific suggestion for how the applicant could improve their score, if there is an obvious one."""

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )

            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from backend.ml_core.ml_scorer import CustomData

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

    explainer = ModelExplainer()
    explanation = explainer.explain(sample.get_data_as_dataframe())

    print(f"Predicted credit score: {explanation['prediction']:.0f}")
    print(f"Baseline (average) score: {explanation['base_value']:.0f}")
    print("Top contributing factors:")
    for c in explanation["top_contributions"]:
        print(f"  {c['feature']}: {c['shap_value']:+.1f}")

    narrative = explainer.generate_llm_explanation(explanation)
    print("\n--- LLM explanation ---")
    print(narrative)