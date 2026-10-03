import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from backend.core.exceptions import CustomException
from backend.core.logger import logging


NUMERICAL_FEATURES = [
    "annual_salary_gbp", "credit_card_utilization_pct", "on_time_payment_ratio_pct",
    "rent_payment_monthly_gbp", "rent_on_time_rate_pct", "utility_bills_on_time_pct",
    "mobile_money_tx_monthly", "mobile_money_age_days", "preferred_loan_term_months",
    "existing_loans_count", "months_credit_history",
]
CATEGORICAL_FEATURES = ["gender", "occupation", "mortgage_status", "mobile_money_active"]

PSI_WARN_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.25


@dataclass
class MonitoringConfig:
    reference_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    prediction_log_path: str = os.path.join("artifacts", "prediction_log.csv")
    drift_report_path: str = os.path.join("artifacts", "drift_report.csv")


class ModelMonitor:
    """
    Lightweight production monitoring for the credit scoring model:
      - logs every prediction made via the API/agents, as an audit trail
      - computes Population Stability Index (PSI) for numerical features
        and a proportion-shift score for categorical features, comparing
        live traffic against the original training distribution
      - compares the distribution of predicted scores themselves against
        what the model produced at training time, which can catch concept
        drift that input-only checks would miss
      - flags features/predictions that have drifted enough to warrant
        a closer look or a retrain

    No dedicated monitoring platform required - PSI and proportion
    comparisons are simple enough to compute directly, which keeps this
    dependency-free and easy to run as a scheduled job (cron, Task
    Scheduler, Airflow, etc) independent of the API process.
    """

    def __init__(self):
        self.config = MonitoringConfig()

    def log_prediction(
        self,
        applicant: dict,
        ml_score: float,
        traditional_score: Optional[float] = None,
        decision: Optional[str] = None,
    ) -> None:
        """
        Appends one row per scored applicant to a running CSV log. This is
        the audit trail of what was scored, when, and what the system
        decided - independent of whether the row is later added to the
        training dataset via the RAG statement pipeline.
        """
        try:
            os.makedirs(os.path.dirname(self.config.prediction_log_path), exist_ok=True)

            row = dict(applicant)
            row["timestamp"] = datetime.utcnow().isoformat()
            row["ml_score"] = ml_score
            row["traditional_score"] = traditional_score
            row["decision"] = decision

            df_row = pd.DataFrame([row])
            file_exists = os.path.exists(self.config.prediction_log_path)

            df_row.to_csv(
                self.config.prediction_log_path,
                mode="a" if file_exists else "w",
                header=not file_exists,
                index=False,
            )

            logging.info(f"Logged prediction: ml_score={ml_score}, decision={decision}")

        except Exception as e:
            raise CustomException(e, sys)

    def _psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """
        Population Stability Index for a numerical feature. Buckets the
        reference distribution into quantile bins, then compares what
        proportion of the current distribution falls into each of those
        same bins.

        Rule of thumb: PSI < 0.1 = no significant shift, 0.1-0.25 = some
        shift worth watching, > 0.25 = significant shift, investigate.
        """
        try:
            reference = pd.to_numeric(reference, errors="coerce").dropna()
            current = pd.to_numeric(current, errors="coerce").dropna()

            if len(reference) == 0 or len(current) == 0:
                return 0.0

            quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
            if len(quantiles) < 3:
                return 0.0  # not enough distinct values to bin meaningfully

            ref_counts, _ = np.histogram(reference, bins=quantiles)
            cur_counts, _ = np.histogram(current, bins=quantiles)

            ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / ref_counts.sum())
            cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / cur_counts.sum())

            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return float(psi)

        except Exception as e:
            raise CustomException(e, sys)

    def _categorical_drift(self, reference: pd.Series, current: pd.Series) -> float:
        """
        Proportion-shift score for categorical features: sum of absolute
        differences in category proportions between reference and
        current, halved so it ranges 0 (identical) to 1 (completely
        different). Simpler to read on a dashboard than a chi-square
        p-value, at the cost of not being a formal hypothesis test.
        """
        try:
            reference = reference.dropna().astype(str)
            current = current.dropna().astype(str)

            if len(reference) == 0 or len(current) == 0:
                return 0.0

            ref_props = reference.value_counts(normalize=True)
            cur_props = current.value_counts(normalize=True)

            all_categories = set(ref_props.index) | set(cur_props.index)
            diff = sum(
                abs(ref_props.get(cat, 0) - cur_props.get(cat, 0)) for cat in all_categories
            )
            return float(diff / 2)

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _status_from_score(value: float) -> str:
        if value < PSI_WARN_THRESHOLD:
            return "stable"
        elif value < PSI_ALERT_THRESHOLD:
            return "watch"
        else:
            return "drifted"

    def check_drift(self, current_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compares current feature distributions (defaults to whatever has
        been logged via log_prediction so far) against the original
        training distribution, and returns a per-feature drift report
        sorted by severity.
        """
        try:
            reference_df = pd.read_csv(self.config.reference_data_path)

            if current_df is None:
                if not os.path.exists(self.config.prediction_log_path):
                    raise ValueError(
                        "No prediction log found yet and no current_df provided. "
                        "Score some applicants first, or pass a DataFrame to check_drift()."
                    )
                current_df = pd.read_csv(self.config.prediction_log_path)

            rows = []

            for feature in NUMERICAL_FEATURES:
                if feature not in current_df.columns:
                    continue
                psi = self._psi(reference_df[feature], current_df[feature])
                rows.append({
                    "feature": feature,
                    "type": "numerical",
                    "metric": "PSI",
                    "drift_score": round(psi, 4),
                    "status": self._status_from_score(psi),
                })

            for feature in CATEGORICAL_FEATURES:
                if feature not in current_df.columns:
                    continue
                drift = self._categorical_drift(
                    reference_df[feature].astype(str), current_df[feature].astype(str)
                )
                rows.append({
                    "feature": feature,
                    "type": "categorical",
                    "metric": "proportion shift",
                    "drift_score": round(drift, 4),
                    "status": self._status_from_score(drift),
                })

            report_df = pd.DataFrame(rows).sort_values("drift_score", ascending=False)

            os.makedirs(os.path.dirname(self.config.drift_report_path), exist_ok=True)
            report_df.to_csv(self.config.drift_report_path, index=False)

            logging.info(f"Drift report generated: {len(report_df)} features checked")
            return report_df

        except Exception as e:
            raise CustomException(e, sys)

    def prediction_score_drift(self) -> dict:
        """
        Compares the distribution of ML scores the model is currently
        producing (from the prediction log) against what it produced on
        the original held-out test set. A model that starts predicting
        systematically higher or lower scores than at training time -
        even with stable input features - can indicate the relationship
        between features and outcomes has shifted (concept drift), which
        input-only PSI checks would miss.
        """
        try:
            if not os.path.exists(self.config.prediction_log_path):
                raise ValueError("No prediction log found yet.")

            log_df = pd.read_csv(self.config.prediction_log_path)
            if "ml_score" not in log_df.columns or log_df["ml_score"].dropna().empty:
                raise ValueError("Prediction log has no ml_score values yet.")

            live_scores = log_df["ml_score"].dropna()

            reference_df = pd.read_csv(self.config.reference_data_path)
            reference_scores = reference_df["credit_score"].dropna()

            psi = self._psi(reference_scores, live_scores)

            return {
                "live_mean": round(float(live_scores.mean()), 1),
                "reference_mean": round(float(reference_scores.mean()), 1),
                "live_std": round(float(live_scores.std()), 1),
                "reference_std": round(float(reference_scores.std()), 1),
                "n_live_predictions": int(len(live_scores)),
                "psi": round(psi, 4),
                "status": self._status_from_score(psi),
            }

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    monitor = ModelMonitor()

    # Demo: treat the held-out test set as a stand-in for "live traffic" so
    # the report runs end to end even before any real predictions have
    # been logged yet. In production, drop current_df to use the real log.
    test_df = pd.read_csv(monitor.config.test_data_path)

    print("=== Feature drift report (test set vs training set) ===")
    report = monitor.check_drift(current_df=test_df)
    print(report.to_string(index=False))

    drifted = report[report["status"] == "drifted"]
    watch = report[report["status"] == "watch"]

    print()
    if len(drifted) > 0:
        print(f"{len(drifted)} feature(s) show significant drift: {drifted['feature'].tolist()}")
    if len(watch) > 0:
        print(f"{len(watch)} feature(s) worth watching: {watch['feature'].tolist()}")
    if len(drifted) == 0 and len(watch) == 0:
        print("No features show meaningful drift.")
