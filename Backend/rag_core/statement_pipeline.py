import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.rag_core.pdf_loader import extract_text_from_pdf, chunk_text
from backend.rag_core.retriever import ChunkRetriever
from backend.rag_core.feature_extractor import extract_features_from_chunks

from backend.ml_core.ml_scorer import ModelScorer
from backend.ml_core.explainer import ModelExplainer


# A single retrieval query covering the whole financial profile. Splitting
# this into one query per field would retrieve more targeted chunks, but
# costs more LLM calls; one combined query keeps this fast for a single
# statement while still giving the retriever something to rank chunks
# against (rather than dumping the entire document into the prompt).
PROFILE_QUERY = (
    "salary income payroll deposit rent payment direct debit standing order "
    "utility bill electricity gas water internet overdraft returned payment "
    "insufficient funds mobile money paypal venmo transfer"
)

# Dataset columns, for building the row to append and keeping column order
# consistent with the training CSV.
DATASET_COLUMNS = [
    "customer_id", "gender", "occupation", "annual_salary_gbp",
    "credit_card_utilization_pct", "on_time_payment_ratio_pct", "mortgage_status",
    "rent_payment_monthly_gbp", "rent_on_time_rate_pct", "utility_bills_on_time_pct",
    "mobile_money_active", "mobile_money_tx_monthly", "mobile_money_age_days",
    "preferred_loan_term_months", "existing_loans_count", "months_credit_history",
    "credit_score",
]


@dataclass
class StatementPipelineConfig:
    dataset_path: str = os.path.join("notebook", "data", "synthetic", "credit_scoring_dataset.csv")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k_chunks: int = 8


class StatementAnalysisPipeline:
    def __init__(self):
        self.config = StatementPipelineConfig()
        self.scorer = ModelScorer()
        self.explainer = ModelExplainer()

    def _extract_from_pdf(self, pdf_path: str) -> dict:
        try:
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)

            retriever = ChunkRetriever(chunks)
            relevant_chunks = retriever.retrieve(PROFILE_QUERY, top_k=self.config.top_k_chunks)

            if not relevant_chunks:
                logging.info("No relevant chunks retrieved; falling back to first chunks")
                relevant_chunks = chunks[: self.config.top_k_chunks]

            return extract_features_from_chunks(relevant_chunks)

        except Exception as e:
            raise CustomException(e, sys)

    def _build_feature_row(self, extracted: dict, known_fields: dict) -> dict:
        """
        Merges RAG-extracted fields with user-supplied known fields (e.g.
        gender, occupation, mortgage_status collected via a form) into one
        row matching the model's expected columns. Anything missing from
        both sources becomes NaN, which the preprocessor's imputer fills
        with the median/most-frequent value learned at training time.
        """
        try:
            row = {col: np.nan for col in DATASET_COLUMNS if col not in ("customer_id", "credit_score")}

            # Fields the RAG step can plausibly extract
            for key in [
                "annual_salary_gbp", "on_time_payment_ratio_pct",
                "rent_payment_monthly_gbp", "rent_on_time_rate_pct",
                "utility_bills_on_time_pct", "mobile_money_active",
                "mobile_money_tx_monthly",
            ]:
                if extracted.get(key) is not None:
                    row[key] = extracted[key]

            # User-supplied known fields take priority and fill in anything
            # a bank statement can't reveal (demographics, bureau-style data).
            for key, value in known_fields.items():
                if value is not None:
                    row[key] = value

            # mobile_money_active must be a string to match how the
            # OneHotEncoder was trained (see data_transformation.py).
            if "mobile_money_active" in row and row["mobile_money_active"] is not np.nan:
                row["mobile_money_active"] = str(row["mobile_money_active"])

            return row

        except Exception as e:
            raise CustomException(e, sys)

    def update_dataset(self, row: dict, predicted_score: float) -> str:
        """
        Appends this applicant's features and predicted score to the
        dataset CSV. Note: this grows the dataset for future retraining —
        it does not retroactively change the already-trained model.pkl.
        Re-run train_pipeline.py to have the model learn from new rows.
        """
        try:
            dataset_path = self.config.dataset_path

            if os.path.exists(dataset_path):
                existing = pd.read_csv(dataset_path)
                next_id_num = len(existing) + 1
            else:
                existing = pd.DataFrame(columns=DATASET_COLUMNS)
                next_id_num = 1

            new_row = dict(row)
            new_row["customer_id"] = f"CUST-{next_id_num:06d}"
            new_row["credit_score"] = round(predicted_score)

            new_row_df = pd.DataFrame([new_row])[DATASET_COLUMNS]
            updated = pd.concat([existing, new_row_df], ignore_index=True)

            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            updated.to_csv(dataset_path, index=False)

            logging.info(f"Appended {new_row['customer_id']} to dataset: {dataset_path}")
            return new_row["customer_id"]

        except Exception as e:
            raise CustomException(e, sys)

    def process_statement(self, pdf_path: str, known_fields: Optional[dict] = None) -> dict:
        """
        Full flow: PDF -> RAG feature extraction -> merge with known fields
        -> score -> SHAP + LLM explanation -> append to dataset.
        """
        try:
            known_fields = known_fields or {}

            logging.info(f"Processing statement: {pdf_path}")
            extracted = self._extract_from_pdf(pdf_path)
            row = self._build_feature_row(extracted, known_fields)

            features_df = pd.DataFrame([{k: v for k, v in row.items()}])

            prediction = self.scorer.predict(features_df)[0]
            explanation = self.explainer.explain(features_df)
            narrative = self.explainer.generate_llm_explanation(explanation)

            customer_id = self.update_dataset(row, prediction)

            return {
                "customer_id": customer_id,
                "extracted_fields": extracted,
                "final_features": row,
                "predicted_credit_score": float(prediction),
                "shap_explanation": explanation,
                "narrative": narrative,
            }

        except Exception as e:
            raise CustomException(e, sys)


def _parse_args():
    parser = argparse.ArgumentParser(description="Analyse a bank statement PDF and score it.")
    parser.add_argument("pdf_path", help="Path to the bank statement PDF")
    parser.add_argument("--gender", default=None)
    parser.add_argument("--occupation", default=None)
    parser.add_argument("--mortgage-status", dest="mortgage_status", default=None)
    parser.add_argument("--preferred-loan-term-months", type=int, default=None)
    parser.add_argument("--existing-loans-count", type=int, default=None)
    parser.add_argument("--months-credit-history", type=int, default=None)
    parser.add_argument("--credit-card-utilization-pct", type=float, default=None)
    parser.add_argument("--mobile-money-age-days", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    known_fields = {
        "gender": args.gender,
        "occupation": args.occupation,
        "mortgage_status": args.mortgage_status,
        "preferred_loan_term_months": args.preferred_loan_term_months,
        "existing_loans_count": args.existing_loans_count,
        "months_credit_history": args.months_credit_history,
        "credit_card_utilization_pct": args.credit_card_utilization_pct,
        "mobile_money_age_days": args.mobile_money_age_days,
    }

    pipeline = StatementAnalysisPipeline()
    result = pipeline.process_statement(args.pdf_path, known_fields)

    print(f"\nCustomer ID: {result['customer_id']}")
    print(f"Predicted credit score: {result['predicted_credit_score']:.0f}")
    print(f"\nFields extracted from statement:")
    for k, v in result["extracted_fields"].items():
        print(f"  {k}: {v}")
    print(f"\n--- Explanation ---")
    print(result["narrative"])
    print(f"\nDataset updated: {pipeline.config.dataset_path}")
