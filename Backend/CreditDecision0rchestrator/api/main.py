import os
import sys
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.core.exceptions import CustomException
from backend.core.logger import logging

from backend.agents.orchestrator import CreditDecisionOrchestrator
from backend.rag_core.statement_pipeline import StatementAnalysisPipeline

app = FastAPI(title="Credit Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = CreditDecisionOrchestrator()
statement_pipeline = StatementAnalysisPipeline()


class ApplicantIn(BaseModel):
    gender: Optional[str] = None
    occupation: Optional[str] = None
    mortgage_status: Optional[str] = None
    mobile_money_active: Optional[str] = None
    annual_salary_gbp: Optional[float] = None
    credit_card_utilization_pct: Optional[float] = None
    on_time_payment_ratio_pct: Optional[float] = None
    rent_payment_monthly_gbp: Optional[float] = None
    rent_on_time_rate_pct: Optional[float] = None
    utility_bills_on_time_pct: Optional[float] = None
    mobile_money_tx_monthly: Optional[float] = None
    mobile_money_age_days: Optional[float] = None
    preferred_loan_term_months: Optional[int] = None
    existing_loans_count: Optional[int] = None
    months_credit_history: Optional[float] = None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/score")
def score_applicant(applicant: ApplicantIn):
    try:
        report = orchestrator.run(applicant.model_dump())
        return report
    except CustomException as e:
        logging.info(f"API error in /api/score: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-statement")
def analyze_statement(
    file: UploadFile = File(...),
    gender: Optional[str] = Form(None),
    occupation: Optional[str] = Form(None),
    mortgage_status: Optional[str] = Form(None),
    preferred_loan_term_months: Optional[int] = Form(None),
    existing_loans_count: Optional[int] = Form(None),
    months_credit_history: Optional[float] = Form(None),
    credit_card_utilization_pct: Optional[float] = Form(None),
    mobile_money_age_days: Optional[float] = Form(None),
):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        known_fields = {
            "gender": gender,
            "occupation": occupation,
            "mortgage_status": mortgage_status,
            "preferred_loan_term_months": preferred_loan_term_months,
            "existing_loans_count": existing_loans_count,
            "months_credit_history": months_credit_history,
            "credit_card_utilization_pct": credit_card_utilization_pct,
            "mobile_money_age_days": mobile_money_age_days,
        }

        extracted = statement_pipeline._extract_from_pdf(tmp_path)
        applicant_row = statement_pipeline._build_feature_row(extracted, known_fields)

        report = orchestrator.run(applicant_row)

        customer_id = statement_pipeline.update_dataset(
            applicant_row, report["ml_scoring"]["score"]
        )

        report["customer_id"] = customer_id
        report["extracted_fields"] = extracted
        report["final_features"] = applicant_row
        return report

    except CustomException as e:
        logging.info(f"API error in /api/analyze-statement: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)


