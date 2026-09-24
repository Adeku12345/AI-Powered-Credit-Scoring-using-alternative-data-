from pydantic import BaseModel, Field
from typing import Optional, Literal

class ApplicantData(BaseModel):
    """Input data for credit scoring — matches your dataset columns exactly"""
    gender: Literal["Male", "Female", "Other"]
    occupation: Literal["Professional", "Skilled", "Clerical", "Self-Employed", "Unemployed"]
    annual_salary_gbp: int = Field(..., ge=0, description="Annual income in GBP")
    credit_card_utilization_pct: float = Field(..., ge=0, le=100, description="% of credit limit used")
    on_time_payment_ratio_pct: float = Field(..., ge=0, le=100, description="% of payments made on time")
    mortgage_status: Literal["None", "Current", "Arrears", "PaidOff"]
    rent_payment_monthly_gbp: int = Field(..., ge=0, description="Monthly rent in GBP (0 if own home)")
    rent_on_time_rate_pct: float = Field(..., ge=0, le=100)
    utility_bills_on_time_pct: float = Field(..., ge=0, le=100)
    mobile_money_active: bool
    mobile_money_tx_monthly: int = Field(..., ge=0)
    mobile_money_age_days: int = Field(..., ge=0)
    preferred_loan_term_months: Literal[6, 12, 24, 36, 60]
    existing_loans_count: int = Field(..., ge=0)
    months_credit_history: int = Field(..., ge=0)

class CreditScoreResponse(BaseModel):
    """API Response — credit score + risk + grade + factors"""
    success: bool
    credit_score_300_850: int = Field(..., ge=300, le=850, description="Credit score on standard scale")
    score_grade: Literal["A", "B", "C", "D", "E"] = Field(..., description="A=Excellent → E=High Risk")
    default_risk_percent: float = Field(..., ge=0, le=100, description="Probability of default in next 12 months")
    risk_category: Literal["Low Risk", "Moderate Risk", "Medium Risk", "High Risk", "Very High Risk"]
    top_positive_factor: str
    top_negative_factor: str
    model_version: str = "1.0.0"