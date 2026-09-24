import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to saved model
MODEL_PATH = Path(__file__).parent.parent / "models" / "credit_scoring_model.pkl"

# Global model cache — load once, serve forever
_model = None

def load_model():
    """Load trained model from disk — call once at startup"""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}\n"
                "→ Run: python scripts/train_credit_model.py  first!"
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
    return _model

def calculate_score_and_risk(default_prob: float) -> Tuple[int, str, str]:
    """Convert default probability → 300-850 score + grade + risk category"""
    # Score formula: lower default risk = higher credit score
    score = int(round(850 - (default_prob * 550)))
    score = max(300, min(850, score))  # Clamp to valid range

    # Grade bands (industry standard)
    if score >= 750:
        grade = "A"
        risk_cat = "Low Risk"
    elif score >= 650:
        grade = "B"
        risk_cat = "Moderate Risk"
    elif score >= 550:
        grade = "C"
        risk_cat = "Medium Risk"
    elif score >= 400:
        grade = "D"
        risk_cat = "High Risk"
    else:
        grade = "E"
        risk_cat = "Very High Risk"

    return score, grade, risk_cat

def identify_key_factors(applicant: Dict) -> Tuple[str, str]:
    """Identify top positive & negative factors from input data"""
    factors = []

    # Positive factors
    if applicant["on_time_payment_ratio_pct"] >= 95:
        factors.append(("Excellent payment history", +3))
    if applicant["credit_card_utilization_pct"] <= 30:
        factors.append(("Low credit utilization", +2))
    if applicant["months_credit_history"] >= 60:
        factors.append(("Long credit history", +2))
    if applicant["rent_on_time_rate_pct"] >= 98 and applicant["rent_payment_monthly_gbp"] > 0:
        factors.append(("Consistent rent payments", +2))
    if applicant["annual_salary_gbp"] >= 50000:
        factors.append(("Stable high income", +1))

    # Negative factors
    if applicant["credit_card_utilization_pct"] > 70:
        factors.append(("High credit utilization", -3))
    if applicant["on_time_payment_ratio_pct"] < 80:
        factors.append(("Late payment history", -3))
    if applicant["mortgage_status"] == "Arrears":
        factors.append(("Mortgage in arrears", -4))
    if applicant["months_credit_history"] < 12:
        factors.append(("Short credit history", -2))
    if applicant["existing_loans_count"] >= 3:
        factors.append(("Too many existing loans", -2))

    # Sort by impact
    positives = [f for f in factors if f[1] > 0]
    negatives = [f for f in factors if f[1] < 0]

    top_pos = sorted(positives, key=lambda x: -x[1])[0][0] if positives else "No strong positive factors"
    top_neg = sorted(negatives, key=lambda x: x[1])[0][0] if negatives else "No significant negative factors"

    return top_pos, top_neg

def score_applicant(applicant_data: Dict) -> Dict:
    """
    Main scoring function — takes applicant data, returns complete score result
    This is the single source of truth for all scoring requests
    """
    model = load_model()

    # Convert to DataFrame (matches training format)
    input_df = pd.DataFrame([applicant_data])

    # Predict default probability
    default_prob = float(model.predict_proba(input_df)[:, 1][0])
    default_risk_pct = round(default_prob * 100, 1)

    # Calculate score & grade
    score, grade, risk_cat = calculate_score_and_risk(default_prob)

    # Identify factors
    top_pos, top_neg = identify_key_factors(applicant_data)

    return {
        "success": True,
        "credit_score_300_850": score,
        "score_grade": grade,
        "default_risk_percent": default_risk_pct,
        "risk_category": risk_cat,
        "top_positive_factor": top_pos,
        "top_negative_factor": top_neg,
        "model_version": "1.0.0"
    }