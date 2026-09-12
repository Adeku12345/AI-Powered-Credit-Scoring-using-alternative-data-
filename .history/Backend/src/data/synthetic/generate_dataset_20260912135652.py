#SOFTINT AI — SYNTHETIC CREDIT SCORING DATASET GENERATOR
#Generates realistic, correlated features ready for model training
#Output: CSV with customer profiles + default label (supervised learning ready)
#"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ──
np.random.seed(42)  # Reproducibility
NUM_PROFILES = 50000
OUTPUT_PATH = Path("data/synthetic/credit_scoring_dataset.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"🧪 Generating {NUM_PROFILES:,} synthetic credit profiles...")

# ── Base Demographics ──
customer_ids = [f"CUST-{i+1:06d}" for i in range(NUM_PROFILES)]

gender = np.random.choice(["Male", "Female", "Other"], size=NUM_PROFILES, p=[0.48, 0.48, 0.04])

occupation = np.random.choice(
    ["Professional", "Skilled", "Clerical", "Self-Employed", "Unemployed"],
    size=NUM_PROFILES,
    p=[0.30, 0.25, 0.20, 0.15, 0.10]
)

# Salary correlates with occupation
salary_map = {
    "Professional":    (65000, 12000),
    "Skilled":        (38000,  8000),
    "Clerical":       (26000,  6000),
    "Self-Employed":  (42000, 15000),
    "Unemployed":     (10000,  3000)
}
annual_salary = np.array([np.random.normal(*salary_map[occ]) for occ in occupation]).clip(8000, 95000).round(0)

# ── Credit Card Usage ──
credit_card_utilization = np.random.beta(2.2, 3.5, NUM_PROFILES) * 100  # Skewed low → good

# ── Payment Behaviour ──
# Correlate on-time payments with salary & utilization
base_payment = 0.65 + (annual_salary / 100000) * 0.20 - (credit_card_utilization / 200) * 0.30
on_time_payment_ratio = np.clip(base_payment + np.random.normal(0, 0.08, NUM_PROFILES), 0.30, 1.00).round(4) * 100

# ── Mortgage ──
mortgage_probs = {"Professional": 0.55, "Skilled": 0.35, "Clerical": 0.20, "Self-Employed": 0.25, "Unemployed": 0.02}
mortgage = np.array([
    np.random.choice(
        ["None", "Current", "Arrears", "PaidOff"],
        p=[
            1 - mortgage_probs[occ],
            mortgage_probs[occ] * 0.75,
            mortgage_probs[occ] * 0.10,
            mortgage_probs[occ] * 0.15
        ]
    ) for occ in occupation
])

# ── Rent ──
rent_monthly = np.where(
    mortgage == "None",
    np.random.lognormal(5.5, 0.35, NUM_PROFILES).round(-1).clip(300, 2500),
    0.0
)
rent_on_time_rate = np.where(
    mortgage == "None",
    np.clip(on_time_payment_ratio + np.random.normal(5, 8, NUM_PROFILES), 40, 100).round(1),
    100.0  # Not renting → neutral
)

# ── Utility Bills ──
utility_bills_on_time = np.clip(on_time_payment_ratio + np.random.normal(3, 10, NUM_PROFILES), 25, 100).round(1)

# ── Mobile Money ──
momo_active = np.random.choice([True, False], size=NUM_PROFILES, p=[0.65, 0.35])
momo_tx_monthly = np.where(momo_active, np.random.poisson(12, NUM_PROFILES).clip(0, 120), 0)
momo_age_days = np.where(momo_active, np.random.randint(30, 1825, NUM_PROFILES), 0)

# ── Loan Term ──
loan_term_months = np.random.choice([6, 12, 24, 36, 60], size=NUM_PROFILES, p=[0.15, 0.25, 0.30, 0.20, 0.10])

# ── Additional Credit Features ──
existing_loans_count = np.random.poisson(0.8, NUM_PROFILES).clip(0, 5)
months_credit_history = np.random.exponential(80, NUM_PROFILES).clip(0, 360).astype(int)

# ── TARGET: Default Probability & Label ──
# Formula: higher utilization + late payments + arrears → much higher default risk
default_risk = (
    0.40 * (credit_card_utilization / 100) +
    0.45 * (1 - on_time_payment_ratio / 100) +
    0.30 * np.where(mortgage == "Arrears", 0.8, np.where(mortgage == "Current", 0.1, 0.0)) +
    0.15 * np.minimum(existing_loans_count / 3, 1.0) -
    0.20 * np.minimum(months_credit_history / 120, 1.0) -
    0.10 * np.minimum(annual_salary / 60000, 1.0) +
    np.random.normal(0, 0.06, NUM_PROFILES)
)
default_risk = np.clip(default_risk, 0.01, 0.75)
default_next_12m = np.random.binomial(1, default_risk, NUM_PROFILES)

# ── Build DataFrame ──
df = pd.DataFrame({
    "customer_id": customer_ids,
    "gender": gender,
    "occupation": occupation,
    "annual_salary_gbp": annual_salary.astype(int),
    "credit_card_utilization_pct": credit_card_utilization.round(1),
    "on_time_payment_ratio_pct": on_time_payment_ratio.round(1),
    "mortgage_status": mortgage,
    "rent_payment_monthly_gbp": rent_monthly.astype(int),
    "rent_on_time_rate_pct": rent_on_time_rate,
    "utility_bills_on_time_pct": utility_bills_on_time,
    "mobile_money_active": momo_active,
    "mobile_money_tx_monthly": momo_tx_monthly,
    "mobile_money_age_days": momo_age_days,
    "preferred_loan_term_months": loan_term_months,
    "existing_loans_count": existing_loans_count,
    "months_credit_history": months_credit_history,
    "default_next_12m": default_next_12m
})

# ── Save ──
df.to_csv(OUTPUT_PATH, index=False)

# ── Print Summary ──
print(f"\n✅ DATASET SAVED → {OUTPUT_PATH}")
print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n📈 DEFAULT RATE: {df['default_next_12m'].mean():.1%}")
print(f"\n💰 Salary Distribution:")
print(f"   Mean: £{df['annual_salary_gbp'].mean():,.0f}  |  Median: £{df['annual_salary_gbp'].median():,.0f}")
print(f"\n💳 Credit Card Utilization: {df['credit_card_utilization_pct'].mean():.1f}% (avg)")
print(f"📅 On-Time Payments: {df['on_time_payment_ratio_pct'].mean():.1f}% (avg)")
print(f"\n🏠 Mortgage Status:")
for status, pct in df["mortgage_status"].value_counts(normalize=True).items():print(f"   {status:12} {pct:.1%}")
print(f"\n📱 Mobile Money Users: {df['mobile_money_active'].mean():.1%}")
print(f"\n⚖️  Fairness Check — Default Rate by Gender:")
for g in sorted(df["gender"].unique()):
    rate = df[df["gender"]==g]["default_next_12m"].mean()
    print(f"   {g:10} {rate:.1%}")
    
df.head()    