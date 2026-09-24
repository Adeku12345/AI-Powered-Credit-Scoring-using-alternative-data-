import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def load_data():
    df = pd.read_csv("data/processed/credit_data.csv")
    y = (df["target"] >= df["target"].median()).astype(int)
    return df, y

def train_and_save(model, X, y, name, scaler=False):
    path = Path(__file__).parent.parent / "artifacts"
    path.mkdir(exist_ok=True)
    X_proc = StandardScaler().fit_transform(X) if scaler else X
    model.fit(X_proc, y)
    joblib.dump(model, path / f"{name}_model.pkl")
    print(f"✅ Saved {name} model")

if __name__ == "__main__":
    df, y = load_data()
    # Payment
    X_pay = df[["on_time_pct_bureau","rent_on_time_pct"]]/100
    train_and_save(XGBClassifier(), X_pay, y, "payment")
    # Finance
    X_fin = pd.DataFrame({
        "salary_norm": df["annual_salary_gbp"]/100000,
        "dti": np.minimum(df["outstanding_debt_gbp"]/df["annual_salary_gbp"],1),
        "util": df["credit_card_utilization_pct"]/100,
    })
    train_and_save(MLPClassifier(max_iter=500, early_stopping=True), X_fin, y, "finance", scaler=True)
    # History
    X_hist = pd.DataFrame({
        "history": np.minimum(df["months_credit_history"]/240,1),
        "defaults": np.minimum(df["defaults_count"]/5,1),
        "enquiries": np.minimum(df["enquiries_last_6m"]/10,1),
    })
    train_and_save(LGBMClassifier(verbose=-1), X_hist, y, "history")
    # Alternative
    X_alt = pd.DataFrame({
        "rent": df["rent_on_time_pct"]/100,
        "age": np.minimum(df["age"]/65,1),
    })
    train_and_save(RandomForestClassifier(), X_alt, y, "alternative")
    # Fraud
    X_all = pd.concat([X_pay, X_fin, X_hist, X_alt], axis=1)
    fraud = IsolationForest(contamination=0.05, random_state=42).fit(X_all)
    joblib.dump(fraud, path / "fraud_model.pkl")
    print("✅ Saved fraud model")
    print("\n🎉 ALL MODELS TRAINED & SAVED!")