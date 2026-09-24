import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from backend.core.logging import logger

class DataProcessor:
    """PHASE 2: ML pipeline — clean & prepare data"""
    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def load_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=['on_time_payments_6m', 'income_stability'])
        df['rent_weighted'] = df['rent_on_time'] * 1.5
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df

    def split_and_scale(self, df: pd.DataFrame, target_col='default_label'):
        feature_cols = [
            'on_time_payments_6m', 'income_stability',
            'debt_to_income', 'credit_utilisation', 'rent_weighted'
        ]
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self._fitted = True
        
        return X_train_scaled, X_test_scaled, y_train, y_test