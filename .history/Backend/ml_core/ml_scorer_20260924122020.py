from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from backend.core.logging import logger
import numpy as np

class MLScorer:
    """PHASE 2: ML model — learns patterns from data"""
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_names = None

    def train(self, X_train, y_train, feature_names):
        self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        logger.info("ML model training complete")

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, preds, output_dict=True)
        roc = roc_auc_score(y_test, probs)
        logger.info(f"Model Accuracy: {report['accuracy']:.2f}, ROC-AUC: {roc:.2f}")
        return {"report": report, "roc_auc": roc}

    def predict_score(self, X):
        """Convert default probability → 0–1000 credit score"""
        default_risk = self.model.predict_proba(X)[:, 0]
        score = 1000 - (default_risk * 600)
        return np.clip(score, 300, 1000).round(0)

    def feature_importance(self):
        """Explainability — which features matter most"""
        return dict(zip(self.feature_names, self.model.feature_importances_.round(3)))