import pandas as pd
from backend.data.preprocessing import DataProcessor
from backend.ml_core.ml_scorer import MLScorer

# 1. Load generated dataset
df = pd.read_csv("data/synthetic_credit_data.csv")

# 2. Preprocess
processor = DataProcessor()
df_clean = processor.load_and_clean(df)
X_train, X_test, y_train, y_test = processor.split_and_scale(df_clean)

# 3. Train & evaluate
scorer = MLScorer()
scorer.train(X_train, y_train, feature_names=X_test.columns.tolist())
metrics = scorer.evaluate(X_test, y_test)

print("✅ Model Trained!")
print(f"ROC-AUC: {metrics['roc_auc']:.2f}")
print("\nFeature Importance:")
for name, imp in scorer.feature_importance().items():
    print(f"  {name}: {imp:.1%}")