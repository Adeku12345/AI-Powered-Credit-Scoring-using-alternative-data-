import os
import sys
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from backend.core.exceptions import CustomException


import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from backend.core.exceptions import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = param.get(model_name, {})

            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            # Use predicted probabilities for ROC-AUC where available,
            # fall back to decision_function, then raw predictions.
            if hasattr(model, "predict_proba"):
                y_test_scores = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_test_scores = model.decision_function(X_test)
            else:
                y_test_scores = model.predict(X_test)

            test_model_score = roc_auc_score(y_test, y_test_scores)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)