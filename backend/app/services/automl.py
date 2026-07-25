import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBRegressor, XGBClassifier
import joblib
from io import BytesIO

def train_and_select_best(X, y, problem_type: str) -> dict:
    if problem_type == "classification":
        class_counts = y.value_counts()
        if (class_counts < 2).any():
            raise ValueError(
                "Target column has classes with fewer than 2 samples "
                "(minimum required for train/test split). "
                "This target may have too many unique categories for the dataset size."
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type == "classification" else None,
    )

    if problem_type == "regression":
        candidates = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(random_state=42, verbosity=0),
        }
    else:
        candidates = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"),
        }

    results = {}
    trained_models = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained_models[name] = model

        if problem_type == "regression":
            results[name] = {
                "r2": round(float(r2_score(y_test, preds)), 4),
                "mae": round(float(mean_absolute_error(y_test, preds)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            }
        else:
            is_binary = len(np.unique(y)) == 2
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
            }
            if is_binary and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
            results[name] = metrics

    if problem_type == "regression":
        best_name = max(results, key=lambda n: results[n]["r2"])
    else:
        best_name = max(results, key=lambda n: results[n]["f1"])

    best_model = trained_models[best_name]

    buffer = BytesIO()
    joblib.dump(best_model, buffer)
    model_bytes = buffer.getvalue()

    return {
        "all_results": results,
        "best_model_name": best_name,
        "best_model_metrics": results[best_name],
        "model_bytes": model_bytes,
    }