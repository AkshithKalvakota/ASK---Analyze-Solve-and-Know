import shap
import numpy as np
import pandas as pd

def get_feature_importance(model, X_sample: pd.DataFrame) -> dict:
    """
    Computes global feature importance using SHAP values averaged over a sample
    of the training data. Returns features sorted by importance, most impactful first.
    """
    explainer = shap.TreeExplainer(model) if _is_tree_model(model) else shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    values = shap_values.values
    if values.ndim == 3:
        # Multi-class classification: average across classes
        values = np.abs(values).mean(axis=2)
    else:
        values = np.abs(values)

    mean_abs_shap = values.mean(axis=0)

    importance = sorted(
        zip(X_sample.columns, mean_abs_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {"feature_importance": [{"feature": f, "importance": round(v, 4)} for f, v in importance]}


def explain_single_prediction(model, X_row: pd.DataFrame, X_background: pd.DataFrame) -> dict:
    """
    Explains one specific prediction: how much each feature pushed the prediction
    away from the baseline (average) value.
    """
    explainer = shap.TreeExplainer(model) if _is_tree_model(model) else shap.Explainer(model, X_background)
    shap_values = explainer(X_row)

    values = shap_values.values[0]
    base_value = shap_values.base_values[0]

    if np.ndim(values) == 2:
        # Multi-class: pick the class with highest predicted contribution sum
        class_idx = int(np.argmax(np.abs(values).sum(axis=0)))
        values = values[:, class_idx]
        base_value = base_value[class_idx] if np.ndim(base_value) > 0 else base_value

    contributions = sorted(
        zip(X_row.columns, values.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "base_value": round(float(base_value), 4),
        "contributions": [
            {"feature": f, "impact": round(v, 4)} for f, v in contributions
        ],
    }


def _is_tree_model(model) -> bool:
    return type(model).__name__ in {
        "RandomForestRegressor", "RandomForestClassifier",
        "XGBRegressor", "XGBClassifier",
    }