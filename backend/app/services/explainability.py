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

    Includes 'sum_check': base_value + sum(all contributions), which should equal
    the model's raw prediction for this row (SHAP's additivity guarantee). Useful
    for verifying the explanation is mathematically correct, not just plausible-looking.
    """
    explainer = shap.TreeExplainer(model) if _is_tree_model(model) else shap.Explainer(model, X_background)
    shap_values = explainer(X_row)

    values = shap_values.values[0]
    base_value = shap_values.base_values[0]

    if np.ndim(values) == 2:
        class_idx = int(np.argmax(np.abs(values).sum(axis=0)))
        values = values[:, class_idx]
        base_value = base_value[class_idx] if np.ndim(base_value) > 0 else base_value

    contributions = sorted(
        zip(X_row.columns, values.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    sum_check = float(base_value) + float(np.sum(values))

    return {
        "base_value": round(float(base_value), 4),
        "contributions": [
            {"feature": f, "impact": round(v, 4)} for f, v in contributions
        ],
        "sum_check": round(sum_check, 4),
    }


def generate_plain_english_explanation(
    contributions: list[dict],
    input_values: dict,
    prediction: float,
    top_n: int = 4,
) -> list[str]:
    """
    Converts SHAP contributions into human-readable sentences.
    Groups one-hot encoded dummy columns back into their original feature
    (e.g. basement_yes + basement_no -> just "basement") before ranking,
    so each original input field appears at most once. Also ensures both
    the top increasing and top decreasing features are represented.
    """
    grouped: dict[str, float] = {}
    for c in contributions:
        feature = c["feature"]
        impact = c["impact"]

        matched_key = None
        for key in input_values.keys():
            if feature == key or feature.startswith(f"{key}_"):
                matched_key = key
                break

        group_key = matched_key if matched_key else feature
        grouped[group_key] = grouped.get(group_key, 0.0) + impact

    combined = [{"feature": k, "impact": v} for k, v in grouped.items()]

    increases = sorted(
        [c for c in combined if c["impact"] > 0],
        key=lambda c: c["impact"],
        reverse=True,
    )
    decreases = sorted(
        [c for c in combined if c["impact"] < 0],
        key=lambda c: c["impact"],
    )

    def to_sentence(c: dict) -> str:
        feature = c["feature"]
        impact = c["impact"]
        display_value = input_values.get(feature)

        direction = "increased" if impact > 0 else "decreased"
        magnitude = abs(impact)

        if display_value is not None:
            return f"{feature} = {display_value} {direction} the prediction by about {magnitude:,.0f}"
        return f"{feature} {direction} the prediction by about {magnitude:,.0f}"

    half = max(1, top_n // 2)
    top_increases = increases[:half]
    top_decreases = decreases[:top_n - half] if decreases else []

    remaining = top_n - len(top_increases) - len(top_decreases)
    if remaining > 0:
        if len(increases) > len(top_increases):
            top_increases += increases[len(top_increases):len(top_increases) + remaining]
        elif len(decreases) > len(top_decreases):
            top_decreases += decreases[len(top_decreases):len(top_decreases) + remaining]

    result = sorted(top_increases + top_decreases, key=lambda c: abs(c["impact"]), reverse=True)

    return [to_sentence(c) for c in result]


def _is_tree_model(model) -> bool:
    return type(model).__name__ in {
        "RandomForestRegressor", "RandomForestClassifier",
        "XGBRegressor", "XGBClassifier",
    }