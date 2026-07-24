import pandas as pd
from typing import Tuple, Any

def preprocess(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Splits df into X (features) and y (target), handles missing values,
    and one-hot encodes categorical columns.

    Returns:
        X: preprocessed feature DataFrame, ready for model training
        y: target Series
        preprocessing_log: dict describing every transformation applied,
                            needed later to replay the same steps on new rows (what-if)
    """
    df = df.copy()
    y = df[target_column]
    X = df.drop(columns=[target_column])

    log: dict[str, Any] = {
        "target_column": target_column,
        "imputation": {},
        "encoding": {},
        "feature_columns_after_encoding": [],
    }

    # --- Step 1: Impute missing values ---
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                fill_value = X[col].median()
                X[col] = X[col].fillna(fill_value)
                log["imputation"][col] = {"strategy": "median", "value": float(fill_value)}
            else:
                fill_value = X[col].mode().iloc[0] if not X[col].mode().empty else "missing"
                X[col] = X[col].fillna(fill_value)
                log["imputation"][col] = {"strategy": "mode", "value": str(fill_value)}

    # --- Step 2: One-hot encode categorical columns ---
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    for col in categorical_cols:
        categories = sorted(X[col].astype(str).unique().tolist())
        log["encoding"][col] = {"categories": categories}

    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, dummy_na=False)

    log["feature_columns_after_encoding"] = X.columns.tolist()

    # --- Step 3: Handle target column for classification ---
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category")
        log["target_categories"] = y.cat.categories.tolist()
        y = y.cat.codes  # convert to integer codes for sklearn/XGBoost
    else:
        y = y.fillna(y.median())

    return X, y, log