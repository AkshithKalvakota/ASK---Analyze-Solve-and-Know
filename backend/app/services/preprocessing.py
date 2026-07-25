import re
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

    # --- Step 3: Sanitize column names — XGBoost rejects [, ], < in feature names ---
    X.columns = [re.sub(r"[\[\]<]", "_", str(col)) for col in X.columns]

    log["feature_columns_after_encoding"] = X.columns.tolist()

    # --- Step 4: Handle target column for classification ---
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category")
        log["target_categories"] = y.cat.categories.tolist()
        y = y.cat.codes  # convert to integer codes for sklearn/XGBoost
    else:
        y = y.fillna(y.median())

    return X, y, log

def apply_preprocessing_to_row(row: dict, preprocessing_log: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])

    encoding_log = preprocessing_log.get("encoding", {})
    categorical_columns = set(encoding_log.keys())

    # Convert any column NOT marked as categorical to numeric
    for col in df.columns:
        if col not in categorical_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply the same imputation values recorded during training
    for col, info in preprocessing_log.get("imputation", {}).items():
        if col in df.columns and pd.isnull(df[col].iloc[0]):
            df[col] = info["value"]

    # Recreate one-hot encoding using the exact categories seen at training time
    for col, info in encoding_log.items():
        if col in df.columns:
            value = str(df[col].iloc[0])
            for category in info["categories"]:
                col_name = f"{col}_{category}"
                df[col_name] = 1 if value == category else 0
            df = df.drop(columns=[col])

    # Sanitize column names identically to training time
    df.columns = [re.sub(r"[\[\]<]", "_", str(c)) for c in df.columns]

    # Ensure all training-time columns exist, in the same order, filling missing ones with 0
    expected_cols = preprocessing_log["feature_columns_after_encoding"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    return df