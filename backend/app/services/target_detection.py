import pandas as pd

def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
    series = df[target_column].dropna()
    n_unique = series.nunique()

    if pd.api.types.is_numeric_dtype(series):
        # Numeric column: could still be classification if very few unique values
        # (e.g., a 0/1 flag, or a 1-5 rating stored as int)
        if n_unique <= 10:
            return "classification"
        return "regression"
    else:
        # Non-numeric (string/object/bool) is always classification
        return "classification"