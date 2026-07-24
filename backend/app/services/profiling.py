import pandas as pd
import numpy as np
from io import BytesIO

def load_dataframe(file_bytes: bytes, content_type: str) -> pd.DataFrame:
    if content_type == "text/csv":
        return pd.read_csv(BytesIO(file_bytes))
    else:
        return pd.read_excel(BytesIO(file_bytes))

def profile_dataset(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape

    # Missing values per column
    missing_counts = df.isnull().sum()
    missing_pct_total = float(missing_counts.sum() / (n_rows * n_cols) * 100) if n_rows * n_cols > 0 else 0

    missing_by_column = {
        col: {
            "missing_count": int(missing_counts[col]),
            "missing_pct": round(float(missing_counts[col] / n_rows * 100), 2) if n_rows > 0 else 0,
        }
        for col in df.columns
    }

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = round(duplicate_count / n_rows * 100, 2) if n_rows > 0 else 0

    # Data types
    dtypes = {col: str(df[col].dtype) for col in df.columns}

    # Outliers (numeric columns only, using IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_by_column = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outliers_by_column[col] = outlier_count

    # Correlation matrix (numeric columns only)
    correlation = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(numeric_only=True).round(2)
        correlation = corr_matrix.to_dict()

    # --- Quality score: start at 100, deduct for issues ---
    score = 100.0
    score -= min(missing_pct_total * 0.5, 30)  # up to -30 for missing data
    score -= min(duplicate_pct * 0.5, 20)       # up to -20 for duplicates
    total_outliers = sum(outliers_by_column.values())
    outlier_pct = (total_outliers / n_rows * 100) if n_rows > 0 else 0
    score -= min(outlier_pct * 0.3, 15)          # up to -15 for outliers
    score = max(round(score, 1), 0)

    return {
        "n_rows": n_rows,
        "n_columns": n_cols,
        "quality_score": score,
        "missing_pct_total": round(missing_pct_total, 2),
        "missing_by_column": missing_by_column,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "dtypes": dtypes,
        "outliers_by_column": outliers_by_column,
        "correlation": correlation,
    }