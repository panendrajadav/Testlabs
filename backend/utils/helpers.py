import polars as pl
import numpy as np
from typing import Dict, Any, List

def detect_task_type(y: pl.Series) -> str:
    """Detect if task is classification or regression."""
    if y.dtype in [pl.Utf8, pl.Categorical] or y.n_unique() < 20:
        return "classification"
    return "regression"

def get_column_types(df: pl.DataFrame) -> Dict[str, List[str]]:
    """Categorize columns by type."""
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]]
    categorical_cols = [col for col in df.columns if df[col].dtype in [pl.Utf8, pl.Categorical]]
    return {"numeric": numeric_cols, "categorical": categorical_cols}

def calculate_missing_percentage(df: pl.DataFrame) -> Dict[str, float]:
    """Calculate missing value percentage for each column."""
    return {col: (df[col].null_count() / df.height * 100) for col in df.columns}

def detect_outliers_iqr(series: pl.Series) -> int:
    """Detect outliers using IQR method."""
    if series.dtype not in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
        return 0
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
    return int(outliers)
