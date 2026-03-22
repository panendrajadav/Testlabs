import polars as pl
import numpy as np
import time
import logging
from typing import Dict, Any, List

def llm_invoke_with_retry(llm, messages, retries: int = 3) -> Any:
    """Invoke LLM with exponential backoff retry on transient errors."""
    for attempt in range(1, retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            logging.getLogger("testlabs").warning(f"LLM call failed (attempt {attempt}/{retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

def detect_task_type(y: pl.Series) -> str:
    if y.dtype in [pl.String, pl.Categorical, pl.Enum] or y.n_unique() < 20:
        return "classification"
    return "regression"

def get_column_types(df: pl.DataFrame) -> Dict[str, List[str]]:
    numeric_dtypes = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    string_dtypes  = {pl.String, pl.Categorical}
    numeric_cols     = [col for col in df.columns if df[col].dtype in numeric_dtypes]
    categorical_cols = [col for col in df.columns if df[col].dtype in string_dtypes]
    return {"numeric": numeric_cols, "categorical": categorical_cols}

def calculate_missing_percentage(df: pl.DataFrame) -> Dict[str, float]:
    return {col: (df[col].null_count() / df.height * 100) for col in df.columns}

def detect_outliers_iqr(series: pl.Series) -> int:
    if series.dtype not in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
        return 0
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
    return int(outliers)
