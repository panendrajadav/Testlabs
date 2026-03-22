from typing import Dict, Any, List, Tuple
from graph.state import AutoMLState
from utils.logger import logger
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import pandas as pd
import polars as pl
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
_MISSING_DROP_THRESHOLD  = 0.50   # drop column if >50% values missing
_VARIANCE_THRESHOLD      = 1e-6   # drop column if variance near zero (constant)
_HIGH_CARDINALITY_LIMIT  = 50     # treat categorical as high-cardinality above this
_KNN_NEIGHBORS           = 5      # KNN imputer neighbours
_IQR_MULTIPLIER          = 1.5    # Winsorize fence: Q1 - k*IQR, Q3 + k*IQR


# ── Step 1: Drop columns with too many missing values ─────────────────────────
def _drop_high_missing(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    missing_frac = X.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    if drop_cols:
        X = X.drop(columns=drop_cols)
        logger.info(f"Dropped {len(drop_cols)} high-missing columns (>{threshold*100:.0f}%): {drop_cols}")
    return X, drop_cols


# ── Step 2: Drop zero-variance / near-constant columns ────────────────────────
def _drop_zero_variance(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns
    variances = X[num_cols].var()
    drop_cols = variances[variances <= threshold].index.tolist()
    # Also drop columns where a single value covers >99% of rows
    for col in X.columns:
        if col not in drop_cols:
            top_freq = X[col].value_counts(normalize=True).iloc[0] if X[col].nunique() > 0 else 1.0
            if top_freq >= 0.99:
                drop_cols.append(col)
    drop_cols = list(set(drop_cols))
    if drop_cols:
        X = X.drop(columns=drop_cols)
        logger.info(f"Dropped {len(drop_cols)} zero-variance/near-constant columns: {drop_cols}")
    return X, drop_cols


# ── Step 3: Impute missing values ─────────────────────────────────────────────
def _impute(X: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    num_present = [c for c in numeric_cols if c in X.columns]
    cat_present = [c for c in categorical_cols if c in X.columns]

    # Categorical: simple most_frequent (KNN doesn't work on strings)
    if cat_present and X[cat_present].isnull().any().any():
        si = SimpleImputer(strategy="most_frequent")
        X[cat_present] = si.fit_transform(X[cat_present])
        logger.info(f"Imputed {len(cat_present)} categorical columns (most_frequent)")

    # Numeric: KNN imputer — uses feature correlations, better than median
    if num_present and X[num_present].isnull().any().any():
        n_neighbors = min(_KNN_NEIGHBORS, max(1, len(X) // 10))
        knn = KNNImputer(n_neighbors=n_neighbors)
        X[num_present] = knn.fit_transform(X[num_present])
        missing_count = X[num_present].isnull().sum().sum()
        logger.info(f"KNN-imputed {len(num_present)} numeric columns (k={n_neighbors}), remaining nulls: {missing_count}")

    return X


# ── Step 4: Outlier capping (Winsorize) ───────────────────────────────────────
def _cap_outliers(X: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    num_present = [c for c in numeric_cols if c in X.columns]
    report: Dict[str, int] = {}
    for col in num_present:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - _IQR_MULTIPLIER * iqr
        upper = q3 + _IQR_MULTIPLIER * iqr
        n_outliers = int(((X[col] < lower) | (X[col] > upper)).sum())
        if n_outliers > 0:
            X[col] = X[col].clip(lower=lower, upper=upper)
            report[col] = n_outliers
    if report:
        total = sum(report.values())
        logger.info(f"Winsorized {total} outlier values across {len(report)} columns: {list(report.keys())}")
    return X, report


# ── Step 5: Encode categorical features ───────────────────────────────────────
def _encode_categoricals(X: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    cat_present = [c for c in categorical_cols if c in X.columns]
    encoded = []
    for col in cat_present:
        n_unique = X[col].nunique()
        if n_unique > _HIGH_CARDINALITY_LIMIT:
            # High cardinality: frequency encode (rank by count, preserves signal)
            freq_map = X[col].value_counts().to_dict()
            X[col] = X[col].map(freq_map).fillna(0).astype(int)
            logger.info(f"Frequency-encoded high-cardinality column '{col}' ({n_unique} unique)")
        else:
            # Low/medium cardinality: ordinal encode (handles unseen values gracefully)
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X[col] = enc.fit_transform(X[[col]]).astype(int)
        encoded.append(col)
    return X, encoded


# ── Step 6: Final type coercion — ensure all columns are numeric ───────────────
def _coerce_to_numeric(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    coerced = []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            coerced.append(col)
    if coerced:
        logger.info(f"Force-coerced {len(coerced)} remaining non-numeric columns: {coerced}")
    return X, coerced


# ── Main agent ────────────────────────────────────────────────────────────────
def preprocessing_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    logger.info("=== Preprocessing Agent Started ===")

    df = state["raw_data"].clone()
    target_col = state["target_column"]
    eda = state["eda_summary"]
    numeric_cols: List[str] = eda["numeric_cols"]
    categorical_cols: List[str] = eda["categorical_cols"]

    df_pd = df.to_pandas()

    # Drop rows where target is NaN — cannot impute the label
    n_before = len(df_pd)
    df_pd = df_pd.dropna(subset=[target_col]).reset_index(drop=True)
    n_dropped = n_before - len(df_pd)
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} rows with NaN target ('{target_col}')")

    X = df_pd.drop(columns=[target_col])
    y = df_pd[target_col]

    original_shape = X.shape
    logger.info(f"Raw input: {original_shape[0]} rows x {original_shape[1]} cols")

    # ── 1. Drop high-missing columns ──────────────────────────────────────────
    X, dropped_missing = _drop_high_missing(X, _MISSING_DROP_THRESHOLD)

    # Sync column lists after drops
    numeric_cols    = [c for c in numeric_cols    if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    # ── 2. Drop zero-variance / near-constant columns ─────────────────────────
    X, dropped_variance = _drop_zero_variance(X, _VARIANCE_THRESHOLD)
    numeric_cols    = [c for c in numeric_cols    if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    # ── 3. Impute missing values ───────────────────────────────────────────────
    X = _impute(X, numeric_cols, categorical_cols)

    # ── 4. Cap outliers (Winsorize) ────────────────────────────────────────────
    X, outlier_report = _cap_outliers(X, numeric_cols)

    # ── 5. Encode categorical features ────────────────────────────────────────
    X, encoded_cols = _encode_categoricals(X, categorical_cols)

    # ── 6. Encode target label ────────────────────────────────────────────────
    if state["task_type"] == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)
        logger.info(f"Target label encoded: {list(le.classes_)}")

    # ── 7. Final coercion — guarantee all numeric ──────────────────────────────
    X, coerced_cols = _coerce_to_numeric(X)

    # ── 8. Assemble processed dataframe ───────────────────────────────────────
    X[target_col] = y.values
    processed_df = pl.from_pandas(X)

    # ── Summary ───────────────────────────────────────────────────────────────
    final_shape = (processed_df.height, processed_df.width - 1)  # exclude target
    remaining_nulls = X.drop(columns=[target_col]).isnull().sum().sum()

    preprocessing_report = {
        "original_shape":    list(original_shape),
        "final_shape":       list(final_shape),
        "dropped_missing":   dropped_missing,
        "dropped_variance":  dropped_variance,
        "outliers_capped":   outlier_report,
        "encoded_cols":      encoded_cols,
        "coerced_cols":      coerced_cols,
        "remaining_nulls":   int(remaining_nulls),
    }

    logger.info(
        f"Preprocessing complete: {original_shape} -> {final_shape} | "
        f"dropped={len(dropped_missing)+len(dropped_variance)} cols | "
        f"outliers capped={sum(outlier_report.values())} | "
        f"nulls remaining={remaining_nulls}"
    )

    state["processed_data"]       = processed_df
    state["preprocessing_config"] = preprocessing_report
    state["agent_logs"].append(
        f"Preprocessing: {original_shape[0]}x{original_shape[1]} -> "
        f"{final_shape[0]}x{final_shape[1]} | "
        f"dropped {len(dropped_missing)+len(dropped_variance)} cols | "
        f"capped {sum(outlier_report.values())} outliers | "
        f"{remaining_nulls} nulls remaining"
    )

    logger.info("=== Preprocessing Agent Completed ===")
    return state
