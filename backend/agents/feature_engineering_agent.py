from typing import Dict, Any
from graph.state import AutoMLState
from utils.logger import logger
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import polars as pl

# Pure sklearn feature selection — no LLM, deterministic and fast

def feature_engineering_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    logger.info("=== Feature Engineering Agent Started ===")

    df = state["processed_data"].clone()
    target_col = state["target_column"]
    task_type = state["task_type"]

    df_pd = df.to_pandas()
    X = df_pd.drop(columns=[target_col])
    y = df_pd[target_col]

    # Safety net: drop any rows where y is still NaN
    nan_mask = y.isna()
    if nan_mask.any():
        logger.warning(f"Dropping {nan_mask.sum()} rows with NaN target in feature engineering")
        X = X[~nan_mask].reset_index(drop=True)
        y = y[~nan_mask].reset_index(drop=True)
    n_features = X.shape[1]
    selected_features = X.columns.tolist()

    # Auto feature selection: only when > 15 features, keep top sqrt(n)*2 capped at 20
    if n_features > 15:
        k = min(max(int(n_features ** 0.5) * 2, 10), n_features - 1)
        score_func = f_classif if task_type == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        X = X[selected_features]
        logger.info(f"Selected top {k} features from {n_features}: {selected_features}")
    else:
        logger.info(f"Keeping all {n_features} features")

    X[target_col] = y.values
    df_engineered = pl.from_pandas(X)

    state["processed_data"] = df_engineered
    state["selected_features"] = selected_features
    state["feature_config"] = {"strategy": "selectkbest", "n_features": len(selected_features)}
    state["agent_logs"].append(f"Feature Engineering Agent: Selected {len(selected_features)} features")

    logger.info("=== Feature Engineering Agent Completed ===")
    return state
