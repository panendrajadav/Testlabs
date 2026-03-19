from typing import Dict, Any
from graph.state import AutoMLState
from utils.logger import logger
from tools.optuna_tools import optimize_hyperparameters, get_search_space
from tools.sklearn_tools import train_and_evaluate_sklearn
from tools.xgboost_tools import train_and_evaluate_xgboost, train_and_evaluate_lightgbm
import numpy as np

def hyperparameter_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    """Optimize hyperparameters using Optuna — no LLM, pure search."""
    logger.info("=== Hyperparameter Tuning Agent Started ===")

    model_name = state["current_model"]
    task_type  = state["task_type"]
    df         = state["processed_data"].to_pandas()
    target_col = state["target_column"]

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # Cap trials: 2 for small datasets, 5 otherwise
    n_samples = X.shape[0]
    n_trials  = 2 if n_samples < 500 else min(config["optuna"]["n_trials"], 5)

    def objective(trial):
        try:
            params = get_search_space(model_name, trial)
            if model_name == "xgboost":
                result = train_and_evaluate_xgboost(X, y, task_type, params, cv_folds=3)
            elif model_name == "lightgbm":
                result = train_and_evaluate_lightgbm(X, y, task_type, params, cv_folds=3)
            else:
                result = train_and_evaluate_sklearn(X, y, model_name, task_type, params, cv_folds=3)
            if task_type == "classification":
                return result["metrics"].get("accuracy", 0.0)
            else:
                # Use CV R² — never train R² which can be 1.0 on overfit models
                return result["metrics"].get("r2_score", -999999)
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0 if task_type == "classification" else -999999

    result     = optimize_hyperparameters(objective, n_trials=n_trials, direction="maximize")
    best_params = result["best_params"]

    logger.info(f"Best Hyperparameters for {model_name}: {best_params}")
    logger.info(f"Best CV Score: {result['best_value']:.4f}")

    state["current_params"] = best_params
    state["agent_logs"].append(f"Hyperparameter Agent: Optimized {model_name} — {n_trials} trials, best={result['best_value']:.4f}")

    logger.info("=== Hyperparameter Tuning Agent Completed ===")
    return state
