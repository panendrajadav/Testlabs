import optuna
from typing import Dict, Any, Callable
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_search_space(model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    """Define hyperparameter search space for each model."""
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    elif model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    elif model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        }
    elif model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
        }
    elif model_name == "svm":
        return {
            "C": trial.suggest_float("C", 0.1, 10.0),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        }
    elif model_name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
    elif model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    elif model_name == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
        }
    elif model_name == "lasso":
        return {
            "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
        }
    elif model_name == "gradient_boosting":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
    elif model_name == "extra_trees":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 300),
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        }
    else:
        return {}

def optimize_hyperparameters(
    objective_fn: Callable,
    n_trials: int = 30,
    direction: str = "maximize"
) -> Dict[str, Any]:
    """Run Optuna optimization study."""
    study = optuna.create_study(direction=direction)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
