import optuna
from typing import Dict, Any, Callable
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_search_space(model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    """Define hyperparameter search space for each model."""
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
    elif model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }
    elif model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 60),
        }
    elif model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 200, 2000),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        }
    elif model_name == "svm":
        return {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    elif model_name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }
    elif model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
        }
    elif model_name == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky"]),
        }
    elif model_name == "lasso":
        return {
            "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 500, 3000),
        }
    elif model_name == "gradient_boosting":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.05, 0.3),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
    elif model_name == "extra_trees":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 150),
            "max_depth":         trial.suggest_int("max_depth", 3, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 5),
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
