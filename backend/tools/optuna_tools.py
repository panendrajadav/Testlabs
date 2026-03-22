import optuna
from typing import Dict, Any, Callable
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_search_space(model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    """Hyperparameter search spaces with regularisation floors to prevent overfitting."""
    if model_name == "random_forest":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 200),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),      # cap at 10
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),  # floor at 5
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 2, 10),   # floor at 2
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    elif model_name == "xgboost":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 200),
            "max_depth":       trial.suggest_int("max_depth", 3, 7),         # cap at 7
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample":       trial.suggest_float("subsample", 0.6, 0.9),   # no 1.0
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 1.0),   # L1
            "reg_lambda":      trial.suggest_float("reg_lambda", 1.0, 5.0),  # L2 floor at 1
            "min_child_weight":trial.suggest_int("min_child_weight", 3, 10), # floor at 3
        }
    elif model_name == "lightgbm":
        return {
            "n_estimators":   trial.suggest_int("n_estimators", 50, 200),
            "max_depth":      trial.suggest_int("max_depth", 3, 7),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.2),
            "num_leaves":     trial.suggest_int("num_leaves", 15, 50),       # cap at 50
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 30),  # floor at 10
            "reg_alpha":      trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":     trial.suggest_float("reg_lambda", 1.0, 5.0),
        }
    elif model_name == "logistic_regression":
        return {
            "C":        trial.suggest_float("C", 0.001, 1.0, log=True),  # cap at 1.0 (more regularisation)
            "max_iter": trial.suggest_int("max_iter", 500, 2000),
            "solver":   trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        }
    elif model_name == "svm":
        return {
            "C":      trial.suggest_float("C", 0.01, 5.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
            "gamma":  trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    elif model_name == "decision_tree":
        return {
            "max_depth":         trial.suggest_int("max_depth", 2, 8),       # cap at 8
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),  # floor at 5
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 3, 15),   # floor at 3
            "criterion":         trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "ccp_alpha":         trial.suggest_float("ccp_alpha", 0.0, 0.05),    # pruning
        }
    elif model_name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 25),  # floor at 5 (no k=1)
            "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric":      trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
        }
    elif model_name == "ridge":
        return {
            "alpha":  trial.suggest_float("alpha", 0.1, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky"]),
        }
    elif model_name == "lasso":
        return {
            "alpha":    trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 1000, 3000),
        }
    elif model_name == "gradient_boosting":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 200),
            "max_depth":         trial.suggest_int("max_depth", 2, 6),       # cap at 6
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
            "subsample":         trial.suggest_float("subsample", 0.6, 0.9),
        }
    elif model_name == "extra_trees":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 200),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 2, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    elif model_name == "elastic_net":
        return {
            "alpha":    trial.suggest_float("alpha", 0.01, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            "max_iter": trial.suggest_int("max_iter", 1000, 3000),
        }
    elif model_name == "huber":
        return {
            "epsilon": trial.suggest_float("epsilon", 1.1, 2.5),
            "alpha":   trial.suggest_float("alpha", 1e-4, 1.0, log=True),
        }
    elif model_name == "bayesian_ridge":
        return {}   # no meaningful HPO — priors are auto-fitted
    elif model_name == "naive_bayes":
        return {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-6, log=True),
        }
    elif model_name == "linear_discriminant":
        return {
            "solver": trial.suggest_categorical("solver", ["svd", "lsqr"]),
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
