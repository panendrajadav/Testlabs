from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from typing import Dict, Any

def get_sklearn_model(model_name: str, task_type: str, params: Dict[str, Any] = None):
    params = params or {}
    models = {
        "classification": {
            "logistic_regression": LogisticRegression,
            "random_forest":       RandomForestClassifier,
            "decision_tree":       DecisionTreeClassifier,
            "knn":                 KNeighborsClassifier,
            "svm":                 SVC,
            "gradient_boosting":   GradientBoostingClassifier,
            "extra_trees":         ExtraTreesClassifier,
        },
        "regression": {
            "random_forest":     RandomForestRegressor,
            "decision_tree":     DecisionTreeRegressor,
            "knn":               KNeighborsRegressor,
            "svm":               SVR,
            "ridge":             Ridge,
            "lasso":             Lasso,
            "gradient_boosting": GradientBoostingRegressor,
            "extra_trees":       ExtraTreesRegressor,
        }
    }
    model_class = models.get(task_type, {}).get(model_name)
    if not model_class:
        raise ValueError(f"Model {model_name} not supported for {task_type}")
    # Safe defaults injected before HPO params override them
    defaults = {
        "logistic_regression": {"max_iter": 1000, "solver": "lbfgs"},
        "svm":                 {"gamma": "scale", "max_iter": 2000},
    }
    merged = {**defaults.get(model_name, {}), **params}
    # SVC needs probability=True for predict_proba / ROC
    if model_name == "svm" and task_type == "classification":
        merged["probability"] = True
    return model_class(**merged)


def train_and_evaluate_sklearn(
    X, y, model_name: str, task_type: str, params: Dict[str, Any], cv_folds: int = 5
) -> Dict[str, Any]:
    model = get_sklearn_model(model_name, task_type, params)

    # Hold-out split for unbiased test evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    if task_type == "classification":
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1)
        cv_score  = float(np.mean(cv_scores))
        cv_std    = float(np.std(cv_scores))

        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)

        train_score = float(accuracy_score(y_train, train_pred))
        test_score  = float(accuracy_score(y_test,  test_pred))
        overfit_gap = round(train_score - cv_score, 4)

        metrics = {
            "accuracy":    round(cv_score, 4),       # CV accuracy — primary metric
            "test_accuracy": round(test_score, 4),   # held-out test
            "train_accuracy": round(train_score, 4),
            "cv_std":      round(cv_std, 4),
            "overfit_gap": overfit_gap,
        }
        try:
            metrics["f1_score"] = round(float(f1_score(y_test, test_pred, average="weighted")), 4)
        except Exception:
            pass

    else:  # regression
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1)
        cv_score  = float(np.mean(cv_scores))
        cv_std    = float(np.std(cv_scores))

        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)

        train_r2  = float(r2_score(y_train, train_pred))
        test_r2   = float(r2_score(y_test,  test_pred))
        overfit_gap = round(train_r2 - cv_score, 4)

        metrics = {
            "r2_score":    round(cv_score, 4),        # CV R² — primary metric
            "test_r2":     round(test_r2, 4),         # held-out test
            "train_r2":    round(train_r2, 4),
            "rmse":        round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 4),
            "mae":         round(float(mean_absolute_error(y_test, test_pred)), 4),
            "cv_std":      round(cv_std, 4),
            "overfit_gap": overfit_gap,
        }

    return {"model": model, "metrics": metrics, "X_test": X_test, "y_test": y_test}
