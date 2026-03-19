import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from typing import Dict, Any


def train_and_evaluate_xgboost(
    X, y, task_type: str, params: Dict[str, Any], cv_folds: int = 5
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    if task_type == "classification":
        model = xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1)
        cv_scores   = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1)
        cv_score    = float(np.mean(cv_scores))
        model.fit(X_train, y_train)
        train_score = float(accuracy_score(y_train, model.predict(X_train)))
        test_pred   = model.predict(X_test)
        test_score  = float(accuracy_score(y_test, test_pred))
        metrics = {
            "accuracy":      round(cv_score, 4),
            "test_accuracy": round(test_score, 4),
            "train_accuracy": round(train_score, 4),
            "cv_std":        round(float(np.std(cv_scores)), 4),
            "overfit_gap":   round(train_score - cv_score, 4),
        }
        try:
            metrics["f1_score"] = round(float(f1_score(y_test, test_pred, average="weighted")), 4)
        except Exception:
            pass
    else:
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1)
        cv_scores  = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1)
        cv_score   = float(np.mean(cv_scores))
        model.fit(X_train, y_train)
        train_r2   = float(r2_score(y_train, model.predict(X_train)))
        test_pred  = model.predict(X_test)
        metrics = {
            "r2_score":    round(cv_score, 4),
            "test_r2":     round(float(r2_score(y_test, test_pred)), 4),
            "train_r2":    round(train_r2, 4),
            "rmse":        round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 4),
            "mae":         round(float(mean_absolute_error(y_test, test_pred)), 4),
            "cv_std":      round(float(np.std(cv_scores)), 4),
            "overfit_gap": round(train_r2 - cv_score, 4),
        }

    return {"model": model, "metrics": metrics, "X_test": X_test, "y_test": y_test}


def train_and_evaluate_lightgbm(
    X, y, task_type: str, params: Dict[str, Any], cv_folds: int = 5
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    if task_type == "classification":
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
        cv_scores   = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1)
        cv_score    = float(np.mean(cv_scores))
        model.fit(X_train, y_train)
        train_score = float(accuracy_score(y_train, model.predict(X_train)))
        test_pred   = model.predict(X_test)
        test_score  = float(accuracy_score(y_test, test_pred))
        metrics = {
            "accuracy":      round(cv_score, 4),
            "test_accuracy": round(test_score, 4),
            "train_accuracy": round(train_score, 4),
            "cv_std":        round(float(np.std(cv_scores)), 4),
            "overfit_gap":   round(train_score - cv_score, 4),
        }
        try:
            metrics["f1_score"] = round(float(f1_score(y_test, test_pred, average="weighted")), 4)
        except Exception:
            pass
    else:
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=-1)
        cv_scores  = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1)
        cv_score   = float(np.mean(cv_scores))
        model.fit(X_train, y_train)
        train_r2   = float(r2_score(y_train, model.predict(X_train)))
        test_pred  = model.predict(X_test)
        metrics = {
            "r2_score":    round(cv_score, 4),
            "test_r2":     round(float(r2_score(y_test, test_pred)), 4),
            "train_r2":    round(train_r2, 4),
            "rmse":        round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 4),
            "mae":         round(float(mean_absolute_error(y_test, test_pred)), 4),
            "cv_std":      round(float(np.std(cv_scores)), 4),
            "overfit_gap": round(train_r2 - cv_score, 4),
        }

    return {"model": model, "metrics": metrics, "X_test": X_test, "y_test": y_test}
