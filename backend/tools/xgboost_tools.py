import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Dict, Any


def _safe_cv_folds(n_samples: int, n_classes: int, requested_folds: int) -> int:
    max_folds = max(2, min(requested_folds, n_samples // max(n_classes, 1) // 2))
    return min(requested_folds, max_folds)


def _xgb_fit_early_stop(model, X_train, y_train, X_val, y_val, task_type: str):
    eval_metric = "logloss" if task_type == "classification" else "rmse"
    model.set_params(early_stopping_rounds=20, n_estimators=500)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              eval_metric=eval_metric, verbose=False)
    return model


def train_and_evaluate_xgboost(
    X, y, task_type: str, params: Dict[str, Any], cv_folds: int = 5
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )
    # 15% of train as early-stopping validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=0,
        stratify=y_train if task_type == "classification" else None
    )

    if task_type == "classification":
        model = xgb.XGBClassifier(**params, random_state=42, verbosity=0, n_jobs=-1)
        n_classes = len(np.unique(y_train))
        safe_folds = _safe_cv_folds(len(y_train), n_classes, cv_folds)
        cv = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=42)
        cv_scores   = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_score    = float(np.mean(cv_scores))
        model = _xgb_fit_early_stop(model, X_tr, y_tr, X_val, y_val, task_type)
        train_score = float(accuracy_score(y_train, model.predict(X_train)))
        test_pred   = model.predict(X_test)
        test_score  = float(accuracy_score(y_test, test_pred))
        metrics = {
            "accuracy":       round(cv_score, 4),
            "test_accuracy":  round(test_score, 4),
            "train_accuracy": round(train_score, 4),
            "cv_std":         round(float(np.std(cv_scores)), 4),
            "overfit_gap":    round(train_score - test_score, 4),
        }
        try:
            metrics["f1_score"] = round(float(f1_score(y_test, test_pred, average="weighted")), 4)
        except Exception:
            pass
    else:
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1)
        safe_folds = max(2, min(cv_folds, len(y_train) // 5))
        cv = KFold(n_splits=safe_folds, shuffle=True, random_state=42)
        cv_scores  = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_score   = float(np.mean(cv_scores))
        model = _xgb_fit_early_stop(model, X_tr, y_tr, X_val, y_val, task_type)
        train_r2   = float(r2_score(y_train, model.predict(X_train)))
        test_pred  = model.predict(X_test)
        metrics = {
            "r2_score":    round(cv_score, 4),
            "test_r2":     round(float(r2_score(y_test, test_pred)), 4),
            "train_r2":    round(train_r2, 4),
            "rmse":        round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 4),
            "mae":         round(float(mean_absolute_error(y_test, test_pred)), 4),
            "cv_std":      round(float(np.std(cv_scores)), 4),
            "overfit_gap": round(train_r2 - float(r2_score(y_test, test_pred)), 4),
        }

    return {"model": model, "metrics": metrics, "X_test": X_test, "y_test": y_test}


def _lgb_fit_early_stop(model, X_train, y_train, X_val, y_val):
    callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
    model.set_params(n_estimators=500)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return model


def train_and_evaluate_lightgbm(
    X, y, task_type: str, params: Dict[str, Any], cv_folds: int = 5
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=0,
        stratify=y_train if task_type == "classification" else None
    )

    if task_type == "classification":
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
        n_classes = len(np.unique(y_train))
        safe_folds = _safe_cv_folds(len(y_train), n_classes, cv_folds)
        cv = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=42)
        cv_scores   = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_score    = float(np.mean(cv_scores))
        model = _lgb_fit_early_stop(model, X_tr, y_tr, X_val, y_val)
        train_score = float(accuracy_score(y_train, model.predict(X_train)))
        test_pred   = model.predict(X_test)
        test_score  = float(accuracy_score(y_test, test_pred))
        metrics = {
            "accuracy":       round(cv_score, 4),
            "test_accuracy":  round(test_score, 4),
            "train_accuracy": round(train_score, 4),
            "cv_std":         round(float(np.std(cv_scores)), 4),
            "overfit_gap":    round(train_score - test_score, 4),
        }
        try:
            metrics["f1_score"] = round(float(f1_score(y_test, test_pred, average="weighted")), 4)
        except Exception:
            pass
    else:
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=-1)
        safe_folds = max(2, min(cv_folds, len(y_train) // 5))
        cv = KFold(n_splits=safe_folds, shuffle=True, random_state=42)
        cv_scores  = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_score   = float(np.mean(cv_scores))
        model = _lgb_fit_early_stop(model, X_tr, y_tr, X_val, y_val)
        train_r2   = float(r2_score(y_train, model.predict(X_train)))
        test_pred  = model.predict(X_test)
        metrics = {
            "r2_score":    round(cv_score, 4),
            "test_r2":     round(float(r2_score(y_test, test_pred)), 4),
            "train_r2":    round(train_r2, 4),
            "rmse":        round(float(np.sqrt(mean_squared_error(y_test, test_pred))), 4),
            "mae":         round(float(mean_absolute_error(y_test, test_pred)), 4),
            "cv_std":      round(float(np.std(cv_scores)), 4),
            "overfit_gap": round(train_r2 - float(r2_score(y_test, test_pred)), 4),
        }

    return {"model": model, "metrics": metrics, "X_test": X_test, "y_test": y_test}
