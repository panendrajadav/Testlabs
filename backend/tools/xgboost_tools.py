import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from typing import Dict, Any

def train_and_evaluate_xgboost(
    X, y, task_type: str, params: Dict[str, Any], cv_folds: int = 5
) -> Dict[str, Any]:
    """Train and evaluate XGBoost model."""
    if task_type == "classification":
        model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
        scoring = 'accuracy'
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        metrics = {
            "accuracy": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "train_accuracy": float(accuracy_score(y, y_pred)),
        }
        
        try:
            metrics["f1_score"] = float(f1_score(y, y_pred, average='weighted'))
        except:
            pass
    else:
        model = xgb.XGBRegressor(**params, random_state=42)
        scoring = 'neg_mean_squared_error'
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        metrics = {
            "rmse": float(np.sqrt(-np.mean(cv_scores))),
            "cv_std": float(np.std(cv_scores)),
            "r2_score": float(r2_score(y, y_pred)),
            "mae": float(mean_absolute_error(y, y_pred)),
        }
    
    return {"model": model, "metrics": metrics}

def train_and_evaluate_lightgbm(
    X, y, task_type: str, params: Dict[str, Any], cv_folds: int = 5
) -> Dict[str, Any]:
    """Train and evaluate LightGBM model."""
    if task_type == "classification":
        model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
        scoring = 'accuracy'
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        metrics = {
            "accuracy": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "train_accuracy": float(accuracy_score(y, y_pred)),
        }
        
        try:
            metrics["f1_score"] = float(f1_score(y, y_pred, average='weighted'))
        except:
            pass
    else:
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        scoring = 'neg_mean_squared_error'
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        metrics = {
            "rmse": float(np.sqrt(-np.mean(cv_scores))),
            "cv_std": float(np.std(cv_scores)),
            "r2_score": float(r2_score(y, y_pred)),
            "mae": float(mean_absolute_error(y, y_pred)),
        }
    
    return {"model": model, "metrics": metrics}
