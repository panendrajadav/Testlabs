from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from typing import Dict, Any, Tuple

def get_sklearn_model(model_name: str, task_type: str, params: Dict[str, Any] = None):
    """Get sklearn model instance."""
    params = params or {}
    
    models = {
        "classification": {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "decision_tree": DecisionTreeClassifier,
            "knn": KNeighborsClassifier,
            "svm": SVC,
        },
        "regression": {
            "random_forest": RandomForestRegressor,
            "decision_tree": DecisionTreeRegressor,
            "knn": KNeighborsRegressor,
            "svm": SVR,
            "ridge": Ridge,
            "lasso": Lasso,
        }
    }
    
    model_class = models.get(task_type, {}).get(model_name)
    if not model_class:
        raise ValueError(f"Model {model_name} not supported for {task_type}")
    
    return model_class(**params)

def train_and_evaluate_sklearn(
    X, y, model_name: str, task_type: str, params: Dict[str, Any], cv_folds: int = 5
) -> Dict[str, Any]:
    """Train and evaluate sklearn model with cross-validation."""
    model = get_sklearn_model(model_name, task_type, params)
    
    if task_type == "classification":
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
            
    else:  # regression
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
