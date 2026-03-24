from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from joblib import Memory
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile, numpy as np
from typing import Dict, Any, List, Tuple
from utils.logger import logger

# Shared joblib memory cache for pipeline fitting (cleared per process)
_CACHE_DIR = tempfile.mkdtemp(prefix="automl_pipe_cache_")
_memory    = Memory(location=_CACHE_DIR, verbose=0)

# ── Model registry — ordered simple → complex ─────────────────────────────────
# This ordering is used for logging/reporting; training itself is parallel.

CLASSIFICATION_MODELS: Dict[str, Any] = {
    "logistic_regression": LogisticRegression(max_iter=2000, solver="saga"),   # saga supports L1+L2
    "decision_tree":       DecisionTreeClassifier(random_state=42),
    "random_forest":       RandomForestClassifier(n_jobs=-1, random_state=42),
    "gradient_boosting":   GradientBoostingClassifier(random_state=42),
    "svm":                 SVC(probability=True, max_iter=3000),
    "knn":                 KNeighborsClassifier(n_jobs=-1),
}

REGRESSION_MODELS: Dict[str, Any] = {
    "linear_regression":   LinearRegression(n_jobs=-1),
    "ridge":               Ridge(),                          # L2
    "lasso":               Lasso(max_iter=3000),             # L1
    "elastic_net":         ElasticNet(max_iter=3000),        # L1+L2
    "decision_tree":       DecisionTreeRegressor(random_state=42),
    "random_forest":       RandomForestRegressor(n_jobs=-1, random_state=42),
    "gradient_boosting":   GradientBoostingRegressor(random_state=42),
    "svm":                 SVR(),
    "knn":                 KNeighborsRegressor(n_jobs=-1),
}

# Try to add XGBoost if installed
try:
    from xgboost import XGBClassifier, XGBRegressor
    CLASSIFICATION_MODELS["xgboost"] = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, n_jobs=1,
    )
    REGRESSION_MODELS["xgboost"] = XGBRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, n_jobs=1,
    )
    logger.info("XGBoost available — added to model registry")
except ImportError:
    logger.info("XGBoost not installed — skipping")

# Try to add LightGBM if installed
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    CLASSIFICATION_MODELS["lightgbm"] = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=-1, n_jobs=1,
    )
    REGRESSION_MODELS["lightgbm"] = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=-1, n_jobs=1,
    )
    logger.info("LightGBM available — added to model registry")
except ImportError:
    logger.info("LightGBM not installed — skipping")

# ── HPO search spaces with explicit L1/L2 regularization ─────────────────────

_HPO_GRIDS: Dict[str, Dict[str, Any]] = {
    # L1 (penalty="l1"), L2 (penalty="l2"), ElasticNet (penalty="elasticnet")
    "logistic_regression": {
        "C":          [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty":    ["l1", "l2", "elasticnet"],
        "l1_ratio":   [0.0, 0.25, 0.5, 0.75, 1.0],   # only used for elasticnet
    },
    "decision_tree": {
        "max_depth":        [3, 4, 5, 6, 8, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "min_samples_split":[2, 5, 10, 20],
        "ccp_alpha":        [0.0, 0.001, 0.005, 0.01],  # cost-complexity pruning (anti-overfit)
    },
    "random_forest": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [4, 6, 8, 10, None],
        "min_samples_leaf": [1, 3, 5],
        "max_features":     ["sqrt", "log2", 0.5],
        "min_impurity_decrease": [0.0, 0.001, 0.005],
    },
    "gradient_boosting": {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample":     [0.6, 0.7, 0.8, 1.0],
        "min_samples_leaf": [1, 3, 5],
    },
    "svm": {
        "C":      [0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel": ["rbf", "linear", "poly"],
        "gamma":  ["scale", "auto"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan"],
    },
    "xgboost": {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [3, 4, 5, 6],
        "learning_rate":   [0.01, 0.05, 0.1],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "reg_alpha":       [0.0, 0.1, 0.5, 1.0],   # L1
        "reg_lambda":      [0.5, 1.0, 2.0, 5.0],   # L2
    },
    "linear_regression": {},
    "ridge":       {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "lasso":       {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
    "elastic_net": {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "lightgbm": {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [4, 6, 8, -1],
        "learning_rate":   [0.01, 0.05, 0.1],
        "num_leaves":      [15, 31, 63],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "reg_alpha":       [0.0, 0.1, 0.5],
        "reg_lambda":      [0.5, 1.0, 2.0],
    },
}

_NEEDS_SCALING   = {"logistic_regression", "svm", "knn", "ridge", "lasso", "linear_regression", "elastic_net"}
_HPO_N_ITER      = 10   # RandomizedSearchCV iterations per model (phase 2 / full data)
_HPO_N_ITER_FAST = 5    # phase 1 screening iterations
_SUBSAMPLE_RATIO = 0.4  # fraction used for phase-1 model screening
_TOP_K_MODELS    = 5    # top-K models promoted to phase-2 full training
# Models that support early stopping via eval_set
_EARLY_STOP_MODELS = {"xgboost", "lightgbm"}


# ── Adaptive CV folds: 5 for small datasets, up to 10 for large ───────────────

def _adaptive_cv_folds(n_train: int, n_classes: int, min_folds: int = 5, max_folds: int = 10) -> int:
    """Scale CV folds with dataset size: 5 folds for <500 rows, 10 for >5000."""
    if n_train < 200:
        folds = min_folds
    elif n_train < 500:
        folds = 5
    elif n_train < 2000:
        folds = 7
    else:
        folds = max_folds
    # Never exceed n_train // (n_classes * 2) to avoid empty folds
    safe = max(2, n_train // max(n_classes, 1) // 2)
    return min(folds, safe)


def _subsample(X: np.ndarray, y: np.ndarray, task_type: str, ratio: float = _SUBSAMPLE_RATIO):
    """Stratified subsample for phase-1 screening. Returns full data if already small."""
    n = len(y)
    n_sub = max(200, int(n * ratio))
    if n_sub >= n:
        return X, y
    if task_type == "classification":
        X_s, y_s = resample(X, y, n_samples=n_sub, stratify=y, random_state=42)
    else:
        idx = np.random.default_rng(42).choice(n, size=n_sub, replace=False)
        X_s, y_s = X[idx], y[idx]
    return X_s, y_s


def _build_pipe(model_name: str, model, task_type: str, y_train: np.ndarray, use_smote: bool, cached: bool = False):
    scale = model_name in _NEEDS_SCALING
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    if use_smote and task_type == "classification":
        k = max(1, min(5, int(np.bincount(y_train.astype(int)).min()) - 1))
        steps.append(("smote", SMOTE(random_state=42, k_neighbors=k)))
    steps.append(("model", model))
    if use_smote and task_type == "classification":
        return ImbPipeline(steps, memory=_memory if cached else None)
    if scale:
        return Pipeline(steps, memory=_memory if cached else None)
    return model


def _count_combinations(grid: Dict[str, Any]) -> int:
    total = 1
    for v in grid.values():
        total *= len(v)
    return total


def _run_hpo(model_name: str, model, X_train: np.ndarray, y_train: np.ndarray,
             task_type: str, cv_folds: int, n_iter_override: int = None):
    grid = _HPO_GRIDS.get(model_name, {})
    if not grid:
        return model, {}

    scoring = "accuracy" if task_type == "classification" else "r2"
    n_classes = len(np.unique(y_train)) if task_type == "classification" else 1
    safe_folds = _adaptive_cv_folds(len(X_train), n_classes)

    n_iter = min(n_iter_override or _HPO_N_ITER, _count_combinations(grid))
    search = RandomizedSearchCV(
        model, grid,
        n_iter=n_iter,
        cv=safe_folds,
        scoring=scoring,
        random_state=42,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    try:
        search.fit(X_train, y_train)
        logger.info(f"  [{model_name}] HPO best: {search.best_params_}  cv={search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_
    except Exception as e:
        logger.warning(f"  [{model_name}] HPO failed ({e}), using defaults")
        return model, {}


# ── Anti-overfitting post-processing ─────────────────────────────────────────

def _apply_regularization_fallback(model_name: str, model, overfit_gap: float):
    """
    If a model severely overfits (gap > 0.20), apply stronger regularization
    and refit. Returns the adjusted model (or original if not applicable).
    """
    if overfit_gap <= 0.20:
        return model, False

    import copy
    m = copy.deepcopy(model)
    try:
        if model_name == "decision_tree":
            m.set_params(max_depth=min(m.get_params().get("max_depth") or 10, 4), ccp_alpha=0.01)
        elif model_name == "random_forest":
            m.set_params(max_depth=min(m.get_params().get("max_depth") or 10, 6), min_samples_leaf=5)
        elif model_name == "gradient_boosting":
            m.set_params(subsample=0.6, max_depth=3, min_samples_leaf=5)
        elif model_name == "xgboost":
            m.set_params(reg_alpha=1.0, reg_lambda=5.0, max_depth=3, subsample=0.6)
        elif model_name == "logistic_regression":
            m.set_params(C=0.01)
        elif model_name == "svm":
            m.set_params(C=0.1)
        else:
            return model, False
        return m, True
    except Exception:
        return model, False


def _fit_with_early_stopping(model_name: str, model, X_train, y_train, X_val, y_val, task_type: str):
    """Fit XGBoost/LightGBM with early stopping on a validation split."""
    try:
        if model_name == "xgboost":
            model.set_params(early_stopping_rounds=20, n_estimators=500)
            eval_metric = "logloss" if task_type == "classification" else "rmse"
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric=eval_metric,
                      verbose=False)
        elif model_name == "lightgbm":
            model.set_params(n_estimators=500)
            callbacks = []
            try:
                import lightgbm as lgb
                callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
            except Exception:
                pass
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=callbacks)
        else:
            model.fit(X_train, y_train)
        return model
    except Exception as e:
        logger.warning(f"  [{model_name}] early stopping fit failed ({e}), falling back")
        model.fit(X_train, y_train)
        return model


def _train_single(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    cv_folds: int,
    use_smote: bool,
    hpo_n_iter: int = _HPO_N_ITER,
) -> Dict[str, Any]:
    try:
        # ── 1. HPO on train set ───────────────────────────────────────────────
        best_params: Dict[str, Any] = {}
        grid = _HPO_GRIDS.get(model_name, {})
        if grid:
            model, best_params = _run_hpo(model_name, model, X_train, y_train, task_type, cv_folds,
                                          n_iter_override=hpo_n_iter)

        # ── 2. Build pipeline (scaler + SMOTE + model) ────────────────────────
        pipe = _build_pipe(model_name, model, task_type, y_train, use_smote, cached=True)

        # ── 3. Adaptive CV on train only ──────────────────────────────────────
        n_classes = len(np.unique(y_train)) if task_type == "classification" else 1
        safe_folds = _adaptive_cv_folds(len(y_train), n_classes)

        if task_type == "classification":
            cv = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=42)
            cv_scores     = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
            cv_roc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc_ovr", n_jobs=-1)
        else:
            cv = KFold(n_splits=safe_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)

        cv_score = float(np.mean(cv_scores))
        cv_std   = float(np.std(cv_scores))
        cv_roc   = float(np.mean(cv_roc_scores)) if task_type == "classification" else None

        # ── 4. Final fit + predict (early stopping for XGB/LGB) ──────────────
        if model_name in _EARLY_STOP_MODELS and not isinstance(pipe, (Pipeline, ImbPipeline)):
            # No scaler/SMOTE wrapping — fit directly with early stopping
            X_tr_es, X_val_es, y_tr_es, y_val_es = train_test_split(
                X_train, y_train, test_size=0.15, random_state=0,
                stratify=y_train if task_type == "classification" else None
            )
            pipe = _fit_with_early_stopping(model_name, pipe, X_tr_es, y_tr_es, X_val_es, y_val_es, task_type)
        else:
            pipe.fit(X_train, y_train)
        train_pred = pipe.predict(X_train)
        test_pred  = pipe.predict(X_test)

        # Extract raw model + scaled X_test for ROC/SHAP
        if isinstance(pipe, (Pipeline, ImbPipeline)):
            fitted_model = pipe.named_steps["model"]
            X_test_t = pipe.named_steps["scaler"].transform(X_test) if "scaler" in pipe.named_steps else X_test
        else:
            fitted_model = pipe
            X_test_t = X_test

        # ── 5. Metrics ────────────────────────────────────────────────────────
        if task_type == "classification":
            train_score = float(accuracy_score(y_train, train_pred))
            test_score  = float(accuracy_score(y_test,  test_pred))
            overfit_gap = round(train_score - test_score, 4)

            # Anti-overfit fallback: re-regularize and refit if gap > 20%
            if overfit_gap > 0.20:
                adj_model, applied = _apply_regularization_fallback(model_name, fitted_model, overfit_gap)
                if applied:
                    adj_pipe = _build_pipe(model_name, adj_model, task_type, y_train, use_smote)
                    adj_pipe.fit(X_train, y_train)
                    adj_train = adj_pipe.predict(X_train)
                    adj_test  = adj_pipe.predict(X_test)
                    adj_gap   = round(float(accuracy_score(y_train, adj_train)) - float(accuracy_score(y_test, adj_test)), 4)
                    if adj_gap < overfit_gap:
                        pipe, fitted_model = adj_pipe, adj_model
                        train_pred, test_pred = adj_train, adj_test
                        train_score = float(accuracy_score(y_train, train_pred))
                        test_score  = float(accuracy_score(y_test,  test_pred))
                        overfit_gap = round(train_score - test_score, 4)
                        logger.info(f"  [{model_name}] Regularization fallback applied, new gap={overfit_gap:.4f}")

            # F1 + ROC-AUC on test set
            try:
                test_f1 = float(f1_score(y_test, test_pred, average="weighted"))
            except Exception:
                test_f1 = 0.0
            try:
                from sklearn.metrics import roc_auc_score
                if hasattr(pipe, "predict_proba"):
                    y_prob = pipe.predict_proba(X_test)
                    test_roc = float(roc_auc_score(y_test, y_prob if y_prob.shape[1] > 2 else y_prob[:, 1],
                                                   multi_class="ovr", average="weighted"))
                else:
                    test_roc = cv_roc
            except Exception:
                test_roc = cv_roc or cv_score

            metrics = {
                "accuracy":        round(cv_score, 4),
                "test_accuracy":   round(test_score, 4),
                "train_accuracy":  round(train_score, 4),
                "cv_std":          round(cv_std, 4),
                "cv_folds":        safe_folds,
                "overfit_gap":     overfit_gap,
                "overfit_penalty": 0.0,
                "test_set_size":   int(len(y_test)),
                "train_set_size":  int(len(y_train)),
                "best_params":     best_params,
                "f1_score":        round(test_f1, 4),
                "roc_auc":         round(test_roc, 4) if test_roc else None,
                "cv_roc_auc":      round(cv_roc, 4) if cv_roc else None,
            }

            # Composite: 40% CV-ROC-AUC + 30% test-F1 + 20% test-accuracy + 10% stability
            # ROC-AUC + F1 are far better than raw accuracy for imbalanced datasets (e.g. diabetes)
            stability = max(0.0, 1.0 - cv_std * 10)
            primary   = 0.4 * (cv_roc or cv_score) + 0.3 * test_f1 + 0.2 * test_score + 0.1 * stability
            if overfit_gap > 0.10:
                penalty = (overfit_gap - 0.10) * 0.8
                primary = max(0.0, primary - penalty)
                metrics["overfit_penalty"] = round(penalty, 4)

        else:
            train_r2    = float(r2_score(y_train, train_pred))
            test_r2     = float(r2_score(y_test,  test_pred))
            overfit_gap = round(train_r2 - test_r2, 4)
            rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

            if overfit_gap > 0.20:
                adj_model, applied = _apply_regularization_fallback(model_name, fitted_model, overfit_gap)
                if applied:
                    adj_pipe = _build_pipe(model_name, adj_model, task_type, y_train, use_smote)
                    adj_pipe.fit(X_train, y_train)
                    adj_train = adj_pipe.predict(X_train)
                    adj_test  = adj_pipe.predict(X_test)
                    adj_gap   = round(float(r2_score(y_train, adj_train)) - float(r2_score(y_test, adj_test)), 4)
                    if adj_gap < overfit_gap:
                        pipe, fitted_model = adj_pipe, adj_model
                        train_pred, test_pred = adj_train, adj_test
                        train_r2    = float(r2_score(y_train, train_pred))
                        test_r2     = float(r2_score(y_test,  test_pred))
                        overfit_gap = round(train_r2 - test_r2, 4)
                        rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
                        logger.info(f"  [{model_name}] Regularization fallback applied, new gap={overfit_gap:.4f}")

            metrics = {
                "r2_score":        round(cv_score, 4),
                "test_r2":         round(test_r2, 4),
                "train_r2":        round(train_r2, 4),
                "rmse":            round(rmse, 4),
                "mae":             round(float(mean_absolute_error(y_test, test_pred)), 4),
                "cv_std":          round(cv_std, 4),
                "cv_folds":        safe_folds,
                "overfit_gap":     overfit_gap,
                "overfit_penalty": 0.0,
                "test_set_size":   int(len(y_test)),
                "train_set_size":  int(len(y_train)),
                "best_params":     best_params,
            }
            stability = max(0.0, 1.0 - cv_std * 10)
            primary   = 0.5 * cv_score + 0.3 * test_r2 + 0.2 * stability
            if overfit_gap > 0.10:
                penalty = (overfit_gap - 0.10) * 0.8
                primary = max(-1.0, primary - penalty)
                metrics["overfit_penalty"] = round(penalty, 4)

        logger.info(
            f"  [{model_name}] composite={primary:.4f}  cv={cv_score:.4f}({safe_folds}-fold)"
            f"  test={test_score if task_type == 'classification' else test_r2:.4f}"
            f"  std={cv_std:.4f}  gap={overfit_gap:.4f}"
        )
        return {
            "model_name": model_name,
            "model":      fitted_model,
            "metrics":    metrics,
            "score":      round(primary, 4),
            "X_test":     X_test_t,
            "y_test":     y_test,
            "params":     best_params,
            "iteration":  1,
        }

    except Exception as e:
        logger.error(f"  [{model_name}] FAILED: {e}", exc_info=True)
        return {
            "model_name": model_name,
            "model":      None,
            "metrics":    {},
            "score":      -999.0,
            "X_test":     X_test,
            "y_test":     y_test,
            "params":     {},
            "iteration":  1,
            "error":      str(e),
        }


def train_all_models_parallel(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    cv_folds: int = 5,
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Two-phase progressive training:
      Phase 1 — all models on subsample, fast HPO, pick top-K  (skipped for small datasets)
      Phase 2 — top-K retrained on full data, full HPO + early stopping.
    Composite ranking: 0.5*cv + 0.3*test + 0.2*stability, penalised for overfit gap.
    """
    import copy

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    use_smote = False
    if task_type == "classification":
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) > 1 and counts.min() >= 6 and (counts.max() / counts.min()) > 3:
            use_smote = True

    model_registry = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    n_models = len(model_registry)
    effective_workers = min(max_workers, n_models, 8)

    # ── Skip Phase 1 for small datasets — subsample is too noisy to be useful ─
    small_dataset = len(y_train) < 500

    if small_dataset:
        logger.info(
            f"Small dataset ({len(y_train)} rows) — skipping Phase 1, training all "
            f"{n_models} {task_type} models on full data | HPO={_HPO_N_ITER} iter | workers={effective_workers}"
        )
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    _train_single,
                    name, copy.deepcopy(model),
                    X_train, X_test, y_train, y_test,
                    task_type, cv_folds, use_smote,
                    _HPO_N_ITER,
                ): name
                for name, model in model_registry.items()
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda r: r["score"], reverse=True)
        logger.info(f"Final ranking: {[(r['model_name'], r['score']) for r in results]}")
        return results

    # ── Phase 1: screen all models on subsample ───────────────────────────────
    X_sub, y_sub = _subsample(X_train, y_train, task_type)
    n_folds_sub  = _adaptive_cv_folds(len(y_sub), len(np.unique(y_sub)) if task_type == "classification" else 1)
    logger.info(
        f"Phase 1 screening: {n_models} {task_type} models on "
        f"{len(y_sub)}/{len(y_train)} samples ({_SUBSAMPLE_RATIO:.0%}) | "
        f"CV={n_folds_sub}-fold | HPO={_HPO_N_ITER_FAST} iter | workers={effective_workers}"
    )

    phase1: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(
                _train_single,
                name, copy.deepcopy(model),
                X_sub, X_test, y_sub, y_test,
                task_type, cv_folds, use_smote,
                _HPO_N_ITER_FAST,
            ): name
            for name, model in model_registry.items()
        }
        for future in as_completed(futures):
            phase1.append(future.result())

    phase1.sort(key=lambda r: r["score"], reverse=True)
    top_names = [r["model_name"] for r in phase1 if r["score"] > -999.0][:_TOP_K_MODELS]
    logger.info(f"Phase 1 top-{_TOP_K_MODELS}: {top_names}")

    # ── Phase 2: full training on top-K models ────────────────────────────────
    n_folds_full = _adaptive_cv_folds(len(y_train), len(np.unique(y_train)) if task_type == "classification" else 1)
    logger.info(
        f"Phase 2 full training: {top_names} on {len(y_train)} samples | "
        f"CV={n_folds_full}-fold | HPO={_HPO_N_ITER} iter | early_stopping=XGB/LGB"
    )

    phase2: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(effective_workers, _TOP_K_MODELS)) as executor:
        futures = {
            executor.submit(
                _train_single,
                name, copy.deepcopy(model_registry[name]),
                X_train, X_test, y_train, y_test,
                task_type, cv_folds, use_smote,
                _HPO_N_ITER,
            ): name
            for name in top_names
        }
        for future in as_completed(futures):
            phase2.append(future.result())

    # Merge: phase2 results for top-K, phase1 results for the rest
    top_set = set(top_names)
    rest    = [r for r in phase1 if r["model_name"] not in top_set]
    results = phase2 + rest
    results.sort(key=lambda r: r["score"], reverse=True)
    logger.info(f"Final ranking: {[(r['model_name'], r['score']) for r in results]}")
    return results


# ── Legacy single-model API ───────────────────────────────────────────────────

def get_sklearn_model(model_name: str, task_type: str, params: Dict[str, Any] = None):
    import copy
    params = params or {}
    all_models = {**CLASSIFICATION_MODELS, **REGRESSION_MODELS}
    if model_name not in all_models:
        raise ValueError(f"Model '{model_name}' not in registry")
    model = copy.deepcopy(all_models[model_name])
    if params:
        model.set_params(**params)
    return model


def train_and_evaluate_sklearn(
    X, y, model_name: str, task_type: str, params: Dict[str, Any], cv_folds: int = 5
) -> Dict[str, Any]:
    import copy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )
    all_models = {**CLASSIFICATION_MODELS, **REGRESSION_MODELS}
    model = copy.deepcopy(all_models.get(model_name))
    if model is None:
        raise ValueError(f"Model '{model_name}' not found")
    if params:
        model.set_params(**params)
    use_smote = False
    if task_type == "classification":
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) > 1 and counts.min() >= 6 and (counts.max() / counts.min()) > 3:
            use_smote = True
    return _train_single(model_name, model, X_train, X_test, y_train, y_test, task_type, cv_folds, use_smote)