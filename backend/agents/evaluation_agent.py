from typing import Dict, Any, List
from graph.state import AutoMLState
from utils.logger import logger
from tools.sklearn_tools import train_all_models_parallel
import numpy as np


# ── ROC curve ─────────────────────────────────────────────────────────────────
def _compute_roc(model, X, y, task_type: str) -> Dict[str, Any]:
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        if task_type != "classification":
            return {}
        classes = np.unique(y)

        if len(classes) == 2:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X)
            else:
                return {}
            fpr, tpr, _ = roc_curve(y, y_score)
            return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(auc(fpr, tpr), 4), "type": "binary"}
        else:
            if not hasattr(model, "predict_proba"):
                return {}
            y_bin   = label_binarize(y, classes=classes)
            y_score = model.predict_proba(X)
            curves: Dict[str, Any] = {}
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                curves[str(cls)] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(auc(fpr, tpr), 4)}
            return {"type": "multiclass", "classes": classes.tolist(), "curves": curves}
    except Exception as e:
        logger.warning(f"ROC computation failed: {e}")
        return {}


def _extract_roc_auc(roc_data: Dict[str, Any]) -> float | None:
    if not roc_data:
        return None
    if roc_data.get("type") == "binary":
        return roc_data.get("auc")
    curves = roc_data.get("curves", {})
    if not curves:
        return None
    aucs = [v["auc"] for v in curves.values() if "auc" in v]
    return round(sum(aucs) / len(aucs), 4) if aucs else None


# ── SHAP / feature importance ─────────────────────────────────────────────────
def _compute_shap(model, X, y, feature_names: List[str], model_name: str) -> Dict[str, Any]:
    try:
        import shap

        tree_models = {"random_forest", "gradient_boosting"}
        linear_models = {"logistic_regression", "ridge", "lasso", "linear_regression"}

        if model_name in tree_models:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                mean_abs = np.abs(np.array(shap_vals)).mean(axis=(0, 1)).tolist()
            else:
                mean_abs = np.abs(shap_vals).mean(axis=0).tolist()
        elif model_name in linear_models:
            explainer = shap.LinearExplainer(model, X)
            shap_vals = explainer.shap_values(X)
            mean_abs = np.abs(shap_vals).mean(axis=0).tolist()
        else:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            mean_abs = r.importances_mean.tolist()

        importance = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
        return {
            "feature_names": [i[0] for i in importance],
            "mean_abs_shap":  [round(i[1], 6) for i in importance],
        }
    except Exception as e:
        logger.warning(f"SHAP failed for {model_name}: {e}")
        return {}


# ── Rule-based justification ─────────────────────────────────────────────────
def _build_justification(best: Dict[str, Any], all_results: List[Dict[str, Any]], task_type: str) -> str:
    m = best["metrics"]
    name = best["model_name"].replace("_", " ").title()
    is_clf = task_type == "classification"

    cv    = m.get("accuracy" if is_clf else "r2_score", 0)
    test  = m.get("test_accuracy" if is_clf else "test_r2", 0)
    std   = m.get("cv_std", 0)
    gap   = m.get("overfit_gap", 0)
    score = best["score"]

    # Rank position
    rank = next((i + 1 for i, r in enumerate(all_results) if r["model_name"] == best["model_name"]), 1)
    total = len(all_results)

    # Underfitting check
    underfit = cv < 0.6 and test < 0.6
    overfit  = gap > 0.15
    mild_of  = 0.05 < gap <= 0.15

    parts = [
        f"{name} ranked 1st out of {total} models with a composite score of {score:.4f} "
        f"(50% CV + 30% test + 20% stability).",
        f"CV score: {cv:.4f} | Test score: {test:.4f} | CV std dev: {std:.4f} | Overfit gap: {gap:.4f}.",
    ]

    if std <= 0.03:
        parts.append("Highly stable across folds (std dev <= 0.03) — reliable on unseen data.")
    elif std <= 0.06:
        parts.append("Moderate stability across folds.")
    else:
        parts.append("High variance across folds — results may vary with different data splits.")

    if overfit:
        parts.append(f"Warning: severe overfitting detected (gap {gap:.2%}). Model memorised training data.")
    elif mild_of:
        parts.append(f"Mild overfitting (gap {gap:.2%}) — acceptable for most use cases.")
    elif gap <= 0.05:
        parts.append("Excellent generalisation — train and test scores are closely aligned.")

    if underfit:
        parts.append("Note: both CV and test scores are below 0.60 — the model may be underfitting. Consider more features or a more complex model.")

    params = best.get("params", {})
    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        parts.append(f"Optimal hyperparameters found via RandomizedSearchCV (5-fold): {param_str}.")

    return " ".join(parts)


# ── Main parallel evaluation agent ────────────────────────────────────────────
def parallel_evaluation_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    logger.info("=== Parallel Evaluation Agent Started ===")

    task_type     = state["task_type"]
    df            = state["processed_data"].to_pandas()
    target_col    = state["target_column"]
    feature_names = state["selected_features"]
    cv_folds      = config["automl"].get("cv_folds", 3)

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    # ── Train all models in parallel ──────────────────────────────────────────
    all_results = train_all_models_parallel(
        X, y, task_type,
        cv_folds=cv_folds,
        max_workers=len(state.get("_model_list", []) or [5]),  # up to 5 workers
    )

    # ── Pick best (first after sort-by-score) ─────────────────────────────────
    valid = [r for r in all_results if r["score"] > -999.0 and r["model"] is not None]
    if not valid:
        logger.error("All models failed — no valid results")
        state["agent_logs"].append("Evaluation: all models failed")
        return state

    best = valid[0]
    best_model_name = best["model_name"]
    best_score      = best["score"]

    logger.info(f"Best model: {best_model_name} (score={best_score:.4f})")

    # ── ROC + SHAP only for best model ────────────────────────────────────────
    X_test = best["X_test"]
    y_test = best["y_test"]

    roc_data  = _compute_roc(best["model"], X_test, y_test, task_type)
    shap_data = _compute_shap(best["model"], X_test, y_test, feature_names, best_model_name)

    # Attach roc_auc to best model metrics
    best["metrics"]["roc_auc"] = _extract_roc_auc(roc_data)

    # ── Underfitting detection ────────────────────────────────────────────────
    best_cv   = best["metrics"].get("accuracy" if task_type == "classification" else "r2_score", 0)
    best_test = best["metrics"].get("test_accuracy" if task_type == "classification" else "test_r2", 0)
    is_underfit = best_cv < 0.6 and best_test < 0.6

    # ── Rule-based justification ──────────────────────────────────────────────
    justification = _build_justification(best, valid, task_type)

    # ── Build evaluation_results (all models, for comparison chart) ───────────
    # Include only aggregated metrics — exclude X_test, y_test, predictions, raw data
    evaluation_results = []
    for r in all_results:
        evaluation_results.append({
            "model_name": r["model_name"],
            "params":     r.get("params", {}),
            "metrics":    {
                # Include only the aggregated metrics, not raw predictions
                k: v for k, v in r.get("metrics", {}).items()
                if k not in ("predictions", "y_pred", "y_test", "X_test", "raw_predictions")
            },
            "score":      r["score"],
            "iteration":  1,
        })

    # ── Update state ──────────────────────────────────────────────────────────
    state["evaluation_results"] = evaluation_results
    state["models_tried"]       = evaluation_results
    state["best_model"]         = best_model_name
    state["best_score"]         = best_score
    state["best_params"]        = best.get("params", {})
    state["roc_data"]           = roc_data if roc_data else None
    state["shap_values"]        = shap_data if shap_data else None
    state["current_model"]      = "__done__"
    state["iteration"]          = len(all_results) + 1
    state["justification"]      = justification
    state["is_underfit"]        = is_underfit
    # Store fitted model object transiently for artifacts_agent (not serialised to JSON)
    state["_fitted_model"]      = best["model"]

    # Summary log
    summary = " | ".join(f"{r['model_name']}={r['score']:.4f}" for r in all_results)
    state["agent_logs"].append(f"Parallel Evaluation: {len(all_results)} models — {summary}")
    state["agent_logs"].append(f"Best: {best_model_name} ({best_score:.4f})")

    logger.info("=== Parallel Evaluation Agent Completed ===")
    return state
