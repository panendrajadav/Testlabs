from typing import Dict, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AutoMLState
from utils.logger import logger
from utils.helpers import llm_invoke_with_retry
from tools.sklearn_tools import train_and_evaluate_sklearn, get_sklearn_model
from tools.xgboost_tools import train_and_evaluate_xgboost, train_and_evaluate_lightgbm
import numpy as np
import os

def create_llm(config: Dict[str, Any]):
    if config["llm"]["provider"] == "azure":
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-05-01-preview",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "grok-4-1-fast-reasoning"),
            temperature=config["llm"]["temperature"]
        )
    elif config["llm"]["provider"] == "openai":
        return ChatOpenAI(model=config["llm"]["model"], temperature=config["llm"]["temperature"])
    else:
        return ChatAnthropic(model=config["llm"]["model"], temperature=config["llm"]["temperature"])

def _compute_roc(model, X, y, task_type: str) -> Dict[str, Any]:
    """Compute ROC curve data for classification."""
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        import numpy as np

        classes = np.unique(y)
        if task_type != "classification":
            return {}

        if len(classes) == 2:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X)
            else:
                return {}
            fpr, tpr, _ = roc_curve(y, y_score)
            roc_auc = auc(fpr, tpr)
            return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(roc_auc, 4), "type": "binary"}
        else:
            y_bin = label_binarize(y, classes=classes)
            if not hasattr(model, "predict_proba"):
                return {}
            y_score = model.predict_proba(X)
            roc_data = {"type": "multiclass", "classes": classes.tolist(), "curves": {}}
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_data["curves"][str(cls)] = {
                    "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(auc(fpr, tpr), 4)
                }
            return roc_data
    except Exception as e:
        logger.warning(f"ROC computation failed: {e}")
        return {}

def _compute_shap(model, X, y, feature_names: list, model_name: str) -> Dict[str, Any]:
    """Compute SHAP values for feature importance on held-out data."""
    try:
        import shap
        if model_name in ["xgboost", "lightgbm", "random_forest", "decision_tree", "extra_trees", "gradient_boosting"]:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            mean_abs = np.abs(shap_vals).mean(axis=0).tolist()
        elif model_name in ["logistic_regression", "ridge", "lasso"]:
            explainer = shap.LinearExplainer(model, X)
            shap_vals = explainer.shap_values(X)
            mean_abs = np.abs(shap_vals).mean(axis=0).tolist()
        else:
            # SVM, KNN — use permutation importance as fallback
            from sklearn.inspection import permutation_importance
            r = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            mean_abs = r.importances_mean.tolist()

        importance = sorted(
            zip(feature_names, mean_abs),
            key=lambda x: x[1], reverse=True
        )
        return {
            "feature_names": [i[0] for i in importance],
            "mean_abs_shap": [round(i[1], 6) for i in importance]
        }
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {}

def evaluation_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    logger.info("=== Evaluation Agent Started ===")

    model_name = state["current_model"]

    # Sentinel: model selection exhausted all models
    if model_name == "__done__":
        logger.info("All models evaluated — skipping evaluation node")
        state["iteration"] += 1
        return state
    params = state["current_params"]
    task_type = state["task_type"]
    df = state["processed_data"].to_pandas()
    target_col = state["target_column"]
    feature_names = state["selected_features"]

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    cv_folds = config["automl"]["cv_folds"]

    try:
        if model_name == "xgboost":
            result = train_and_evaluate_xgboost(X, y, task_type, params, cv_folds)
        elif model_name == "lightgbm":
            result = train_and_evaluate_lightgbm(X, y, task_type, params, cv_folds)
        else:
            result = train_and_evaluate_sklearn(X, y, model_name, task_type, params, cv_folds)

        metrics = result["metrics"]
        trained_model = result["model"]
        X_test = result.get("X_test", X)
        y_test = result.get("y_test", y)

        # Add F1 if not present — use held-out test data
        if task_type == "classification" and "f1_score" not in metrics:
            from sklearn.metrics import f1_score
            y_pred = trained_model.predict(X_test)
            metrics["f1_score"] = round(float(f1_score(y_test, y_pred, average="weighted")), 4)

        primary_score = metrics.get("accuracy", metrics.get("r2_score", 0.0))

        # Penalise overfitting: if train score is much higher than CV score,
        # reduce the effective score so an overfit model never wins
        overfit_gap = metrics.get("overfit_gap", 0.0)
        OVERFIT_THRESHOLD = 0.05   # >5% gap triggers penalty
        if overfit_gap > OVERFIT_THRESHOLD:
            penalty = (overfit_gap - OVERFIT_THRESHOLD) * 0.5
            adjusted_score = max(0.0, primary_score - penalty)
            logger.warning(
                f"{model_name}: overfit gap={overfit_gap:.4f} → penalised score "
                f"{primary_score:.4f} → {adjusted_score:.4f}"
            )
            metrics["overfit_penalty"] = round(penalty, 4)
            primary_score = adjusted_score
        else:
            metrics["overfit_penalty"] = 0.0

        # Compute ROC and SHAP on held-out test data only
        roc_data = _compute_roc(trained_model, X_test, y_test, task_type)
        if roc_data:
            metrics["roc_auc"] = roc_data.get("auc", roc_data.get("curves", {}).get(str(list(roc_data.get("curves", {}).keys())[0]), {}).get("auc")) if roc_data else None

        shap_data = _compute_shap(trained_model, X_test, y_test, feature_names, model_name)

        logger.info(f"Model: {model_name}, Score: {primary_score:.4f}, Metrics: {metrics}")

        eval_result = {
            "model_name": model_name,
            "params": params,
            "metrics": metrics,
            "score": primary_score,
            "iteration": state["iteration"]
        }

        state["evaluation_results"].append(eval_result)
        state["models_tried"].append(eval_result)

        if primary_score > state["best_score"]:
            state["best_score"] = primary_score
            state["best_model"] = model_name
            state["best_params"] = params
            state["roc_data"] = roc_data
            state["shap_values"] = shap_data
            logger.info(f"New best model: {model_name} with score {primary_score:.4f}")

        # LLM analysis - skip for small datasets
        if df.shape[0] >= 500:
            llm = create_llm(config)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert ML engineer analyzing model performance."),
                ("user", "Model: {model_name}\nScore: {score:.4f}\nMetrics: {metrics}\nBest so far: {best_score:.4f}\nIteration: {iteration}/{max_iterations}\n\nShould we continue? (1 sentence)")
            ])
            response = llm_invoke_with_retry(llm, prompt.format_messages(
                model_name=model_name, score=primary_score, metrics=metrics,
                best_score=state["best_score"], iteration=state["iteration"],
                max_iterations=state["max_iterations"]
            ))
            logger.info(f"LLM Analysis: {response.content}")

        state["agent_logs"].append(f"Evaluation Agent: {model_name} scored {primary_score:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        state["agent_logs"].append(f"Evaluation Agent: {model_name} failed - {str(e)}")

    state["iteration"] += 1
    logger.info("=== Evaluation Agent Completed ===")
    return state
