from typing import Dict, Any
from graph.state import AutoMLState
from utils.logger import logger

def model_selection_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    """Select next model to try based on task and previous results."""
    logger.info("=== Model Selection Agent Started ===")
    
    task_type = state["task_type"]
    n_samples = state["processed_data"].height
    n_features = len(state["selected_features"])
    models_tried = [m["model_name"] for m in state["models_tried"]]
    
    # Available models based on task — full list, always evaluate all
    available_models = {
        "classification": [
            "logistic_regression", "random_forest", "xgboost", "lightgbm",
            "svm", "decision_tree", "knn", "gradient_boosting", "extra_trees",
            "naive_bayes", "linear_discriminant",
        ],
        "regression": [
            "ridge", "lasso", "elastic_net", "bayesian_ridge", "huber",
            "random_forest", "xgboost", "lightgbm",
            "svm", "decision_tree", "knn", "gradient_boosting", "extra_trees",
        ]
    }

    candidate_models = [m for m in available_models[task_type] if m not in models_tried]

    if not candidate_models:
        logger.info("All models evaluated.")
        # Return a sentinel so evaluation agent knows to skip
        state["current_model"] = "__done__"
        state["agent_logs"].append("Model Selection Agent: All models evaluated")
        logger.info("=== Model Selection Agent Completed ===")
        return state

    # Always round-robin through candidates — no LLM needed, saves 30-60s per iteration
    selected_model = candidate_models[0]
    logger.info(f"Selected Model: {selected_model}")

    state["current_model"] = selected_model
    state["agent_logs"].append(f"Model Selection Agent: Selected {selected_model}")

    logger.info("=== Model Selection Agent Completed ===")
    return state
