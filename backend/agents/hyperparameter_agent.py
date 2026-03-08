from typing import Dict, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AutoMLState
from utils.logger import logger
from tools.optuna_tools import optimize_hyperparameters, get_search_space
from tools.sklearn_tools import train_and_evaluate_sklearn
from tools.xgboost_tools import train_and_evaluate_xgboost, train_and_evaluate_lightgbm
from sklearn.model_selection import cross_val_score
import numpy as np
import os

def create_llm(config: Dict[str, Any]):
    """Create LLM instance based on config."""
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

def hyperparameter_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    """Optimize hyperparameters using Optuna."""
    logger.info("=== Hyperparameter Tuning Agent Started ===")
    
    model_name = state["current_model"]
    task_type = state["task_type"]
    df = state["processed_data"].to_pandas()
    target_col = state["target_column"]
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    n_trials = config["optuna"]["n_trials"]
    
    # LLM suggests optimization strategy
    llm = create_llm(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert ML engineer planning hyperparameter optimization."),
        ("user", """Planning hyperparameter tuning for {model_name} on {task_type} task.

Dataset: {n_samples} samples, {n_features} features

Suggest optimization focus (1-2 sentences):
- Which hyperparameters are most critical?
- Any special considerations?""")
    ])
    
    response = llm.invoke(prompt.format_messages(
        model_name=model_name,
        task_type=task_type,
        n_samples=X.shape[0],
        n_features=X.shape[1]
    ))
    
    llm_suggestion = response.content
    logger.info(f"LLM Optimization Strategy: {llm_suggestion}")
    
    # Define objective function
    def objective(trial):
        try:
            params = get_search_space(model_name, trial)
            
            if model_name == "xgboost":
                result = train_and_evaluate_xgboost(X, y, task_type, params, cv_folds=3)
            elif model_name == "lightgbm":
                result = train_and_evaluate_lightgbm(X, y, task_type, params, cv_folds=3)
            else:
                result = train_and_evaluate_sklearn(X, y, model_name, task_type, params, cv_folds=3)
            
            # Return primary metric
            if task_type == "classification":
                return result["metrics"]["accuracy"]
            else:
                return -result["metrics"]["rmse"]  # Minimize RMSE
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0 if task_type == "classification" else -999999
    
    # Run optimization
    direction = "maximize" if task_type == "classification" else "maximize"
    result = optimize_hyperparameters(objective, n_trials=n_trials, direction=direction)
    
    best_params = result["best_params"]
    logger.info(f"Best Hyperparameters: {best_params}")
    logger.info(f"Best CV Score: {result['best_value']:.4f}")
    
    state["current_params"] = best_params
    state["agent_logs"].append(f"Hyperparameter Agent: Optimized {model_name} with {n_trials} trials")
    
    logger.info("=== Hyperparameter Tuning Agent Completed ===")
    return state
