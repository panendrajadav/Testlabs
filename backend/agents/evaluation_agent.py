from typing import Dict, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AutoMLState
from utils.logger import logger
from tools.sklearn_tools import train_and_evaluate_sklearn
from tools.xgboost_tools import train_and_evaluate_xgboost, train_and_evaluate_lightgbm
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

def evaluation_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    """Train and evaluate model with best hyperparameters."""
    logger.info("=== Evaluation Agent Started ===")
    
    model_name = state["current_model"]
    params = state["current_params"]
    task_type = state["task_type"]
    df = state["processed_data"].to_pandas()
    target_col = state["target_column"]
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    cv_folds = config["automl"]["cv_folds"]
    
    # Train and evaluate
    try:
        if model_name == "xgboost":
            result = train_and_evaluate_xgboost(X, y, task_type, params, cv_folds)
        elif model_name == "lightgbm":
            result = train_and_evaluate_lightgbm(X, y, task_type, params, cv_folds)
        else:
            result = train_and_evaluate_sklearn(X, y, model_name, task_type, params, cv_folds)
        
        metrics = result["metrics"]
        
        # Determine primary score
        if task_type == "classification":
            primary_score = metrics["accuracy"]
        else:
            primary_score = metrics["r2_score"]
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Metrics: {metrics}")
        logger.info(f"Primary Score: {primary_score:.4f}")
        
        # Record results
        eval_result = {
            "model_name": model_name,
            "params": params,
            "metrics": metrics,
            "score": primary_score,
            "iteration": state["iteration"]
        }
        
        state["evaluation_results"].append(eval_result)
        state["models_tried"].append(eval_result)
        
        # Update best model if improved
        if primary_score > state["best_score"]:
            state["best_score"] = primary_score
            state["best_model"] = model_name
            state["best_params"] = params
            logger.info(f"New best model: {model_name} with score {primary_score:.4f}")
        
        # LLM analyzes results
        llm = create_llm(config)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert ML engineer analyzing model performance."),
            ("user", """Analyze this model's performance:

Model: {model_name}
Score: {score:.4f}
Metrics: {metrics}
Best score so far: {best_score:.4f}
Iteration: {iteration}/{max_iterations}

Should we continue trying more models or is this good enough? (1-2 sentences)""")
        ])
        
        response = llm.invoke(prompt.format_messages(
            model_name=model_name,
            score=primary_score,
            metrics=metrics,
            best_score=state["best_score"],
            iteration=state["iteration"],
            max_iterations=state["max_iterations"]
        ))
        
        llm_analysis = response.content
        logger.info(f"LLM Analysis: {llm_analysis}")
        
        state["agent_logs"].append(
            f"Evaluation Agent: {model_name} scored {primary_score:.4f}"
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        state["agent_logs"].append(f"Evaluation Agent: {model_name} failed - {str(e)}")
    
    # Increment iteration
    state["iteration"] += 1
    
    logger.info("=== Evaluation Agent Completed ===")
    return state
