from typing import Dict, Any, List
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AutoMLState
from utils.logger import logger
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

def model_selection_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    """Select next model to try based on task and previous results."""
    logger.info("=== Model Selection Agent Started ===")
    
    task_type = state["task_type"]
    n_samples = state["processed_data"].height
    n_features = len(state["selected_features"])
    models_tried = [m["model_name"] for m in state["models_tried"]]
    
    # Available models based on task
    available_models = {
        "classification": ["logistic_regression", "random_forest", "xgboost", "lightgbm", "svm", "decision_tree", "knn"],
        "regression": ["ridge", "lasso", "random_forest", "xgboost", "lightgbm", "svm", "decision_tree", "knn"]
    }
    
    candidate_models = [m for m in available_models[task_type] if m not in models_tried]
    
    if not candidate_models:
        logger.warning("All models tried, selecting best performing model again")
        candidate_models = available_models[task_type][:1]
    
    # LLM selects best model
    llm = create_llm(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert ML engineer selecting the best model."),
        ("user", """Select the best model for this scenario:

Task: {task_type}
Dataset: {n_samples} samples, {n_features} features
Models already tried: {models_tried}
Available models: {candidate_models}

Previous results:
{previous_results}

Select ONE model from available models. Respond with just the model name.""")
    ])
    
    previous_results = "\n".join([
        f"{m['model_name']}: {m.get('score', 'N/A')}" 
        for m in state["models_tried"][-3:]
    ]) if state["models_tried"] else "None yet"
    
    response = llm.invoke(prompt.format_messages(
        task_type=task_type,
        n_samples=n_samples,
        n_features=n_features,
        models_tried=models_tried,
        candidate_models=candidate_models,
        previous_results=previous_results
    ))
    
    selected_model = response.content.strip().lower()
    
    # Validate selection
    if selected_model not in candidate_models:
        selected_model = candidate_models[0]
        logger.warning(f"LLM selected invalid model, defaulting to {selected_model}")
    
    logger.info(f"Selected Model: {selected_model}")
    logger.info(f"LLM Reasoning: {response.content}")
    
    state["current_model"] = selected_model
    state["agent_logs"].append(f"Model Selection Agent: Selected {selected_model}")
    
    logger.info("=== Model Selection Agent Completed ===")
    return state
