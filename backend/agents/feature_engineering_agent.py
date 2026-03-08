from typing import Dict, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AutoMLState
from utils.logger import logger
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import polars as pl
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

def feature_engineering_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    """Select and engineer features intelligently."""
    logger.info("=== Feature Engineering Agent Started ===")
    
    df = state["processed_data"].clone()
    target_col = state["target_column"]
    task_type = state["task_type"]
    
    # Convert to pandas for sklearn
    df_pd = df.to_pandas()
    X = df_pd.drop(columns=[target_col])
    y = df_pd[target_col]
    
    n_features = X.shape[1]
    
    # LLM decides feature strategy
    llm = create_llm(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert ML engineer deciding feature engineering strategy."),
        ("user", """Dataset has {n_features} features for {task_type} task.

Decide:
- Should we do feature selection? (yes/no)
- If yes, keep top K features where K = ?
- Suggest K as a number between 5 and {n_features}

Keep response brief: just "yes, K=X" or "no, keep all".""")
    ])
    
    response = llm.invoke(prompt.format_messages(
        n_features=n_features,
        task_type=task_type
    ))
    
    llm_decision = response.content
    logger.info(f"LLM Feature Engineering Decision: {llm_decision}")
    
    # Feature selection
    selected_features = X.columns.tolist()
    
    if "yes" in llm_decision.lower() and n_features > 10:
        try:
            k = min(int([s for s in llm_decision.split() if s.isdigit()][0]), n_features - 1)
        except:
            k = min(10, n_features - 1)
        
        score_func = f_classif if task_type == "classification" else f_regression
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        X = X[selected_features]
        logger.info(f"Selected top {k} features: {selected_features}")
    else:
        logger.info("Keeping all features")
    
    # Update dataframe and convert back to polars
    X[target_col] = y.values
    df_engineered = pl.from_pandas(X)
    
    state["processed_data"] = df_engineered
    state["selected_features"] = selected_features
    state["feature_config"] = {"llm_decision": llm_decision, "n_features": len(selected_features)}
    state["agent_logs"].append(f"Feature Engineering Agent: Selected {len(selected_features)} features")
    
    logger.info("=== Feature Engineering Agent Completed ===")
    return state
