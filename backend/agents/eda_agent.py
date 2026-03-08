from typing import Dict, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AutoMLState
from utils.logger import logger
from utils.helpers import detect_task_type, get_column_types, calculate_missing_percentage, detect_outliers_iqr
import polars as pl
import numpy as np
import os

def create_llm(config: Dict[str, Any]):
    """Create LLM instance based on config."""
    if config["llm"]["provider"] == "azure":
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-05-01-preview",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            temperature=config["llm"]["temperature"]
        )
    elif config["llm"]["provider"] == "openai":
        return ChatOpenAI(model=config["llm"]["model"], temperature=config["llm"]["temperature"])
    else:
        return ChatAnthropic(model=config["llm"]["model"], temperature=config["llm"]["temperature"])

def eda_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    """Analyze dataset and generate EDA summary."""
    logger.info("=== EDA Agent Started ===")
    
    df = state["raw_data"]
    target_col = state["target_column"]
    
    # Basic statistics
    n_rows, n_cols = df.height, df.width
    column_types = get_column_types(df.drop(target_col))
    missing_pct = calculate_missing_percentage(df)
    
    # Target analysis
    y = df[target_col]
    if state["task_type"] == "auto":
        task_type = detect_task_type(y)
    else:
        task_type = state["task_type"]
    
    # Outlier detection
    outliers = {}
    for col in column_types["numeric"]:
        outliers[col] = detect_outliers_iqr(df[col])
    
    # Class balance (for classification)
    class_balance = None
    if task_type == "classification":
        value_counts = df[target_col].value_counts()
        class_balance = {str(row[0]): int(row[1]) for row in value_counts.iter_rows()}
    
    eda_summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_cols": column_types["numeric"],
        "categorical_cols": column_types["categorical"],
        "missing_values": {k: v for k, v in missing_pct.items() if v > 0},
        "outliers": {k: v for k, v in outliers.items() if v > 0},
        "class_balance": class_balance,
        "task_type": task_type,
    }
    
    # LLM reasoning
    llm = create_llm(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert data scientist analyzing a dataset."),
        ("user", """Analyze this EDA summary and provide key insights:
        
Dataset: {n_rows} rows, {n_cols} columns
Task: {task_type}
Numeric columns: {numeric_cols}
Categorical columns: {categorical_cols}
Missing values: {missing_values}
Outliers detected: {outliers}
Class balance: {class_balance}

Provide 3-5 key insights about data quality and potential challenges.""")
    ])
    
    response = llm.invoke(prompt.format_messages(**eda_summary))
    insights = response.content
    
    eda_summary["llm_insights"] = insights
    
    logger.info(f"Task Type Detected: {task_type}")
    logger.info(f"Dataset Shape: {n_rows} rows, {n_cols} columns")
    logger.info(f"LLM Insights: {insights[:200]}...")
    
    state["eda_summary"] = eda_summary
    state["task_type"] = task_type
    state["agent_logs"].append(f"EDA Agent: Detected {task_type} task with {n_rows} samples")
    
    logger.info("=== EDA Agent Completed ===")
    return state
