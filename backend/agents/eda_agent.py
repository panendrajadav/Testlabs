from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AutoMLState
from utils.logger import logger
from utils.helpers import detect_task_type, get_column_types, calculate_missing_percentage, detect_outliers_iqr, llm_invoke_with_retry
import polars as pl
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

def _build_plots(df: pl.DataFrame, numeric_cols: list, categorical_cols: list, target_col: str, task_type: str) -> Dict[str, Any]:
    """Build Plotly-compatible chart data."""
    plots = {}

    # Distribution histograms for numeric cols
    for col in numeric_cols[:8]:
        values = df[col].drop_nulls().to_list()
        plots[f"dist_{col}"] = {
            "type": "histogram",
            "title": f"Distribution of {col}",
            "x": values,
            "xaxis": col,
            "yaxis": "Count"
        }

    # Bar charts for categorical cols
    for col in categorical_cols[:5]:
        vc = df[col].value_counts().sort("count", descending=True).head(10)
        plots[f"bar_{col}"] = {
            "type": "bar",
            "title": f"Value Counts: {col}",
            "x": vc[col].to_list(),
            "y": vc["count"].to_list(),
            "xaxis": col,
            "yaxis": "Count"
        }

    # Correlation heatmap data
    if len(numeric_cols) > 1:
        df_num = df.select(numeric_cols).to_pandas()
        corr = df_num.corr().round(3)
        plots["correlation_heatmap"] = {
            "type": "heatmap",
            "title": "Feature Correlation Matrix",
            "z": corr.values.tolist(),
            "x": corr.columns.tolist(),
            "y": corr.columns.tolist()
        }

    # Target distribution
    if task_type == "classification":
        vc = df[target_col].value_counts()
        plots["target_distribution"] = {
            "type": "pie",
            "title": f"Target Distribution: {target_col}",
            "labels": vc[target_col].cast(pl.Utf8).to_list(),
            "values": vc["count"].to_list()
        }
    else:
        values = df[target_col].drop_nulls().to_list()
        plots["target_distribution"] = {
            "type": "histogram",
            "title": f"Target Distribution: {target_col}",
            "x": values,
            "xaxis": target_col,
            "yaxis": "Count"
        }

    return plots

def eda_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    logger.info("=== EDA Agent Started ===")

    df = state["raw_data"]
    target_col = state["target_column"]

    n_rows, n_cols = df.height, df.width
    column_types = get_column_types(df.drop(target_col))
    missing_pct = calculate_missing_percentage(df)

    y = df[target_col]
    task_type = detect_task_type(y) if state["task_type"] == "auto" else state["task_type"]

    numeric_cols = column_types["numeric"]
    categorical_cols = column_types["categorical"]

    # Run outlier detection and plot building in parallel
    def _detect_outliers():
        return {col: detect_outliers_iqr(df[col]) for col in numeric_cols}

    def _build_plots_task():
        return _build_plots(df, numeric_cols, categorical_cols, target_col, task_type)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outlier_future = executor.submit(_detect_outliers)
        plots_future = executor.submit(_build_plots_task)
        outliers = outlier_future.result()
        plots = plots_future.result()

    class_balance = None
    if task_type == "classification":
        vc = df[target_col].value_counts()
        class_balance = {str(row[0]): int(row[1]) for row in vc.iter_rows()}

    eda_summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_values": {k: v for k, v in missing_pct.items() if v > 0},
        "outliers": {k: v for k, v in outliers.items() if v > 0},
        "class_balance": class_balance,
        "task_type": task_type,
    }

    # Skip LLM insights for small datasets
    if df.height >= 500:
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
        response = llm_invoke_with_retry(llm, prompt.format_messages(**eda_summary))
        eda_summary["llm_insights"] = response.content
    else:
        eda_summary["llm_insights"] = f"Small dataset ({df.height} rows). Task: {task_type}. Features: {numeric_cols + categorical_cols}."

    logger.info(f"Task Type: {task_type}, Shape: {n_rows}x{n_cols}, Plots generated: {len(plots)}")

    state["eda_summary"] = eda_summary
    state["eda_plots"] = plots
    state["task_type"] = task_type
    state["agent_logs"].append(f"EDA Agent: Detected {task_type} task with {n_rows} samples, {len(plots)} plots")

    logger.info("=== EDA Agent Completed ===")
    return state
