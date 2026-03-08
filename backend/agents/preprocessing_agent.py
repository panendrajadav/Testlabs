from typing import Dict, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AutoMLState
from utils.logger import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
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

def preprocessing_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    """Preprocess dataset based on EDA insights."""
    logger.info("=== Preprocessing Agent Started ===")
    
    df = state["raw_data"].clone()
    target_col = state["target_column"]
    eda = state["eda_summary"]
    
    # LLM decides preprocessing strategy
    llm = create_llm(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert ML engineer deciding preprocessing strategy."),
        ("user", """Based on this EDA summary, decide preprocessing steps:

Missing values: {missing_values}
Outliers: {outliers}
Task type: {task_type}
Class balance: {class_balance}

Respond with a JSON-like decision:
- imputation_strategy: mean/median/mode
- scaling: yes/no
- handle_imbalance: yes/no (only if severe class imbalance)
- encoding: label/onehot

Keep it brief, just the decisions.""")
    ])
    
    response = llm.invoke(prompt.format_messages(
        missing_values=eda.get("missing_values", {}),
        outliers=eda.get("outliers", {}),
        task_type=eda["task_type"],
        class_balance=eda.get("class_balance", {})
    ))
    
    llm_decision = response.content
    logger.info(f"LLM Preprocessing Decision: {llm_decision}")
    
    # Convert to pandas for sklearn compatibility
    df_pd = df.to_pandas()
    X = df_pd.drop(columns=[target_col])
    y = df_pd[target_col]
    
    # Handle missing values
    numeric_cols = eda["numeric_cols"]
    categorical_cols = eda["categorical_cols"]
    
    if numeric_cols and X[numeric_cols].isnull().any().any():
        imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        logger.info("Applied median imputation to numeric columns")
    
    if categorical_cols and X[categorical_cols].isnull().any().any():
        imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = imputer.fit_transform(X[categorical_cols])
        logger.info("Applied mode imputation to categorical columns")
    
    # Encode categorical variables
    if categorical_cols:
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        logger.info(f"Label encoded {len(categorical_cols)} categorical columns")
    
    # Encode target if classification
    if state["task_type"] == "classification" and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        logger.info("Encoded target variable")
    
    # Scale features
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        logger.info("Applied standard scaling to numeric features")
    
    # Handle class imbalance
    if state["task_type"] == "classification" and eda.get("class_balance"):
        class_counts = list(eda["class_balance"].values())
        if len(class_counts) > 1 and max(class_counts) / min(class_counts) > 3:
            try:
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)
                logger.info(f"Applied SMOTE: {len(X)} samples after resampling")
            except:
                logger.warning("SMOTE failed, skipping class balancing")
    
    # Combine back and convert to polars
    X[target_col] = y
    processed_df = pl.from_pandas(X)
    
    state["processed_data"] = processed_df
    state["preprocessing_config"] = {"llm_decision": llm_decision}
    state["agent_logs"].append(f"Preprocessing Agent: Processed {processed_df.height} samples")
    
    logger.info("=== Preprocessing Agent Completed ===")
    return state
