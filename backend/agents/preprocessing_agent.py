from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AutoMLState
from utils.logger import logger
from utils.helpers import llm_invoke_with_retry
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
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

def _impute_numeric(X, numeric_cols):
    if numeric_cols and X[numeric_cols].isnull().any().any():
        imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
    return X

def _impute_and_encode_categorical(X, categorical_cols):
    if not categorical_cols:
        return X
    if X[categorical_cols].isnull().any().any():
        imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = imputer.fit_transform(X[categorical_cols])
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X

def preprocessing_agent(state: AutoMLState, config: Dict[str, Any]) -> AutoMLState:
    logger.info("=== Preprocessing Agent Started ===")

    df = state["raw_data"].clone()
    target_col = state["target_column"]
    eda = state["eda_summary"]
    numeric_cols = eda["numeric_cols"]
    categorical_cols = eda["categorical_cols"]

    # Skip LLM for small datasets
    if df.height < 500:
        llm_decision = "median imputation, scaling yes, label encoding"
        logger.info("Small dataset - using default preprocessing strategy")
    else:
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
- handle_imbalance: yes/no
- encoding: label/onehot

Keep it brief, just the decisions.""")
        ])
        response = llm_invoke_with_retry(llm, prompt.format_messages(
            missing_values=eda.get("missing_values", {}),
            outliers=eda.get("outliers", {}),
            task_type=eda["task_type"],
            class_balance=eda.get("class_balance", {})
        ))
        llm_decision = response.content
    logger.info(f"Preprocessing Decision: {llm_decision}")

    # Convert to pandas for sklearn
    df_pd = df.to_pandas()
    X = df_pd.drop(columns=[target_col])
    y = df_pd[target_col]

    # Run numeric imputation and categorical imputation+encoding in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        num_future = executor.submit(_impute_numeric, X.copy(), numeric_cols)
        cat_future = executor.submit(_impute_and_encode_categorical, X.copy(), categorical_cols)
        X_num = num_future.result()
        X_cat = cat_future.result()

    # Merge results back
    if numeric_cols:
        X[numeric_cols] = X_num[numeric_cols]
    if categorical_cols:
        X[categorical_cols] = X_cat[categorical_cols]

    # Encode target if classification
    if state["task_type"] == "classification" and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        logger.info("Encoded target variable")

    # Scale numeric features
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        logger.info("Applied standard scaling to numeric features")

    # Handle class imbalance
    if state["task_type"] == "classification" and eda.get("class_balance"):
        class_counts = list(eda["class_balance"].values())
        if len(class_counts) > 1 and max(class_counts) / min(class_counts) > 3:
            try:
                smote = SMOTE(random_state=42, n_jobs=-1)
                X, y = smote.fit_resample(X, y)
                logger.info(f"Applied SMOTE: {len(X)} samples after resampling")
            except:
                logger.warning("SMOTE failed, skipping class balancing")

    X[target_col] = y
    processed_df = pl.from_pandas(X)

    state["processed_data"] = processed_df
    state["preprocessing_config"] = {"llm_decision": llm_decision}
    state["agent_logs"].append(f"Preprocessing Agent: Processed {processed_df.height} samples")

    logger.info("=== Preprocessing Agent Completed ===")
    return state
