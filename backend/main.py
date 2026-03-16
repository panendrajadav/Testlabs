import argparse
import polars as pl
import yaml
import time
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from graph.state import AutoMLState
from graph.pipeline import run_pipeline
from utils.logger import logger
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def initialize_state(df: pl.DataFrame, target_column: str, config: Dict[str, Any]) -> AutoMLState:
    return {
        "raw_data": df,
        "processed_data": None,
        "target_column": target_column,
        "task_type": config["automl"]["task_type"],
        "dataset_metadata": {},
        "eda_summary": {},
        "eda_plots": {},
        "preprocessing_config": {},
        "feature_config": {},
        "selected_features": [],
        "models_tried": [],
        "current_model": "",
        "current_params": {},
        "evaluation_results": [],
        "best_model": None,
        "best_score": -float('inf'),
        "best_params": None,
        "roc_data": None,
        "shap_values": None,
        "iteration": 1,
        "max_iterations": config["automl"]["max_iterations"],
        "score_threshold": config["automl"]["score_threshold"],
        "agent_logs": [],
        "final_report": None,
    }

def detect_target(df: pl.DataFrame) -> str:
    """Detect target column - use last column for small datasets, LLM for large."""
    if df.height < 500:
        target = df.columns[-1]
        logger.info(f"Small dataset - using last column as target: '{target}'")
        return target

    from langchain_openai import AzureChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-05-01-preview",
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "grok-4-1-fast-reasoning"),
        temperature=0.3
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert data scientist."),
        ("user", "Given these column names, identify the TARGET variable.\n\nColumns: {columns}\n\nRespond with ONLY the column name.")
    ])

    for attempt in range(1, 4):
        try:
            response = llm.invoke(prompt.format_messages(columns=list(df.columns)))
            target = response.content.strip()
            if target in df.columns:
                return target
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt}/3): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    target = df.columns[-1]
    logger.warning(f"Auto-detection failed. Falling back to last column: '{target}'")
    return target

def print_final_report(state: AutoMLState):
    print("\n" + "=" * 80)
    print("AUTOML PIPELINE - FINAL REPORT")
    print("=" * 80)
    print(f"\nDataset Information:")
    print(f"   - Samples: {state['raw_data'].height}")
    print(f"   - Features: {len(state['selected_features'])}")
    print(f"   - Task Type: {state['task_type']}")
    print(f"   - Target Column: {state['target_column']}")
    print(f"\nBest Model:")
    print(f"   - Model: {state['best_model']}")
    print(f"   - Score: {state['best_score']:.4f}")
    print(f"   - Hyperparameters: {state['best_params']}")
    print(f"\nAll Models Tried:")
    for i, result in enumerate(state['evaluation_results'], 1):
        print(f"   {i}. {result['model_name']}: {result['score']:.4f}")
    print(f"\nPipeline Statistics:")
    print(f"   - Total Iterations: {state['iteration'] - 1}")
    print(f"   - Models Evaluated: {len(state['models_tried'])}")
    print(f"\nAgent Decision Log:")
    for log in state['agent_logs']:
        print(f"   - {log}")
    print("\n" + "=" * 80)
    print("Pipeline execution completed successfully!")
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="TestLabs AutoML - Automated Machine Learning Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset file")
    parser.add_argument("--target", type=str, required=False, default=None, help="Target column name (auto-detected if not provided)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    try:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        logger.info(f"Loading dataset from {args.data}")
        df = pl.read_csv(args.data)
        logger.info(f"Dataset loaded: {df.height} rows, {df.width} columns")

        # Detect target column in parallel with config loading
        with ThreadPoolExecutor(max_workers=2) as executor:
            if args.target is None:
                target_future = executor.submit(detect_target, df)
                args.target = target_future.result()
            
        if args.target not in df.columns:
            logger.error(f"Target column '{args.target}' not found. Available: {list(df.columns)}")
            sys.exit(1)

        logger.info(f"Target column: '{args.target}'")
        initial_state = initialize_state(df, args.target, config)

        # Run pipeline in ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            future = executor.submit(run_pipeline, initial_state, config)
            final_state = future.result()

        print_final_report(final_state)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
