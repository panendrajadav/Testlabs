import argparse
import polars as pl
import yaml
from typing import Dict, Any
from graph.state import AutoMLState
from graph.pipeline import run_pipeline
from utils.logger import logger
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def initialize_state(df: pl.DataFrame, target_column: str, config: Dict[str, Any]) -> AutoMLState:
    """Initialize AutoML state."""
    return {
        "raw_data": df,
        "processed_data": None,
        "target_column": target_column,
        "task_type": config["automl"]["task_type"],
        "dataset_metadata": {},
        "eda_summary": {},
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
        "iteration": 1,
        "max_iterations": config["automl"]["max_iterations"],
        "score_threshold": config["automl"]["score_threshold"],
        "agent_logs": [],
        "final_report": None,
    }

def print_final_report(state: AutoMLState):
    """Print structured final report."""
    print("\n" + "=" * 80)
    print("AUTOML PIPELINE - FINAL REPORT")
    print("=" * 80)
    
    print(f"\n📊 Dataset Information:")
    print(f"   - Samples: {state['raw_data'].height}")
    print(f"   - Features: {len(state['selected_features'])}")
    print(f"   - Task Type: {state['task_type']}")
    print(f"   - Target Column: {state['target_column']}")
    
    print(f"\n🏆 Best Model:")
    print(f"   - Model: {state['best_model']}")
    print(f"   - Score: {state['best_score']:.4f}")
    print(f"   - Hyperparameters: {state['best_params']}")
    
    print(f"\n📈 All Models Tried:")
    for i, result in enumerate(state['evaluation_results'], 1):
        print(f"   {i}. {result['model_name']}: {result['score']:.4f}")
    
    print(f"\n🔄 Pipeline Statistics:")
    print(f"   - Total Iterations: {state['iteration'] - 1}")
    print(f"   - Models Evaluated: {len(state['models_tried'])}")
    
    print(f"\n📝 Agent Decision Log:")
    for log in state['agent_logs']:
        print(f"   - {log}")
    
    print("\n" + "=" * 80)
    print("Pipeline execution completed successfully!")
    print("=" * 80 + "\n")

def main():
    """Main entry point for AutoML system."""
    parser = argparse.ArgumentParser(
        description="TestLabs AutoML - Automated Machine Learning Pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV dataset file"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        default=None,
        help="Name of target column (optional - will auto-detect if not provided)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Load dataset
        logger.info(f"Loading dataset from {args.data}")
        df = pl.read_csv(args.data)
        logger.info(f"Dataset loaded: {df.height} rows, {df.width} columns")
        
        # Sample large datasets for faster processing
        if df.height > 1000:
            logger.info(f"Large dataset detected ({df.height} rows). Sampling 1000 rows for faster processing...")
            df = df.sample(n=min(1000, df.height), seed=42)
            logger.info(f"Using {df.height} sampled rows")
        
        # Auto-detect target column if not provided
        if args.target is None:
            logger.info("Auto-detecting target column...")
            from langchain_openai import AzureChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            import os
            
            llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-05-01-preview",
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "grok-4-1-fast-reasoning"),
                temperature=0.3
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert data scientist."),
                ("user", """Given these column names from a dataset, identify which column is most likely the TARGET variable to predict.

Columns: {columns}

Respond with ONLY the column name, nothing else.""")
            ])
            
            response = llm.invoke(prompt.format_messages(columns=list(df.columns)))
            target_column = response.content.strip()
            
            # Validate the detected column
            if target_column not in df.columns:
                logger.error(f"Auto-detected target '{target_column}' not found in columns")
                logger.error(f"Available columns: {list(df.columns)}")
                sys.exit(1)
            
            logger.info(f"Auto-detected target column: {target_column}")
            args.target = target_column
        
        # Validate target column
        if args.target not in df.columns:
            logger.error(f"Target column '{args.target}' not found in dataset")
            logger.error(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        
        # Initialize state
        initial_state = initialize_state(df, args.target, config)
        
        # Run pipeline
        final_state = run_pipeline(initial_state, config)
        
        # Print final report
        print_final_report(final_state)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
