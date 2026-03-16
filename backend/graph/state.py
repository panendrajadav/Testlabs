from typing import TypedDict, List, Dict, Any, Optional
import polars as pl

class AutoMLState(TypedDict):
    # Dataset
    raw_data: Optional[pl.DataFrame]
    processed_data: Optional[pl.DataFrame]
    target_column: str

    # Task metadata
    task_type: str
    dataset_metadata: Dict[str, Any]

    # EDA results
    eda_summary: Dict[str, Any]
    eda_plots: Dict[str, Any]  # Plotly chart JSON per column

    # Preprocessing
    preprocessing_config: Dict[str, Any]

    # Feature engineering
    feature_config: Dict[str, Any]
    selected_features: List[str]

    # Model tracking
    models_tried: List[Dict[str, Any]]
    current_model: str
    current_params: Dict[str, Any]

    # Evaluation
    evaluation_results: List[Dict[str, Any]]
    best_model: Optional[str]
    best_score: float
    best_params: Optional[Dict[str, Any]]
    roc_data: Optional[Dict[str, Any]]
    shap_values: Optional[Dict[str, Any]]

    # Iteration control
    iteration: int
    max_iterations: int
    score_threshold: float

    # Logging
    agent_logs: List[str]

    # Final output
    final_report: Optional[Dict[str, Any]]
