from typing import Dict, Any
from langgraph.graph import StateGraph, END
from graph.state import AutoMLState
from agents.eda_agent import eda_agent
from agents.preprocessing_agent import preprocessing_agent
from agents.feature_engineering_agent import feature_engineering_agent
from agents.evaluation_agent import parallel_evaluation_agent
from utils.logger import logger


def _create_setup_pipeline(config: Dict[str, Any], progress_hook=None) -> StateGraph:
    """EDA → Preprocessing → Feature Engineering — runs exactly once."""

    def _hook(name: str):
        if progress_hook:
            progress_hook(name)

    def eda_node(state: AutoMLState) -> AutoMLState:
        _hook("EDA")
        return eda_agent(state, config)

    def preprocessing_node(state: AutoMLState) -> AutoMLState:
        _hook("Preprocessing")
        return preprocessing_agent(state, config)

    def feature_engineering_node(state: AutoMLState) -> AutoMLState:
        _hook("Feature Engineering")
        return feature_engineering_agent(state, config)

    workflow = StateGraph(AutoMLState)
    workflow.add_node("eda", eda_node)
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("feature_engineering", feature_engineering_node)

    workflow.set_entry_point("eda")
    workflow.add_edge("eda", "preprocessing")
    workflow.add_edge("preprocessing", "feature_engineering")
    workflow.add_edge("feature_engineering", END)

    return workflow.compile()


def _create_training_pipeline(config: Dict[str, Any], progress_hook=None) -> StateGraph:
    """Single parallel training node — all models run concurrently, best selected immediately."""

    def _hook(name: str):
        if progress_hook:
            progress_hook(name)

    def training_node(state: AutoMLState) -> AutoMLState:
        _hook("Training Models")
        return parallel_evaluation_agent(state, config)

    workflow = StateGraph(AutoMLState)
    workflow.add_node("training", training_node)
    workflow.set_entry_point("training")
    workflow.add_edge("training", END)

    return workflow.compile()


def run_pipeline(initial_state: AutoMLState, config: Dict[str, Any], progress_hook=None) -> AutoMLState:
    logger.info("=" * 60)
    logger.info("Starting AutoML Pipeline (Fast Parallel Mode)")
    logger.info("=" * 60)

    # Phase 1: Setup — EDA, preprocessing, feature engineering (once)
    logger.info("Phase 1: Setup (EDA + Preprocessing + Feature Engineering)")
    setup_pipeline = _create_setup_pipeline(config, progress_hook)
    state = setup_pipeline.invoke(initial_state)

    # Phase 2: Parallel training — all models at once, pick best
    logger.info("Phase 2: Parallel Training (all models simultaneously)")
    training_pipeline = _create_training_pipeline(config, progress_hook)
    final_state = training_pipeline.invoke(state)

    # Collect async EDA insights if ready (10s timeout, non-blocking fallback)
    future = final_state.get("_eda_insights_future")
    if future is not None:
        try:
            insights = future.result(timeout=10)
            final_state["eda_summary"]["llm_insights"] = insights
            logger.info("EDA LLM insights collected at pipeline end")
        except Exception as e:
            logger.warning(f"EDA insights not ready in time: {e}")

    logger.info("=" * 60)
    logger.info(f"Pipeline done — best: {final_state['best_model']} ({final_state['best_score']:.4f})")
    logger.info("=" * 60)
    return final_state
