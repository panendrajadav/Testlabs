from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from graph.state import AutoMLState
from agents.eda_agent import eda_agent
from agents.preprocessing_agent import preprocessing_agent
from agents.feature_engineering_agent import feature_engineering_agent
from agents.model_selection_agent import model_selection_agent
from agents.hyperparameter_agent import hyperparameter_agent
from agents.evaluation_agent import evaluation_agent
from utils.logger import logger

def create_pipeline(config: Dict[str, Any], progress_hook=None) -> StateGraph:
    """Create LangGraph pipeline for AutoML."""

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

    def model_selection_node(state: AutoMLState) -> AutoMLState:
        _hook("Model Selection")
        return model_selection_agent(state, config)

    def hyperparameter_node(state: AutoMLState) -> AutoMLState:
        if state["current_model"] == "__done__":
            return state
        _hook("HPO")
        return hyperparameter_agent(state, config)

    def evaluation_node(state: AutoMLState) -> AutoMLState:
        _hook("Evaluation")
        return evaluation_agent(state, config)
    
    def should_continue(state: AutoMLState) -> Literal["continue", "end"]:
        """Continue until all models are evaluated."""
        # Sentinel: model selection has no more models
        if state["current_model"] == "__done__":
            logger.info("All models evaluated — ending pipeline")
            return "end"

        iteration  = state["iteration"]
        max_iter   = state["max_iterations"]
        best_score = state["best_score"]
        threshold  = state["score_threshold"]

        logger.info(f"Decision Point - Iteration: {iteration}/{max_iter}, Best Score: {best_score:.4f}, Threshold: {threshold}")

        if iteration >= max_iter:
            logger.info("Max iterations reached — ending pipeline")
            return "end"

        logger.info("Continuing to next model")
        return "continue"
    
    # Build graph
    workflow = StateGraph(AutoMLState)
    
    # Add nodes
    workflow.add_node("eda", eda_node)
    workflow.add_node("preprocessing", preprocessing_node)
    workflow.add_node("feature_engineering", feature_engineering_node)
    workflow.add_node("model_selection", model_selection_node)
    workflow.add_node("hyperparameter_tuning", hyperparameter_node)
    workflow.add_node("evaluation", evaluation_node)
    
    # Add edges
    workflow.set_entry_point("eda")
    workflow.add_edge("eda", "preprocessing")
    workflow.add_edge("preprocessing", "feature_engineering")
    workflow.add_edge("feature_engineering", "model_selection")
    workflow.add_edge("model_selection", "hyperparameter_tuning")
    workflow.add_edge("hyperparameter_tuning", "evaluation")
    
    # Conditional edge from evaluation
    workflow.add_conditional_edges(
        "evaluation",
        should_continue,
        {
            "continue": "model_selection",
            "end": END
        }
    )
    
    return workflow.compile()

def run_pipeline(initial_state: AutoMLState, config: Dict[str, Any], progress_hook=None) -> AutoMLState:
    """Run the complete AutoML pipeline."""
    logger.info("=" * 60)
    logger.info("Starting AutoML Pipeline")
    logger.info("=" * 60)

    pipeline = create_pipeline(config, progress_hook)
    final_state = pipeline.invoke(initial_state)
    
    logger.info("=" * 60)
    logger.info("AutoML Pipeline Completed")
    logger.info("=" * 60)
    
    return final_state
