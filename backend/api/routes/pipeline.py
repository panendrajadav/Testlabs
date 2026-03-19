import os
import json
import yaml
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.schemas import PipelineRequest, PipelineStatusResponse
from graph.pipeline import run_pipeline
from graph.state import AutoMLState
from utils.logger import logger

router = APIRouter()
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# In-memory job store
pipeline_jobs: dict = {}

def _detect_target(df: pl.DataFrame) -> str:
    """Use last column as target — fast and reliable for standard datasets."""
    return df.columns[-1]

def _run_pipeline_job(dataset_id: str, target_column: str, config: dict):
    """Background job that runs the full AutoML pipeline."""
    try:
        pipeline_jobs[dataset_id]["status"] = "running"
        pipeline_jobs[dataset_id]["progress"] = "Loading dataset"

        file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
        df = pl.read_csv(file_path)

        pipeline_jobs[dataset_id]["progress"] = "Detecting target column"
        if not target_column:
            target_column = _detect_target(df)

        initial_state: AutoMLState = {
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
            "best_score": -float("inf"),
            "best_params": None,
            "roc_data": None,
            "shap_values": None,
            "iteration": 1,
            "max_iterations": config["automl"]["max_iterations"],
            "score_threshold": config["automl"]["score_threshold"],
            "agent_logs": [],
            "final_report": None,
        }

        # Update progress per agent so frontend pipeline animation works
        def progress_hook(agent_name: str):
            pipeline_jobs[dataset_id]["progress"] = agent_name

        pipeline_jobs[dataset_id]["progress"] = "EDA"
        final_state = run_pipeline(initial_state, config, progress_hook)

        pipeline_jobs[dataset_id]["status"] = "completed"
        pipeline_jobs[dataset_id]["progress"] = "Done"
        result_payload = {
            "target_column": final_state["target_column"],
            "task_type": final_state["task_type"],
            "best_model": final_state["best_model"],
            "best_score": round(final_state["best_score"], 4),
            "best_params": final_state["best_params"],
            "evaluation_results": final_state["evaluation_results"],
            "eda_summary": {k: v for k, v in final_state["eda_summary"].items() if k != "llm_insights"},
            "eda_plots": final_state["eda_plots"],
            "eda_insights": final_state["eda_summary"].get("llm_insights", ""),
            "selected_features": final_state["selected_features"],
            "roc_data": final_state.get("roc_data"),
            "shap_values": final_state.get("shap_values"),
            "agent_logs": final_state["agent_logs"],
        }
        pipeline_jobs[dataset_id]["result"] = result_payload

        # Persist to disk so results survive backend restarts
        result_file = os.path.join(RESULTS_DIR, f"{dataset_id}.json")
        with open(result_file, "w") as f:
            json.dump(result_payload, f, default=str)

    except Exception as e:
        logger.error(f"Pipeline job failed: {e}", exc_info=True)
        pipeline_jobs[dataset_id]["status"] = "failed"
        pipeline_jobs[dataset_id]["error"] = str(e)

_executor = ThreadPoolExecutor(max_workers=4)

@router.post("/run", response_model=PipelineStatusResponse)
async def run_pipeline_endpoint(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Trigger AutoML pipeline for an uploaded dataset."""
    file_path = os.path.join(UPLOAD_DIR, f"{request.dataset_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    pipeline_jobs[request.dataset_id] = {"status": "queued", "progress": "Queued", "result": None, "error": None}

    background_tasks.add_task(
        _run_pipeline_job,
        request.dataset_id,
        request.target_column,
        config
    )

    return PipelineStatusResponse(
        dataset_id=request.dataset_id,
        status="queued",
        progress="Pipeline queued"
    )

@router.get("/status/{dataset_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(dataset_id: str):
    """Poll pipeline status and results. Falls back to persisted result on backend restart."""
    # Check in-memory job store first
    if dataset_id in pipeline_jobs:
        job = pipeline_jobs[dataset_id]
        return PipelineStatusResponse(
            dataset_id=dataset_id,
            status=job["status"],
            progress=job.get("progress"),
            result=job.get("result"),
            error=job.get("error")
        )

    # Fall back to persisted result on disk (survives backend restart)
    result_file = os.path.join(RESULTS_DIR, f"{dataset_id}.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            result = json.load(f)
        return PipelineStatusResponse(
            dataset_id=dataset_id,
            status="completed",
            progress="Done",
            result=result,
            error=None
        )

    raise HTTPException(status_code=404, detail="No pipeline job found for this dataset_id")
