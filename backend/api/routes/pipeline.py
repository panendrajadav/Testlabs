import os
import json
import asyncio
import time
import yaml
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from api.schemas import PipelineRequest, PipelineStatusResponse
from graph.pipeline import run_pipeline
from graph.state import AutoMLState
from agents.artifacts_agent import artifacts_agent
from utils.logger import logger

router = APIRouter()
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR  = os.path.join(_BASE, "uploads")
RESULTS_DIR = os.path.join(_BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── In-memory stores ──────────────────────────────────────────────────────────
pipeline_jobs: dict = {}       # dataset_id → {status, progress, result, error}
_result_cache: dict = {}       # dataset_id → result payload (fastest read path)
_ws_clients: dict = {}         # dataset_id → list[WebSocket]

# Dedicated thread pool — keeps FastAPI event loop free during CPU-bound pipeline
_pipeline_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pipeline")


def _load_config() -> dict:
    import re
    config_path = os.path.join(_BASE, "config.yaml")
    with open(config_path) as f:
        raw = f.read()
    raw = re.sub(r'\$\{([^}]+)\}', lambda m: os.getenv(m.group(1), m.group(0)), raw)
    return yaml.safe_load(raw)


def _detect_target(df: pl.DataFrame) -> str:
    return df.columns[-1]


def _write_status(dataset_id: str, status: str, progress: str):
    """Persist lightweight status file so restarts can show in-progress jobs."""
    path = os.path.join(RESULTS_DIR, f"{dataset_id}.status.json")
    with open(path, "w") as f:
        json.dump({"status": status, "progress": progress}, f)


async def _broadcast(dataset_id: str, payload: dict):
    """Push update to all WebSocket clients watching this dataset."""
    clients = _ws_clients.get(dataset_id, [])
    dead = []
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.remove(ws)


def _run_pipeline_job(dataset_id: str, target_column: str, config: dict, loop: asyncio.AbstractEventLoop):
    """Runs in a thread — never awaits, uses loop.call_soon_threadsafe for WS pushes."""

    def _push(payload: dict):
        asyncio.run_coroutine_threadsafe(_broadcast(dataset_id, payload), loop)

    try:
        pipeline_jobs[dataset_id]["status"] = "running"
        pipeline_jobs[dataset_id]["progress"] = "Starting"
        _write_status(dataset_id, "running", "Starting")
        _push({"status": "running", "progress": "Starting"})

        file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
        df = pl.read_csv(
            file_path,
            null_values=["NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "", "?"],
            infer_schema_length=10000,
            ignore_errors=True,
        )

        if not target_column:
            target_column = _detect_target(df)

        initial_state: AutoMLState = {
            "raw_data":             df,
            "processed_data":       None,
            "target_column":        target_column,
            "task_type":            config["automl"]["task_type"],
            "dataset_metadata":     {},
            "eda_summary":          {},
            "eda_plots":            {},
            "preprocessing_config": {},
            "preprocessing_report": {},
            "feature_config":       {},
            "selected_features":    [],
            "models_tried":         [],
            "current_model":        "",
            "current_params":       {},
            "evaluation_results":   [],
            "best_model":           None,
            "best_score":           -float("inf"),
            "best_params":          None,
            "roc_data":             None,
            "shap_values":          None,
            "iteration":            1,
            "max_iterations":       config["automl"]["max_iterations"],
            "score_threshold":      config["automl"]["score_threshold"],
            "agent_logs":           [],
            "final_report":         None,
            "justification":        None,
            "is_underfit":          None,
            "artifact_version":     None,
            "artifact_dir":         None,
            "artifact_metadata":    None,
            "_dataset_id":          dataset_id,
        }

        def progress_hook(agent_name: str):
            pipeline_jobs[dataset_id]["progress"] = agent_name
            _write_status(dataset_id, "running", agent_name)
            _push({"status": "running", "progress": agent_name})

        training_start = time.time()
        final_state = run_pipeline(initial_state, config, progress_hook)

        # ── Save production artifacts ─────────────────────────────────────────
        progress_hook("Saving Artifacts")
        try:
            # Extract fitted best model stored transiently by evaluation_agent
            fitted_model = final_state.pop("_fitted_model", None)
            final_state = artifacts_agent(
                final_state, config,
                fitted_model=fitted_model,
                dataset_id=dataset_id,
                training_start_time=training_start,
            )
        except Exception as ae:
            logger.warning(f"Artifacts agent failed (non-fatal): {ae}")

        best_score = final_state["best_score"]
        safe_score = round(best_score, 4) if best_score not in (float("inf"), float("-inf")) else None

        result_payload = {
            "target_column":      final_state["target_column"],
            "task_type":          final_state["task_type"],
            "best_model":         final_state["best_model"],
            "best_score":         safe_score,
            "best_params":        final_state["best_params"],
            "evaluation_results": final_state["evaluation_results"],
            "eda_summary":        {k: v for k, v in final_state["eda_summary"].items() if k != "llm_insights"},
            "eda_plots":          final_state["eda_plots"],
            "eda_insights":       final_state["eda_summary"].get("llm_insights", ""),
            "selected_features":  final_state["selected_features"],
            "preprocessing_report": final_state.get("preprocessing_config", {}),
            "roc_data":           final_state.get("roc_data"),
            "shap_values":        final_state.get("shap_values"),
            "agent_logs":         final_state["agent_logs"],
            "justification":      final_state.get("justification"),
            "is_underfit":        final_state.get("is_underfit", False),
            "artifact_version":   final_state.get("artifact_version"),
            "artifact_metadata":  final_state.get("artifact_metadata"),
        }

        _result_cache[dataset_id] = result_payload
        pipeline_jobs[dataset_id].update({"status": "completed", "progress": "Done", "result": result_payload})

        # Persist result + remove status file (result file IS the completed marker)
        result_file = os.path.join(RESULTS_DIR, f"{dataset_id}.json")
        with open(result_file, "w") as f:
            json.dump(result_payload, f, default=str)

        status_file = os.path.join(RESULTS_DIR, f"{dataset_id}.status.json")
        if os.path.exists(status_file):
            os.remove(status_file)

        _push({"status": "completed", "progress": "Done", "result": result_payload})
        logger.info(f"Pipeline completed for {dataset_id}: {final_state['best_model']} ({safe_score})")

    except Exception as e:
        logger.error(f"Pipeline job failed for {dataset_id}: {e}", exc_info=True)
        pipeline_jobs[dataset_id].update({"status": "failed", "error": str(e)})
        _write_status(dataset_id, "failed", str(e))
        _push({"status": "failed", "error": str(e)})


@router.post("/run", response_model=PipelineStatusResponse)
async def run_pipeline_endpoint(request: PipelineRequest):
    file_path = os.path.join(UPLOAD_DIR, f"{request.dataset_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")

    # Return immediately if already completed (memory cache)
    if request.dataset_id in _result_cache:
        return PipelineStatusResponse(
            dataset_id=request.dataset_id,
            status="completed",
            progress="Done",
            result=_result_cache[request.dataset_id]
        )

    # Idempotent — don't re-run if already running/queued
    if request.dataset_id in pipeline_jobs:
        job = pipeline_jobs[request.dataset_id]
        if job["status"] in ("running", "queued"):
            return PipelineStatusResponse(
                dataset_id=request.dataset_id,
                status=job["status"],
                progress=job.get("progress", "Queued")
            )

    config = _load_config()
    pipeline_jobs[request.dataset_id] = {"status": "queued", "progress": "Queued", "result": None, "error": None}
    _write_status(request.dataset_id, "queued", "Queued")

    # Submit to thread pool — event loop stays free
    loop = asyncio.get_event_loop()
    _pipeline_executor.submit(
        _run_pipeline_job,
        request.dataset_id,
        request.target_column or "",
        config,
        loop
    )

    return PipelineStatusResponse(dataset_id=request.dataset_id, status="queued", progress="Pipeline queued")


@router.get("/status/{dataset_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(dataset_id: str):
    # 1. Memory cache — fastest (no I/O)
    if dataset_id in _result_cache:
        return PipelineStatusResponse(
            dataset_id=dataset_id, status="completed", progress="Done",
            result=_result_cache[dataset_id]
        )

    # 2. Active in-memory job
    if dataset_id in pipeline_jobs:
        job = pipeline_jobs[dataset_id]
        return PipelineStatusResponse(
            dataset_id=dataset_id,
            status=job["status"],
            progress=job.get("progress"),
            result=job.get("result"),
            error=job.get("error")
        )

    # 3. Disk — completed result (survives restarts)
    result_file = os.path.join(RESULTS_DIR, f"{dataset_id}.json")
    if os.path.exists(result_file):
        with open(result_file) as f:
            result = json.load(f)
        _result_cache[dataset_id] = result
        return PipelineStatusResponse(dataset_id=dataset_id, status="completed", progress="Done", result=result)

    # 4. Disk — in-progress status file (running job survived restart)
    status_file = os.path.join(RESULTS_DIR, f"{dataset_id}.status.json")
    if os.path.exists(status_file):
        with open(status_file) as f:
            s = json.load(f)
        return PipelineStatusResponse(
            dataset_id=dataset_id,
            status=s.get("status", "unknown"),
            progress=s.get("progress")
        )

    raise HTTPException(status_code=404, detail="No pipeline job found for this dataset_id")


@router.websocket("/ws/{dataset_id}")
async def pipeline_websocket(websocket: WebSocket, dataset_id: str):
    """Real-time pipeline progress via WebSocket. Falls back gracefully if client disconnects."""
    await websocket.accept()

    # Register client
    _ws_clients.setdefault(dataset_id, []).append(websocket)

    # Send current state immediately on connect
    if dataset_id in _result_cache:
        await websocket.send_json({"status": "completed", "progress": "Done", "result": _result_cache[dataset_id]})
    elif dataset_id in pipeline_jobs:
        job = pipeline_jobs[dataset_id]
        await websocket.send_json({"status": job["status"], "progress": job.get("progress", "")})

    try:
        # Keep connection alive — client sends pings, we echo
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        clients = _ws_clients.get(dataset_id, [])
        if websocket in clients:
            clients.remove(websocket)
