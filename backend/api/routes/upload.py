import uuid
import os
import shutil
import polars as pl
from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.helpers import detect_task_type, get_column_types

router = APIRouter()

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR    = os.path.join(_BASE, "uploads")
ARTIFACTS_DIR = os.path.join(_BASE, "artifacts")
RESULTS_DIR   = os.path.join(_BASE, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _auto_detect_target(df: pl.DataFrame) -> str:
    """Heuristic: last column, or first column whose name hints at a label."""
    hint_words = {"target", "label", "class", "churn", "outcome", "y", "output", "result", "survived", "default"}
    for col in reversed(df.columns):
        if any(h in col.lower() for h in hint_words):
            return col
    return df.columns[-1]


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    df = pl.read_csv(
        file_path,
        null_values=["NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "", "?"],
        infer_schema_length=10000,
        ignore_errors=True,
    )
    target_col = _auto_detect_target(df)
    col_types = get_column_types(df.drop(target_col))
    task_type = detect_task_type(df[target_col])

    return {
        "dataset_id":    dataset_id,
        "filename":      file.filename,
        "rows":          df.height,
        "columns":       df.width,
        "column_names":  df.columns,
        "target_column": target_col,
        "task_type":     task_type,
        "numeric_cols":  col_types["numeric"],
        "categorical_cols": col_types["categorical"],
        "preview":       df.head(5).to_dicts(),
    }


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Hard-delete all server-side data for a dataset: CSV, artifacts, results."""
    deleted = []

    # 1. Uploaded CSV
    csv_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
        deleted.append("upload")

    # 2. Artifacts folder (all versions)
    artifacts_path = os.path.join(ARTIFACTS_DIR, dataset_id)
    if os.path.exists(artifacts_path):
        shutil.rmtree(artifacts_path)
        deleted.append("artifacts")

    # 3. Pipeline result JSON
    result_path = os.path.join(RESULTS_DIR, f"{dataset_id}.json")
    if os.path.exists(result_path):
        os.remove(result_path)
        deleted.append("result")

    # 4. Pipeline status JSON (in-progress marker)
    status_path = os.path.join(RESULTS_DIR, f"{dataset_id}.status.json")
    if os.path.exists(status_path):
        os.remove(status_path)
        deleted.append("status")

    # 5. Clear in-memory pipeline caches (imported lazily to avoid circular import)
    try:
        from api.routes.pipeline import pipeline_jobs, _result_cache
        pipeline_jobs.pop(dataset_id, None)
        _result_cache.pop(dataset_id, None)
        deleted.append("memory_cache")
    except Exception:
        pass

    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {"deleted": True, "dataset_id": dataset_id, "removed": deleted}
