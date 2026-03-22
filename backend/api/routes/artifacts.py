import os
import json
import csv
import shutil
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any
from utils.logger import logger

router = APIRouter()
ARTIFACTS_DIR = "artifacts"


def _version_meta(dataset_id: str, version: str) -> Dict[str, Any]:
    path = os.path.join(ARTIFACTS_DIR, dataset_id, version, "metadata.json")
    if not os.path.exists(path):
        return {"version": version, "dataset_id": dataset_id}
    with open(path) as f:
        return json.load(f)


def _list_versions(dataset_id: str) -> List[Dict[str, Any]]:
    base = os.path.join(ARTIFACTS_DIR, dataset_id)
    if not os.path.exists(base):
        return []
    versions = sorted(
        [d for d in os.listdir(base) if d.startswith("v") and d[1:].isdigit()],
        key=lambda v: int(v[1:]),
        reverse=True,
    )
    return [_version_meta(dataset_id, v) for v in versions]


# ── List all dataset IDs that have artifacts ─────────────────────────────────
@router.get("/datasets/all")
def list_artifact_datasets():
    if not os.path.exists(ARTIFACTS_DIR):
        return {"datasets": []}
    datasets = [
        d for d in os.listdir(ARTIFACTS_DIR)
        if os.path.isdir(os.path.join(ARTIFACTS_DIR, d)) and d != "all_experiments.csv"
    ]
    return {"datasets": datasets}


# ── List all versions for a dataset ──────────────────────────────────────────
@router.get("/{dataset_id}/versions")
def list_versions(dataset_id: str):
    versions = _list_versions(dataset_id)
    return {
        "dataset_id": dataset_id,
        "versions": versions,
        "latest": versions[0]["version"] if versions else None,
    }


# ── Get active version ────────────────────────────────────────────────────────
@router.get("/{dataset_id}/active")
def get_active_version(dataset_id: str):
    pointer = os.path.join(ARTIFACTS_DIR, dataset_id, "active_version.json")
    if os.path.exists(pointer):
        with open(pointer) as f:
            data = json.load(f)
        version = data.get("active_version")
    else:
        versions = _list_versions(dataset_id)
        if not versions:
            raise HTTPException(status_code=404, detail="No artifacts found")
        version = versions[0]["version"]
    return {"dataset_id": dataset_id, "active_version": version, **_version_meta(dataset_id, version)}


# ── Global experiment history (all datasets) ──────────────────────────────────
@router.get("/experiments/all")
def all_experiments():
    path = os.path.join(ARTIFACTS_DIR, "all_experiments.csv")
    if not os.path.exists(path):
        return {"rows": []}
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return {"rows": rows}


# ── Summary: everything for one version in one call ───────────────────────────
@router.get("/{dataset_id}/{version}/summary")
def get_summary(dataset_id: str, version: str):
    vdir = os.path.join(ARTIFACTS_DIR, dataset_id, version)
    if not os.path.exists(vdir):
        raise HTTPException(status_code=404, detail="Version not found")

    def _read_json(name):
        p = os.path.join(vdir, name)
        return json.load(open(p)) if os.path.exists(p) else None

    def _read_csv(name):
        p = os.path.join(vdir, name)
        if not os.path.exists(p):
            return []
        with open(p, newline="") as f:
            return list(csv.DictReader(f))

    def _read_text(name):
        p = os.path.join(vdir, name)
        return open(p).read() if os.path.exists(p) else None

    meta = _read_json("metadata.json") or {}
    model_file = os.path.join(vdir, "best_model.pkl")
    model_size_kb = os.path.getsize(model_file) // 1024 if os.path.exists(model_file) else None

    return {
        "metadata":          meta,
        "experiment":        _read_json("experiment_log.json"),
        "training_log":      _read_csv("training_log.csv"),
        "reproducibility":   _read_json("reproducibility.json"),
        "inference_samples": _read_json("inference_samples.json"),
        "api_export_code":   _read_text("api_export.py"),
        "drift_hooks":       _read_json("drift_hooks.json"),
        "model_file_exists": os.path.exists(model_file),
        "model_size_kb":     model_size_kb,
    }


# ── Get metadata for a specific version ──────────────────────────────────────
@router.get("/{dataset_id}/{version}/metadata")
def get_metadata(dataset_id: str, version: str):
    meta = _version_meta(dataset_id, version)
    if not meta.get("model_name"):
        raise HTTPException(status_code=404, detail="Version not found")
    return meta


# ── Get experiment log for a version ─────────────────────────────────────────
@router.get("/{dataset_id}/{version}/experiment")
def get_experiment(dataset_id: str, version: str):
    path = os.path.join(ARTIFACTS_DIR, dataset_id, version, "experiment_log.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Experiment log not found")
    with open(path) as f:
        return json.load(f)


# ── Get training log (CSV → JSON) ─────────────────────────────────────────────
@router.get("/{dataset_id}/{version}/training-log")
def get_training_log(dataset_id: str, version: str):
    path = os.path.join(ARTIFACTS_DIR, dataset_id, version, "training_log.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Training log not found")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return {"version": version, "rows": rows}


# ── Get reproducibility config ────────────────────────────────────────────────
@router.get("/{dataset_id}/{version}/reproducibility")
def get_reproducibility(dataset_id: str, version: str):
    path = os.path.join(ARTIFACTS_DIR, dataset_id, version, "reproducibility.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Reproducibility config not found")
    with open(path) as f:
        return json.load(f)


# ── Get inference samples ─────────────────────────────────────────────────────
@router.get("/{dataset_id}/{version}/inference-samples")
def get_inference_samples(dataset_id: str, version: str):
    path = os.path.join(ARTIFACTS_DIR, dataset_id, version, "inference_samples.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Inference samples not found")
    with open(path) as f:
        return json.load(f)


# ── Get drift hooks ───────────────────────────────────────────────────────────
@router.get("/{dataset_id}/{version}/drift-hooks")
def get_drift_hooks(dataset_id: str, version: str):
    path = os.path.join(ARTIFACTS_DIR, dataset_id, version, "drift_hooks.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Drift hooks not found")
    with open(path) as f:
        return json.load(f)


# ── Get API export code ───────────────────────────────────────────────────────
@router.get("/{dataset_id}/{version}/api-export")
def get_api_export(dataset_id: str, version: str):
    path = os.path.join(ARTIFACTS_DIR, dataset_id, version, "api_export.py")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="API export not found")
    with open(path) as f:
        return {"version": version, "code": f.read()}


# ── Download model file ───────────────────────────────────────────────────────
@router.get("/{dataset_id}/{version}/download")
def download_model(dataset_id: str, version: str):
    path = os.path.join(ARTIFACTS_DIR, dataset_id, version, "best_model.pkl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model file not found")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{dataset_id}_{version}_model.pkl",
    )


# ── Rollback: promote a previous version as "active" ─────────────────────────
@router.post("/{dataset_id}/{version}/rollback")
def rollback_version(dataset_id: str, version: str):
    vdir = os.path.join(ARTIFACTS_DIR, dataset_id, version)
    if not os.path.exists(vdir):
        raise HTTPException(status_code=404, detail="Version not found")

    # Write a pointer file marking this as the active version
    pointer = os.path.join(ARTIFACTS_DIR, dataset_id, "active_version.json")
    with open(pointer, "w") as f:
        json.dump({"active_version": version}, f)

    logger.info(f"Rolled back {dataset_id} to {version}")
    return {"dataset_id": dataset_id, "active_version": version, "status": "rolled_back"}


# ── Delete a version ──────────────────────────────────────────────────────────
@router.delete("/{dataset_id}/{version}")
def delete_version(dataset_id: str, version: str):
    vdir = os.path.join(ARTIFACTS_DIR, dataset_id, version)
    if not os.path.exists(vdir):
        raise HTTPException(status_code=404, detail="Version not found")
    shutil.rmtree(vdir)
    logger.info(f"Deleted artifact version {dataset_id}/{version}")
    return {"deleted": True, "version": version}


# ── Performance comparison across versions ────────────────────────────────────
@router.get("/{dataset_id}/compare")
def compare_versions(dataset_id: str):
    versions = _list_versions(dataset_id)
    if not versions:
        raise HTTPException(status_code=404, detail="No versions found")
    comparison = []
    for v in versions:
        ver = v.get("version")
        exp_path = os.path.join(ARTIFACTS_DIR, dataset_id, ver, "experiment_log.json")
        if os.path.exists(exp_path):
            with open(exp_path) as f:
                exp = json.load(f)
            comparison.append({
                "version":          ver,
                "timestamp":        exp.get("timestamp"),
                "best_model":       exp.get("best_model"),
                "best_score":       exp.get("best_score"),
                "training_time_s":  exp.get("training_time_s"),
                "n_features":       exp.get("n_features"),
                "dataset_hash":     exp.get("dataset_hash"),
                "is_underfit":      exp.get("is_underfit"),
            })
    return {"dataset_id": dataset_id, "comparison": comparison}
