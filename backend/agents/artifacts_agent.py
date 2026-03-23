"""
Artifacts Agent — Production-ready output hub.

Saves for every pipeline run:
  artifacts/{dataset_id}/{version}/
    best_model.pkl          — serialised sklearn model
    reproducibility.json    — seeds, pipeline steps, preprocessing config
    inference_samples.json  — 5 sample rows + expected predictions
    api_export.py           — drop-in FastAPI inference endpoint
    training_log.csv        — per-model metrics row
    experiment_log.json     — full structured experiment record
    drift_hooks.json        — baseline feature stats for drift detection
    metadata.json           — version manifest
"""

import os
import json
import csv
import hashlib
import pickle
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from graph.state import AutoMLState
from utils.logger import logger

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


# ── Firestore client (lazy init) ──────────────────────────────────────────────

_fs_client = None

def _get_firestore():
    global _fs_client
    if _fs_client is not None:
        return _fs_client
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not firebase_admin._apps:
            # Check env var first, then look next to this file
            cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
            if not cred_path:
                cred_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "firebase-service-account.json")
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
            else:
                cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
        _fs_client = firestore.client()
        return _fs_client
    except Exception as e:
        logger.warning(f"Firestore init failed (non-fatal): {e}")
        return None


def _save_to_firestore_experiments(rows: List[Dict[str, Any]], dataset_id: str, version: str):
    """Write experiment rows to Firestore all_experiments collection."""
    try:
        db = _get_firestore()
        if db is None:
            return
        for row in rows:
            doc_id = f"{dataset_id}_{version}_{row.get('model_name', 'unknown')}"
            clean = json.loads(json.dumps(row, default=str))
            db.collection("all_experiments").document(doc_id).set(clean, merge=True)
        logger.info(f"Saved {len(rows)} experiment rows to Firestore")
    except Exception as e:
        logger.warning(f"Firestore experiment write failed (non-fatal): {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dataset_hash(df: pd.DataFrame) -> str:
    """Stable SHA-256 of the dataframe content (shape + column names + first 100 rows)."""
    raw = f"{df.shape}|{list(df.columns)}|{df.head(100).to_csv(index=False)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _version_dir(dataset_id: str, version: str) -> str:
    path = os.path.join(ARTIFACTS_DIR, dataset_id, version)
    os.makedirs(path, exist_ok=True)
    return path


def _next_version(dataset_id: str) -> str:
    base = os.path.join(ARTIFACTS_DIR, dataset_id)
    if not os.path.exists(base):
        return "v1"
    existing = [d for d in os.listdir(base) if d.startswith("v") and d[1:].isdigit()]
    if not existing:
        return "v1"
    latest = max(int(d[1:]) for d in existing)
    return f"v{latest + 1}"


# ── Step 1: Save model (.pkl) ─────────────────────────────────────────────────

def _save_model(model, vdir: str) -> str:
    path = os.path.join(vdir, "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(path) // 1024
    logger.info(f"Model saved: {path} ({size_kb} KB)")
    return path


# ── Step 2: Reproducibility config ───────────────────────────────────────────

def _save_reproducibility(state: AutoMLState, vdir: str, dataset_hash: str):
    config = {
        "random_seed":          42,
        "train_test_split":     {"test_size": 0.2, "random_state": 42, "stratify": state["task_type"] == "classification"},
        "cv_strategy":          {"folds": 5, "shuffle": True, "random_state": 42},
        "hpo":                  {"method": "RandomizedSearchCV", "n_iter": 8, "random_state": 42},
        "preprocessing_steps":  [
            "drop_high_missing (>50%)",
            "drop_zero_variance",
            "knn_imputation",
            "iqr_winsorization",
            "ordinal_encoding",
            "type_coercion",
        ],
        "preprocessing_config": state.get("preprocessing_config", {}),
        "selected_features":    state["selected_features"],
        "target_column":        state["target_column"],
        "task_type":            state["task_type"],
        "dataset_hash":         dataset_hash,
        "framework":            "scikit-learn",
        "python_version":       _python_version(),
        "sklearn_version":      _sklearn_version(),
    }
    path = os.path.join(vdir, "reproducibility.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def _python_version() -> str:
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "unknown"


# ── Step 3: Inference samples ─────────────────────────────────────────────────

def _save_inference_samples(model, df: pd.DataFrame, target_col: str,
                             feature_names: List[str], vdir: str):
    try:
        X = df[feature_names].head(5)
        preds = model.predict(X.values)
        samples = []
        for i, row in X.iterrows():
            samples.append({
                "input":      row.to_dict(),
                "prediction": float(preds[i]) if isinstance(preds[i], (np.floating, float)) else int(preds[i]),
                "actual":     float(df[target_col].iloc[i]) if pd.api.types.is_float_dtype(df[target_col]) else int(df[target_col].iloc[i]),
            })
        path = os.path.join(vdir, "inference_samples.json")
        with open(path, "w") as f:
            json.dump(samples, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Inference samples failed: {e}")


# ── Step 4: API export (FastAPI snippet) ──────────────────────────────────────

def _save_api_export(model_name: str, feature_names: List[str],
                     task_type: str, vdir: str, version: str, dataset_id: str):
    fields = "\n".join(f"    {f.replace(' ', '_')}: float" for f in feature_names)
    predict_call = "model.predict([list(data.dict().values())])[0]"
    code = f'''"""
Auto-generated FastAPI inference endpoint
Model   : {model_name}
Version : {version}
Dataset : {dataset_id}
Task    : {task_type}
Generated: {datetime.now(timezone.utc).isoformat()}
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — {model_name}")

with open(os.path.join(os.path.dirname(__file__), "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
{fields}

@app.post("/predict")
def predict(data: InputData):
    result = {predict_call}
    return {{"prediction": float(result), "model": "{model_name}", "version": "{version}"}}

@app.get("/health")
def health():
    return {{"status": "ok", "model": "{model_name}", "version": "{version}"}}
'''
    path = os.path.join(vdir, "api_export.py")
    with open(path, "w") as f:
        f.write(code)


# ── Step 5: Training log (CSV — one row per model) ────────────────────────────

def _save_training_log(eval_results: List[Dict[str, Any]], task_type: str,
                       vdir: str, version: str, dataset_id: str, timestamp: str):
    is_clf = task_type == "classification"
    rows = []
    for r in eval_results:
        m = r.get("metrics", {})
        rows.append({
            "dataset_id":    dataset_id,
            "version":       version,
            "timestamp":     timestamp,
            "model_name":    r.get("model_name", ""),
            "composite_score": r.get("score", ""),
            "cv_score":      m.get("accuracy" if is_clf else "r2_score", ""),
            "test_score":    m.get("test_accuracy" if is_clf else "test_r2", ""),
            "train_score":   m.get("train_accuracy" if is_clf else "train_r2", ""),
            "f1_score":      m.get("f1_score", ""),
            "roc_auc":       m.get("roc_auc", ""),
            "rmse":          m.get("rmse", ""),
            "mae":           m.get("mae", ""),
            "cv_std":        m.get("cv_std", ""),
            "overfit_gap":   m.get("overfit_gap", ""),
            "overfit_penalty": m.get("overfit_penalty", ""),
            "train_set_size": m.get("train_set_size", ""),
            "test_set_size": m.get("test_set_size", ""),
            "best_params":   json.dumps(m.get("best_params", {})),
        })

    path = os.path.join(vdir, "training_log.csv")
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Also append to global experiment log CSV (all versions, all datasets)
    global_csv = os.path.join(ARTIFACTS_DIR, "all_experiments.csv")
    write_header = not os.path.exists(global_csv)
    with open(global_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if write_header and rows:
            writer.writeheader()
        writer.writerows(rows)

    # Also write to Firestore global experiments collection
    _save_to_firestore_experiments(rows, dataset_id, version)


# ── Step 6: Experiment log (JSON — full structured record) ────────────────────

def _save_experiment_log(state: AutoMLState, eval_results: List[Dict[str, Any]],
                         vdir: str, version: str, dataset_id: str,
                         timestamp: str, training_time_s: float, dataset_hash: str):
    record = {
        "experiment_id":    f"{dataset_id}_{version}",
        "dataset_id":       dataset_id,
        "version":          version,
        "timestamp":        timestamp,
        "training_time_s":  round(training_time_s, 2),
        "dataset_hash":     dataset_hash,
        "task_type":        state["task_type"],
        "target_column":    state["target_column"],
        "selected_features": state["selected_features"],
        "n_features":       len(state["selected_features"]),
        "best_model":       state["best_model"],
        "best_score":       state["best_score"],
        "best_params":      state.get("best_params", {}),
        "justification":    state.get("justification", ""),
        "is_underfit":      state.get("is_underfit", False),
        "models_evaluated": [
            {
                "model_name": r.get("model_name"),
                "score":      r.get("score"),
                "metrics":    r.get("metrics", {}),
                "params":     r.get("params", {}),
            }
            for r in eval_results
        ],
        "preprocessing_report": state.get("preprocessing_config", {}),
        "agent_logs":       state.get("agent_logs", []),
    }
    path = os.path.join(vdir, "experiment_log.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return record


# ── Step 7: Drift detection hooks (baseline feature stats) ───────────────────

def _save_drift_hooks(df: pd.DataFrame, feature_names: List[str], vdir: str):
    try:
        X = df[feature_names].select_dtypes(include=[np.number])
        stats = {}
        for col in X.columns:
            s = X[col].dropna()
            stats[col] = {
                "mean":   round(float(s.mean()), 6),
                "std":    round(float(s.std()), 6),
                "min":    round(float(s.min()), 6),
                "max":    round(float(s.max()), 6),
                "p25":    round(float(s.quantile(0.25)), 6),
                "p50":    round(float(s.quantile(0.50)), 6),
                "p75":    round(float(s.quantile(0.75)), 6),
                "null_rate": round(float(df[col].isnull().mean()), 6),
            }
        hooks = {
            "baseline_version": vdir.split(os.sep)[-1],
            "n_rows":           len(df),
            "feature_stats":    stats,
            "drift_thresholds": {
                "mean_shift_z":   2.0,
                "std_ratio":      1.5,
                "null_rate_delta": 0.05,
            },
            "note": "Compare incoming batch stats against feature_stats to detect drift.",
        }
        path = os.path.join(vdir, "drift_hooks.json")
        with open(path, "w") as f:
            json.dump(hooks, f, indent=2)
    except Exception as e:
        logger.warning(f"Drift hooks failed: {e}")


# ── Step 8: Version manifest ──────────────────────────────────────────────────

def _save_metadata(model_name: str, version: str, dataset_id: str,
                   timestamp: str, best_score: float, task_type: str,
                   best_params: Dict[str, Any], dataset_hash: str,
                   training_time_s: float, vdir: str) -> Dict[str, Any]:
    meta = {
        "version":         version,
        "dataset_id":      dataset_id,
        "dataset_hash":    dataset_hash,
        "model_name":      model_name,
        "task_type":       task_type,
        "best_score":      round(best_score, 4) if best_score not in (float("inf"), float("-inf")) else None,
        "best_params":     best_params or {},
        "timestamp":       timestamp,
        "training_time_s": round(training_time_s, 2),
        "framework":       "scikit-learn",
        "format":          "pkl",
        "files": [
            "best_model.pkl",
            "reproducibility.json",
            "inference_samples.json",
            "api_export.py",
            "training_log.csv",
            "experiment_log.json",
            "drift_hooks.json",
            "metadata.json",
        ],
    }
    path = os.path.join(vdir, "metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ── Main agent ────────────────────────────────────────────────────────────────

def artifacts_agent(state: AutoMLState, config: Dict[str, Any],
                    fitted_model=None, training_start_time: float = 0.0,
                    dataset_id: str = None) -> AutoMLState:
    logger.info("=== Artifacts Agent Started ===")

    dataset_id    = dataset_id or state.get("_dataset_id") or "unknown"
    task_type     = state["task_type"]
    target_col    = state["target_column"]
    feature_names = state["selected_features"]
    eval_results  = state["evaluation_results"]
    best_model_name = state["best_model"]
    best_score    = state["best_score"]
    best_params   = state.get("best_params") or {}

    df = state["processed_data"].to_pandas()
    dataset_hash = _dataset_hash(df)
    version = _next_version(dataset_id)
    timestamp = datetime.now(timezone.utc).isoformat()
    training_time_s = time.time() - training_start_time if training_start_time else 0.0

    vdir = _version_dir(dataset_id, version)
    logger.info(f"Saving artifacts to {vdir}")

    # 1. Save model
    model_path = None
    if fitted_model is not None:
        model_path = _save_model(fitted_model, vdir)

    # 2. Reproducibility config
    _save_reproducibility(state, vdir, dataset_hash)

    # 3. Inference samples
    if fitted_model is not None:
        _save_inference_samples(fitted_model, df, target_col, feature_names, vdir)

    # 4. API export
    _save_api_export(best_model_name, feature_names, task_type, vdir, version, dataset_id)

    # 5. Training log CSV
    _save_training_log(eval_results, task_type, vdir, version, dataset_id, timestamp)

    # 6. Experiment log JSON
    experiment_record = _save_experiment_log(
        state, eval_results, vdir, version, dataset_id,
        timestamp, training_time_s, dataset_hash
    )

    # 7. Drift hooks
    _save_drift_hooks(df, feature_names, vdir)

    # 8. Metadata manifest
    meta = _save_metadata(
        best_model_name, version, dataset_id, timestamp,
        best_score, task_type, best_params, dataset_hash,
        training_time_s, vdir
    )

    state["artifact_version"]  = version
    state["artifact_dir"]      = vdir
    state["artifact_metadata"] = meta
    state["agent_logs"].append(f"Artifacts saved: {vdir} ({version})")

    logger.info(f"=== Artifacts Agent Completed: {version} ===")
    return state
