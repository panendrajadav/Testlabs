"""
Artifacts Agent — Production-ready output hub.

Saves for every pipeline run:
  artifacts/{dataset_id}/{version}/
    best_model.pkl          — serialised sklearn model
    scaler.pkl              — fitted StandardScaler (if used)
    encoder.pkl             — fitted OrdinalEncoder (if used)
    predict.py              — standalone inference script
    train.py                — full reproducible training script
    requirements.txt        — pinned dependencies
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


# ── Step 1: Save model + scaler + encoder (.pkl) ─────────────────────────────

def _save_model(model, vdir: str) -> str:
    path = os.path.join(vdir, "best_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(path) // 1024
    logger.info(f"Model saved: {path} ({size_kb} KB)")
    return path


def _save_scaler_encoder(fitted_pipeline, vdir: str) -> Dict[str, bool]:
    """Extract and save scaler.pkl / encoder.pkl from the fitted sklearn Pipeline."""
    saved = {"scaler": False, "encoder": False}
    try:
        from sklearn.pipeline import Pipeline
        from imblearn.pipeline import Pipeline as ImbPipeline
        if not isinstance(fitted_pipeline, (Pipeline, ImbPipeline)):
            return saved
        steps = dict(fitted_pipeline.steps)
        if "scaler" in steps:
            path = os.path.join(vdir, "scaler.pkl")
            with open(path, "wb") as f:
                pickle.dump(steps["scaler"], f)
            saved["scaler"] = True
            logger.info(f"Scaler saved: {path}")
        if "encoder" in steps:
            path = os.path.join(vdir, "encoder.pkl")
            with open(path, "wb") as f:
                pickle.dump(steps["encoder"], f)
            saved["encoder"] = True
            logger.info(f"Encoder saved: {path}")
    except Exception as e:
        logger.warning(f"scaler/encoder save failed (non-fatal): {e}")
    return saved


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


# ── Step 4a: predict.py — standalone inference script ────────────────────────

def _save_predict_script(model_name: str, feature_names: List[str],
                          task_type: str, vdir: str, version: str,
                          has_scaler: bool, best_params: Dict[str, Any]):
    feat_list  = repr(feature_names)
    params_str = repr(best_params) if best_params else "{}"
    scale_block = """
# Load scaler (used during training for this model)
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

def _preprocess(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].copy()
    X = X.fillna(X.median(numeric_only=True))
    return scaler.transform(X.values)
""" if has_scaler else """
def _preprocess(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].copy()
    X = X.fillna(X.median(numeric_only=True))
    return X.values
"""
    code = f'''#!/usr/bin/env python3
"""
predict.py — Standalone inference script
Model   : {model_name}
Version : {version}
Task    : {task_type}
Params  : {params_str}

Usage:
    python predict.py --input data.csv [--output predictions.csv]
"""
import os, pickle, argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = {feat_list}
{scale_block}
# Load model
with open(os.path.join(BASE_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)


def predict(input_path: str, output_path: str = None):
    df = pd.read_csv(input_path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {{missing}}")
    X = _preprocess(df)
    preds = model.predict(X)
    df["prediction"] = preds
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {{output_path}}")
    else:
        print(df[[*FEATURES, "prediction"]].to_string(index=False))
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,  help="Path to input CSV")
    parser.add_argument("--output", required=False, help="Path to output CSV (optional)")
    args = parser.parse_args()
    predict(args.input, args.output)
'''
    path = os.path.join(vdir, "predict.py")
    with open(path, "w") as f:
        f.write(code)


# ── Step 4b: train.py — full reproducible training script ────────────────────

def _save_train_script(model_name: str, feature_names: List[str], target_col: str,
                        task_type: str, vdir: str, version: str,
                        best_params: Dict[str, Any], preprocessing_config: Dict[str, Any]):
    feat_list   = repr(feature_names)
    params_repr = repr(best_params) if best_params else "{}"
    is_clf      = task_type == "classification"

    # Build model instantiation block
    model_imports, model_init = _model_code_block(model_name, best_params, task_type)

    dropped_missing  = repr(preprocessing_config.get("dropped_missing",  []))
    dropped_variance = repr(preprocessing_config.get("dropped_variance", []))
    encoded_cols     = repr(preprocessing_config.get("encoded_cols",     []))

    eval_block = """
    # ── Classification metrics ────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  y_pred)
    f1        = f1_score(y_test, y_pred, average="weighted")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}")
    print(f"  F1 Score       : {f1:.4f}")
    print(f"  Overfit Gap    : {train_acc - test_acc:.4f}")
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            roc = roc_auc_score(y_test, y_prob if y_prob.shape[1] > 2 else y_prob[:, 1],
                                multi_class="ovr", average="weighted")
            print(f"  ROC-AUC        : {roc:.4f}")
    except Exception:
        pass
""" if is_clf else """
    # ── Regression metrics ────────────────────────────────────────────────────
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import math
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2  = r2_score(y_test,  y_pred)
    mae      = mean_absolute_error(y_test, y_pred)
    rmse     = math.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  Train R²  : {train_r2:.4f}")
    print(f"  Test  R²  : {test_r2:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  RMSE      : {rmse:.4f}")
    print(f"  Overfit Gap: {train_r2 - test_r2:.4f}")
"""

    code = f'''#!/usr/bin/env python3
"""
train.py — Full reproducible training script
Model   : {model_name}
Version : {version}
Task    : {task_type}
Target  : {target_col}

Usage:
    python train.py --data your_dataset.csv [--target {target_col}]
"""
import os, pickle, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
{model_imports}

RANDOM_STATE   = 42
TEST_SIZE      = 0.2
CV_FOLDS       = 5
FEATURES       = {feat_list}
TARGET         = "{target_col}"
TASK_TYPE      = "{task_type}"
BEST_PARAMS    = {params_repr}
DROP_MISSING   = {dropped_missing}
DROP_VARIANCE  = {dropped_variance}
ENCODE_COLS    = {encoded_cols}


def preprocess(df: pd.DataFrame, target: str):
    df = df.dropna(subset=[target]).reset_index(drop=True)
    X  = df.drop(columns=[target])
    y  = df[target]

    # Drop pre-identified high-missing / zero-variance columns
    drop = [c for c in DROP_MISSING + DROP_VARIANCE if c in X.columns]
    if drop:
        X = X.drop(columns=drop)

    # Impute categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        si = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = si.fit_transform(X[cat_cols])

    # Impute numerics
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols and X[num_cols].isnull().any().any():
        knn = KNNImputer(n_neighbors=5)
        X[num_cols] = knn.fit_transform(X[num_cols])

    # Winsorize outliers (IQR 1.5x)
    for col in num_cols:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            X[col] = X[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    # Encode categoricals
    enc_present = [c for c in ENCODE_COLS if c in X.columns]
    if enc_present:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[enc_present] = enc.fit_transform(X[enc_present]).astype(int)

    # Force numeric
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # Encode target for classification
    if TASK_TYPE == "classification" and y.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y), name=target)

    return X, y


def train(data_path: str, target: str = TARGET):
    print(f"Loading data from {{data_path}} ...")
    df = pd.read_csv(data_path,
                     null_values=["NA", "N/A", "na", "null", "NULL", "", "?"])

    print("Preprocessing ...")
    X, y = preprocess(df, target)

    # Keep only selected features (if present)
    feat_present = [f for f in FEATURES if f in X.columns]
    if feat_present:
        X = X[feat_present]

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y.values if TASK_TYPE == "classification" else None
    )

    print(f"Train: {{len(X_train)}} rows | Test: {{len(X_test)}} rows")

    # ── Build model ───────────────────────────────────────────────────────────
    model = {model_init}
    if BEST_PARAMS:
        model.set_params(**BEST_PARAMS)

    # ── Cross-validation ──────────────────────────────────────────────────────
    print(f"Running {{CV_FOLDS}}-fold cross-validation ...")
    cv = (StratifiedKFold if TASK_TYPE == "classification" else KFold)(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = "accuracy" if TASK_TYPE == "classification" else "r2"
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    print(f"  CV {scoring}: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")

    # ── Final fit ─────────────────────────────────────────────────────────────
    print("Training final model ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
{eval_block}
    # ── Save artifacts ────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(out_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {{model_path}}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True,                help="Path to training CSV")
    parser.add_argument("--target", default=TARGET,              help=f"Target column (default: {{TARGET}})")
    args = parser.parse_args()
    train(args.data, args.target)
'''
    path = os.path.join(vdir, "train.py")
    with open(path, "w") as f:
        f.write(code)


def _model_code_block(model_name: str, params: Dict[str, Any], task_type: str):
    """Return (import_line, instantiation_expr) for the best model."""
    is_clf = task_type == "classification"
    mapping = {
        "logistic_regression": ("from sklearn.linear_model import LogisticRegression",
                                 "LogisticRegression(max_iter=2000, solver='saga', random_state=42)"),
        "linear_regression":   ("from sklearn.linear_model import LinearRegression",
                                 "LinearRegression()"),
        "ridge":               ("from sklearn.linear_model import Ridge",
                                 "Ridge()"),
        "lasso":               ("from sklearn.linear_model import Lasso",
                                 "Lasso(max_iter=3000)"),
        "elastic_net":         ("from sklearn.linear_model import ElasticNet",
                                 "ElasticNet(max_iter=3000)"),
        "decision_tree":       (f"from sklearn.tree import {'DecisionTreeClassifier' if is_clf else 'DecisionTreeRegressor'}",
                                 f"{'DecisionTreeClassifier' if is_clf else 'DecisionTreeRegressor'}(random_state=42)"),
        "random_forest":       (f"from sklearn.ensemble import {'RandomForestClassifier' if is_clf else 'RandomForestRegressor'}",
                                 f"{'RandomForestClassifier' if is_clf else 'RandomForestRegressor'}(n_jobs=-1, random_state=42)"),
        "gradient_boosting":   (f"from sklearn.ensemble import {'GradientBoostingClassifier' if is_clf else 'GradientBoostingRegressor'}",
                                 f"{'GradientBoostingClassifier' if is_clf else 'GradientBoostingRegressor'}(random_state=42)"),
        "svm":                 (f"from sklearn.svm import {'SVC' if is_clf else 'SVR'}",
                                 f"{'SVC(probability=True, max_iter=3000)' if is_clf else 'SVR()'}"),
        "knn":                 (f"from sklearn.neighbors import {'KNeighborsClassifier' if is_clf else 'KNeighborsRegressor'}",
                                 f"{'KNeighborsClassifier' if is_clf else 'KNeighborsRegressor'}(n_jobs=-1)"),
        "xgboost":             (f"from xgboost import {'XGBClassifier' if is_clf else 'XGBRegressor'}",
                                 f"{'XGBClassifier' if is_clf else 'XGBRegressor'}(random_state=42, verbosity=0, n_jobs=1)"),
        "lightgbm":            (f"from lightgbm import {'LGBMClassifier' if is_clf else 'LGBMRegressor'}",
                                 f"{'LGBMClassifier' if is_clf else 'LGBMRegressor'}(random_state=42, verbosity=-1, n_jobs=1)"),
    }
    imp, init = mapping.get(model_name, ("from sklearn.ensemble import RandomForestClassifier",
                                          "RandomForestClassifier(random_state=42)"))
    return imp, init


# ── Step 4c: requirements.txt ─────────────────────────────────────────────────

def _save_requirements(model_name: str, vdir: str):
    base = [
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "imbalanced-learn>=0.11",
        "joblib>=1.3",
    ]
    extras = []
    if model_name == "xgboost":
        extras.append("xgboost>=2.0")
    elif model_name == "lightgbm":
        extras.append("lightgbm>=4.3")
    lines = base + extras + ["", "# API server (optional)", "fastapi>=0.110", "uvicorn>=0.29", "pydantic>=2.0"]
    path = os.path.join(vdir, "requirements.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Step 4d: API export (FastAPI snippet) ─────────────────────────────────────

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
            "scaler.pkl",
            "encoder.pkl",
            "predict.py",
            "train.py",
            "requirements.txt",
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

    preprocessing_config = state.get("preprocessing_config", {})

    # 1. Save model
    model_path = None
    if fitted_model is not None:
        model_path = _save_model(fitted_model, vdir)

    # 1b. Save scaler.pkl / encoder.pkl (extracted from pipeline wrapper)
    scaler_saved = False
    if fitted_model is not None:
        # evaluation_agent stores the raw model; the pipeline wrapper lives in _fitted_pipeline
        fitted_pipeline = state.get("_fitted_pipeline")
        if fitted_pipeline is not None:
            saved = _save_scaler_encoder(fitted_pipeline, vdir)
            scaler_saved = saved["scaler"]

    # 2. Reproducibility config
    _save_reproducibility(state, vdir, dataset_hash)

    # 3. Inference samples
    if fitted_model is not None:
        _save_inference_samples(fitted_model, df, target_col, feature_names, vdir)

    # 4a. predict.py
    try:
        _save_predict_script(best_model_name, feature_names, task_type, vdir, version,
                             scaler_saved, best_params)
    except Exception as e:
        logger.warning(f"predict.py save failed: {e}")

    # 4b. train.py
    try:
        _save_train_script(best_model_name, feature_names, target_col, task_type,
                           vdir, version, best_params, preprocessing_config)
    except Exception as e:
        logger.warning(f"train.py save failed: {e}")

    # 4c. requirements.txt
    try:
        _save_requirements(best_model_name, vdir)
    except Exception as e:
        logger.warning(f"requirements.txt save failed: {e}")

    # 4d. API export
    try:
        _save_api_export(best_model_name, feature_names, task_type, vdir, version, dataset_id)
    except Exception as e:
        logger.warning(f"api_export.py save failed: {e}")

    # 5. Training log CSV
    try:
        _save_training_log(eval_results, task_type, vdir, version, dataset_id, timestamp)
    except Exception as e:
        logger.warning(f"training_log.csv save failed: {e}")

    # 6. Experiment log JSON
    experiment_record = None
    try:
        experiment_record = _save_experiment_log(
            state, eval_results, vdir, version, dataset_id,
            timestamp, training_time_s, dataset_hash
        )
    except Exception as e:
        logger.warning(f"experiment_log.json save failed: {e}")

    # 7. Drift hooks
    _save_drift_hooks(df, feature_names, vdir)

    # 8. Metadata manifest — always written last so it acts as a completion marker
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
