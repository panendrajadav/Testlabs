"""
Auto-generated FastAPI inference endpoint
Model   : logistic_regression
Version : v1
Dataset : unknown
Task    : classification
Generated: 2026-03-22T03:42:06.779507+00:00
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — logistic_regression")

with open(os.path.join(os.path.dirname(__file__), "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    id: float
    age: float
    sex: float
    dataset: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalch: float
    exang: float
    oldpeak: float
    slope: float

@app.post("/predict")
def predict(data: InputData):
    result = model.predict([list(data.dict().values())])[0]
    return {"prediction": float(result), "model": "logistic_regression", "version": "v1"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "logistic_regression", "version": "v1"}
