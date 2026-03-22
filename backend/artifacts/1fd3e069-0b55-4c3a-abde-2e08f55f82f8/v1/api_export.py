"""
Auto-generated FastAPI inference endpoint
Model   : xgboost
Version : v1
Dataset : 1fd3e069-0b55-4c3a-abde-2e08f55f82f8
Task    : classification
Generated: 2026-03-22T13:29:55.603714+00:00
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — xgboost")

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
    return {"prediction": float(result), "model": "xgboost", "version": "v1"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost", "version": "v1"}
