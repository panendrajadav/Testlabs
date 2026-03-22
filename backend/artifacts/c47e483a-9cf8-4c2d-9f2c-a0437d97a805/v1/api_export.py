"""
Auto-generated FastAPI inference endpoint
Model   : xgboost
Version : v1
Dataset : c47e483a-9cf8-4c2d-9f2c-a0437d97a805
Task    : classification
Generated: 2026-03-22T12:37:33.528215+00:00
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — xgboost")

with open(os.path.join(os.path.dirname(__file__), "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict(data: InputData):
    result = model.predict([list(data.dict().values())])[0]
    return {"prediction": float(result), "model": "xgboost", "version": "v1"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost", "version": "v1"}
