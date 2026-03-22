"""
Auto-generated FastAPI inference endpoint
Model   : knn
Version : v4
Dataset : unknown
Task    : regression
Generated: 2026-03-22T05:39:45.557312+00:00
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — knn")

with open(os.path.join(os.path.dirname(__file__), "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    : float
    YearsExperience: float

@app.post("/predict")
def predict(data: InputData):
    result = model.predict([list(data.dict().values())])[0]
    return {"prediction": float(result), "model": "knn", "version": "v4"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "knn", "version": "v4"}
