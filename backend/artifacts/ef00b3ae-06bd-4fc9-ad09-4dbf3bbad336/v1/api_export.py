"""
Auto-generated FastAPI inference endpoint
Model   : svm
Version : v1
Dataset : ef00b3ae-06bd-4fc9-ad09-4dbf3bbad336
Task    : classification
Generated: 2026-03-22T14:54:30.865731+00:00
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — svm")

with open(os.path.join(os.path.dirname(__file__), "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    Id: float

@app.post("/predict")
def predict(data: InputData):
    result = model.predict([list(data.dict().values())])[0]
    return {"prediction": float(result), "model": "svm", "version": "v1"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "svm", "version": "v1"}
