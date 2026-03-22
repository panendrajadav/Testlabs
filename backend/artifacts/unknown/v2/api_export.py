"""
Auto-generated FastAPI inference endpoint
Model   : random_forest
Version : v2
Dataset : unknown
Task    : classification
Generated: 2026-03-22T04:01:17.021190+00:00
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — random_forest")

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
    return {"prediction": float(result), "model": "random_forest", "version": "v2"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "random_forest", "version": "v2"}
