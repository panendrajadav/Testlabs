"""
Auto-generated FastAPI inference endpoint
Model   : random_forest
Version : v3
Dataset : unknown
Task    : regression
Generated: 2026-03-22T05:30:13.692786+00:00
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — random_forest")

with open(os.path.join(os.path.dirname(__file__), "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    sp500_low: float
    sp500_close: float
    sp500_volume: float
    nasdaq_volume: float
    eur_usd: float
    silver_high-low: float
    platinum_open: float
    platinum_high: float
    platinum_low: float
    platinum_close: float
    platinum_high-low: float
    palladium_volume: float

@app.post("/predict")
def predict(data: InputData):
    result = model.predict([list(data.dict().values())])[0]
    return {"prediction": float(result), "model": "random_forest", "version": "v3"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "random_forest", "version": "v3"}
