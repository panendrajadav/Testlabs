"""
Auto-generated FastAPI inference endpoint
Model   : decision_tree
Version : v1
Dataset : e71f473b-af50-4d30-b2c8-a98051acc179
Task    : classification
Generated: 2026-03-22T14:27:04.596359+00:00
"""
import pickle, os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TestLabs Model API — decision_tree")

with open(os.path.join(os.path.dirname(__file__), "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    OverallQual: float
    YearBuilt: float
    YearRemodAdd: float
    ExterQual: float
    Foundation: float
    BsmtQual: float
    BsmtUnfSF: float
    TotalBsmtSF: float
    HeatingQC: float
    KitchenQual: float
    GarageYrBlt: float
    GarageFinish: float
    GarageCars: float
    GarageArea: float
    SaleCondition: float
    SalePrice: float

@app.post("/predict")
def predict(data: InputData):
    result = model.predict([list(data.dict().values())])[0]
    return {"prediction": float(result), "model": "decision_tree", "version": "v1"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "decision_tree", "version": "v1"}
