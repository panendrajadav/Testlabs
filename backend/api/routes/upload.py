import uuid
import os
import polars as pl
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset and return a dataset_id."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dataset_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    # Quick preview
    df = pl.read_csv(file_path)
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": df.height,
        "columns": df.width,
        "column_names": df.columns,
        "preview": df.head(5).to_dicts()
    }
