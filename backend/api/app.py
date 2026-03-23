import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from api.router import router

load_dotenv()

app = FastAPI(
    title="TestLabs AutoML API",
    description="Automated ML pipeline with LangGraph agents, EDA plots, SHAP, ROC, and dataset chat analyst",
    version="1.0.0"
)

_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_extra = os.getenv("ALLOWED_ORIGINS", "")
if _extra:
    _origins += [o.strip() for o in _extra.split(",") if o.strip()]

# If ALLOW_ALL_ORIGINS is set (e.g. for debugging), use wildcard
_allow_all = os.getenv("ALLOW_ALL_ORIGINS", "").lower() in ("1", "true", "yes")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _origins,
    allow_credentials=False if _allow_all else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health():
    return {"status": "ok", "service": "TestLabs AutoML"}
