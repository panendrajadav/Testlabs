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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/health")
def health():
    return {"status": "ok", "service": "TestLabs AutoML"}
