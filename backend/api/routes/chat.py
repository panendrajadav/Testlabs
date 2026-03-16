import os
import re
import polars as pl
import pandas as pd
import traceback
from fastapi import APIRouter, HTTPException
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from api.schemas import ChatRequest, ChatResponse
from utils.helpers import llm_invoke_with_retry
from utils.logger import logger

router = APIRouter()
UPLOAD_DIR = "uploads"

def _create_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-05-01-preview",
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        temperature=0.2
    )

# ── Prompt: smart analyst that decides whether to answer in prose or code ──
_SYSTEM_PROMPT = """You are an expert data analyst assistant. You have access to a pandas DataFrame called `df`.

Decide how to respond based on the question:

CASE 1 - Simple factual question (columns, shape, dtypes, value counts, describe, head):
  Write Python code that sets answer to a clean human-readable string and chart = None.
  Example: answer = "Columns: " + str(list(df.columns))

CASE 2 - Statistical analysis (mean, correlation, distribution, outliers, summary):
  Write Python code that computes the result and sets answer to a readable string.
  Optionally create a chart: fig = px.something(...) then chart = fig.to_dict()
  If no chart needed: chart = None

CASE 3 - Visualization request (plot, chart, graph, visualize, show):
  Write Python code that creates a plotly chart.
  Set answer to a 1-sentence description.
  fig = px.something(...) then chart = fig.to_dict()

RULES:
- Output ONLY raw executable Python code, no prose, no markdown fences
- Always set both answer (string) and chart (dict or None)
- Use only pandas and plotly.express imported as px
- Do NOT call fig.show()
- Do NOT use matplotlib or seaborn
- Make answer strings human-friendly, not raw Python repr"""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("user", "DataFrame info:\nColumns & dtypes:\n{schema}\n\nShape: {shape}\n\nSample (first 3 rows):\n{sample}\n\nQuestion: {question}\n\nPython code only:")
])

@router.post("/chat", response_model=ChatResponse)
async def chat_analyst(request: ChatRequest):
    file_path = os.path.join(UPLOAD_DIR, f"{request.dataset_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")

    df = pl.read_csv(file_path).to_pandas()
    schema = "\n".join([f"  {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
    sample = df.head(3).to_string(index=False)
    shape  = f"{df.shape[0]} rows × {df.shape[1]} columns"

    llm = _create_llm()
    response = llm_invoke_with_retry(llm, CHAT_PROMPT.format_messages(
        schema=schema,
        sample=sample,
        shape=shape,
        question=request.question
    ))

    raw = response.content.strip()

    # Extract code from fenced block if present
    fence_match = re.search(r"```(?:python)?\n([\s\S]*?)```", raw)
    if fence_match:
        raw_code = fence_match.group(1).strip()
    elif any(kw in raw for kw in ["import ", "df[", "df.", "px.", "pd.", "answer =", "answer=", "chart =", "chart="]):
        raw_code = raw
    else:
        raw_code = ""

    logger.info(f"Chat code:\n{raw_code}")

    import plotly.express as px
    local_vars = {"df": df, "pd": pd, "px": px, "answer": "", "chart": None}

    exec_error = None
    if raw_code:
        try:
            exec(raw_code, {"__builtins__": __builtins__}, local_vars)
        except Exception as e:
            exec_error = str(e)
            logger.error(f"Code execution failed: {e}\n{traceback.format_exc()}")

    answer = str(local_vars.get("answer") or "").strip()

    # Fallbacks
    if not answer:
        if exec_error:
            # Try to answer directly without code for simple questions
            answer = f"I encountered an error analyzing that: {exec_error}. Please try rephrasing."
        elif raw_code == "":
            answer = raw  # LLM returned prose — use it directly
        else:
            answer = "Analysis complete."

    # Clean up Python repr artifacts in answer
    answer = answer.strip("'\"")

    return ChatResponse(
        answer=answer,
        chart=local_vars.get("chart"),
        code=raw_code or None
    )
