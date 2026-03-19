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
        temperature=0.1
    )

_SYSTEM_PROMPT = """You are an expert data analyst AI. You have a pandas DataFrame called `df`.

Your job is to answer the user's question by writing Python code that sets two variables:
  - `answer` : a clean, human-friendly string (NO raw Python repr, NO HTML entities, NO angle brackets)
  - `chart`  : a plotly figure dict (fig.to_dict()) OR None

RULES — follow exactly:
1. Output ONLY raw executable Python. No markdown fences, no prose outside code.
2. Always set BOTH `answer` and `chart`.
3. Use only: pandas (as pd), plotly.express (as px), plotly.graph_objects (as go), numpy (as np).
4. Never call fig.show(). Never use matplotlib or seaborn.
5. For `answer`, build a plain English string. Example:
     answer = f"The dataset has {{len(df)}} rows and {{len(df.columns)}} columns."
   NOT: answer = str(df.columns.tolist())   ← this produces ugly repr
6. If the user asks to compare/plot specific columns, use exactly those column names from df.columns.
7. If a visualization is NOT possible (e.g. text data, single value, non-numeric), set chart = None
   and set answer to explain clearly why it cannot be visualised and what you found instead.
8. For correlation/comparison questions, always try to produce a chart.
9. Make charts visually rich: use color_discrete_sequence or color_continuous_scale where appropriate.
10. Column names: use df.columns to find the right column — do NOT guess or hardcode names.

QUESTION TYPES:
- "what is this data about" / "describe" → summarise columns, dtypes, shape, sample values in answer. chart = None.
- "show distribution" → histogram with px.histogram. chart = fig.to_dict().
- "compare X and Y" / "relationship between X and Y" → scatter plot px.scatter. chart = fig.to_dict().
- "correlation" → heatmap using go.Heatmap on df.select_dtypes(include='number').corr(). chart = fig.to_dict().
- "bar chart" / "count" → px.bar or px.pie. chart = fig.to_dict().
- "summary statistics" → df.describe() formatted as readable string. chart = None.
- anything else → answer the question analytically in plain English."""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("user", "DataFrame info:\nColumns: {columns}\nDtypes:\n{schema}\nShape: {shape}\nSample (first 3 rows):\n{sample}\n\nUser question: {question}\n\nPython code only:")
])

def _clean_answer(raw: str) -> str:
    """Strip Python repr artifacts and HTML entities from answer strings."""
    # HTML entities
    raw = raw.replace("&#39;", "'").replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Strip wrapping quotes that come from str(list(...))
    raw = raw.strip()
    if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
        raw = raw[1:-1]
    return raw.strip()

@router.post("/chat", response_model=ChatResponse)
async def chat_analyst(request: ChatRequest):
    file_path = os.path.join(UPLOAD_DIR, f"{request.dataset_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")

    df = pl.read_csv(file_path).to_pandas()

    columns   = ", ".join(df.columns.tolist())
    schema    = "\n".join([f"  {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
    sample    = df.head(3).to_string(index=False)
    shape     = f"{df.shape[0]} rows x {df.shape[1]} columns"

    llm = _create_llm()
    response = llm_invoke_with_retry(llm, CHAT_PROMPT.format_messages(
        columns=columns,
        schema=schema,
        sample=sample,
        shape=shape,
        question=request.question
    ))

    raw = response.content.strip()
    logger.info(f"LLM raw response:\n{raw}")

    # Extract code block if LLM wrapped in fences
    fence = re.search(r"```(?:python)?\n([\s\S]*?)```", raw)
    if fence:
        raw_code = fence.group(1).strip()
    elif any(kw in raw for kw in ["df[", "df.", "px.", "go.", "pd.", "np.", "answer =", "answer=", "chart =", "chart="]):
        raw_code = raw
    else:
        raw_code = ""

    logger.info(f"Executing code:\n{raw_code}")

    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px

    local_vars = {
        "df": df, "pd": pd, "px": px, "go": go, "np": np,
        "answer": "", "chart": None
    }

    exec_error = None
    if raw_code:
        try:
            exec(raw_code, {"__builtins__": __builtins__}, local_vars)
        except Exception as e:
            exec_error = str(e)
            logger.error(f"Code execution error: {e}\n{traceback.format_exc()}")

    answer = str(local_vars.get("answer") or "").strip()
    chart  = local_vars.get("chart")

    # Fallback answers
    if not answer:
        if exec_error:
            # Retry with a simpler direct LLM answer (no code)
            try:
                retry_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a data analyst. Answer the question in plain English based on the dataset info provided. Be concise and helpful."),
                    ("user", f"Dataset columns: {columns}\nShape: {shape}\nSample:\n{sample}\n\nQuestion: {request.question}")
                ])
                retry_resp = llm_invoke_with_retry(llm, retry_prompt.format_messages())
                answer = retry_resp.content.strip()
                chart  = None
            except Exception:
                answer = f"I had trouble analysing that. The dataset has columns: {columns}. Please try rephrasing your question."
        elif raw_code == "":
            # LLM returned prose directly — use it
            answer = raw
        else:
            answer = "Analysis complete."

    answer = _clean_answer(answer)

    # Ensure chart is a plain dict (not a Plotly object)
    if chart is not None and hasattr(chart, "to_dict"):
        chart = chart.to_dict()

    return ChatResponse(
        answer=answer,
        chart=chart,
        code=raw_code or None
    )
