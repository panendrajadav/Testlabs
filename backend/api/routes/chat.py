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

_SYSTEM_PROMPT = """You are a dataset analyst AI. You have a pandas DataFrame called `df` loaded from the user's dataset.

Write Python code that sets:
  - `answer` : a plain-text string with findings FROM the actual data (use real column names, real numbers, real stats)
  - `chart`  : a plotly figure dict (fig.to_dict()) OR None

CRITICAL RULES:
1. Output ONLY raw executable Python. No markdown fences, no prose.
2. Always set BOTH `answer` and `chart`.
3. Use only: pandas (pd), plotly.express (px), plotly.graph_objects (go), numpy (np).
4. Never call fig.show(). Never use matplotlib or seaborn.
5. `answer` must be plain text - NO HTML entities, NO angle brackets. Write "less than 0.7" not "< 0.7".
6. When a chart is produced, set answer to ONE short sentence describing what it shows.
7. Use df.columns to find column names - never hardcode or guess.
8. Only redirect if the question is completely unrelated to data/ML/statistics (e.g. "write me a poem").
   For those only, set: answer = "I can only analyse your dataset." and chart = None.

QUESTION TYPES - handle ALL of these with real data:

- overview / describe / what is this dataset about / summary:
    rows = len(df)
    cols = len(df.columns)
    col_list = ", ".join(df.columns.tolist())
    num_summary = df.describe().to_string()
    answer = "This dataset has " + str(rows) + " rows and " + str(cols) + " columns. Columns: " + col_list + ". Numeric summary: " + num_summary
    chart = None

- feature importance / which features matter / important columns:
    num_df = df.select_dtypes(include='number')
    target = num_df.columns[-1]
    corr = num_df.corr()[target].drop(target).abs().sort_values(ascending=True)
    fig = px.bar(x=corr.values.tolist(), y=corr.index.tolist(), orientation='h', title='Feature Importance (correlation with ' + target + ')')
    chart = fig.to_dict()
    top3 = corr.sort_values(ascending=False).head(3).index.tolist()
    answer = "Top features correlated with " + target + ": " + ", ".join(top3)

- distribution of X: px.histogram(df, x=col, title='Distribution of ' + col). chart = fig.to_dict().
- scatter X vs Y: px.scatter(df, x=col1, y=col2). chart = fig.to_dict().
- correlation / heatmap: go.Heatmap on df.select_dtypes('number').corr(). chart = fig.to_dict().
- missing values / nulls: count nulls per column, bar chart of columns with nulls.
- bar / count / value counts: px.bar or px.pie on value_counts().
- stats / describe: df.describe() as readable string. chart = None.
- top N / highest / lowest: sort and show relevant rows/columns.
- outliers: use IQR on numeric columns, report count and show box plot.
- any other data question: write code to answer it from df directly."""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("user", "DataFrame info:\nColumns: {columns}\nDtypes:\n{schema}\nShape: {shape}\nSample (first 5 rows):\n{sample}\n\nUser question: {question}\n\nPython code only:")
])

def _clean_answer(raw: str) -> str:
    """Decode HTML entities, strip repr artifacts, remove chart-description boilerplate."""
    import html
    # Decode all HTML entities (&amp; &#39; &quot; &#x27; etc.) in one pass
    raw = html.unescape(raw)
    raw = raw.strip()
    # Strip wrapping quotes from str(list(...)) or str(value)
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1].strip()
    # Remove boilerplate chart-description sentences the LLM sometimes adds
    # e.g. "Here is the scatter plot showing..." when a chart is already returned
    boilerplate = (
        r"^here is (the |a |an )?\w[^.]*plot[^.]*\.\s*",
        r"^here is (the |a |an )?\w[^.]*chart[^.]*\.\s*",
        r"^here is (the |a |an )?\w[^.]*graph[^.]*\.\s*",
        r"^the (scatter|bar|line|pie|histogram|heatmap) plot[^.]*\.\s*",
    )
    import re as _re
    for pat in boilerplate:
        raw = _re.sub(pat, "", raw, flags=_re.IGNORECASE).strip()
    return raw or "Done."

@router.post("/chat", response_model=ChatResponse)
async def chat_analyst(request: ChatRequest):
    file_path = os.path.join(UPLOAD_DIR, f"{request.dataset_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")

    df = pl.read_csv(
        file_path,
        null_values=["NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "", "?"],
        infer_schema_length=10000,
        ignore_errors=True,
    ).to_pandas()

    columns   = ", ".join(df.columns.tolist())
    schema    = "\n".join([f"  {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
    sample    = df.head(5).to_string(index=False)
    shape     = f"{df.shape[0]} rows x {df.shape[1]} columns"
    # Append numeric summary to sample so LLM has richer context
    try:
        num_summary = df.describe(include='number').to_string()
        sample = sample + "\n\nNumeric summary:\n" + num_summary
    except Exception:
        pass

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
        if exec_error or raw_code == "":
            # Retry with a direct prose prompt — no code execution
            try:
                retry_prompt = ChatPromptTemplate.from_messages([
                    ("system", (
                        "You are a dataset analyst. Answer the user's question using ONLY the actual data provided below. "
                        "Give concrete numbers, column names, and statistics from the data. "
                        "For feature importance questions, rank columns by their absolute correlation with the last numeric column. "
                        "For overview questions, describe what the dataset is about based on its column names and statistics. "
                        "Plain text only — no HTML entities, no markdown."
                    )),
                    ("user", f"Dataset columns: {columns}\nShape: {shape}\nNumeric summary:\n{sample}\n\nQuestion: {request.question}")
                ])
                retry_resp = llm_invoke_with_retry(llm, retry_prompt.format_messages())
                answer = retry_resp.content.strip()
                chart  = None
            except Exception:
                answer = f"Could not analyse the dataset. Try: show distribution of {df.columns[0]}, or scatter plot of X vs Y."
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
