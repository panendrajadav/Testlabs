import os
import re
import json
import polars as pl
import pandas as pd
import numpy as np
import traceback
from fastapi import APIRouter, HTTPException
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from api.schemas import ChatRequest, ChatResponse
from utils.helpers import llm_invoke_with_retry
from utils.logger import logger

router = APIRouter()
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads")

# ── Code-request detection ────────────────────────────────────────────────────

_CODE_KEYWORDS = (
    "write code", "give me code", "show code", "write the code",
    "linear regression code", "regression code", "write linear",
    "code for", "python code", "write a model", "train a model",
)

_VIZ_KEYWORDS = (
    "plot", "chart", "graph", "visuali", "histogram", "scatter",
    "heatmap", "bar chart", "pie chart", "distribution", "show me",
    "display", "draw",
)

def _is_code_request(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _CODE_KEYWORDS)

def _is_viz_request(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _VIZ_KEYWORDS)


def _run_regression_and_build_answer(df: pd.DataFrame):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    num_df = df.select_dtypes(include="number").dropna()
    tgt = num_df.columns[-1]
    feat_cols = num_df.drop(columns=[tgt]).columns.tolist()

    X = num_df[feat_cols].values
    y = num_df[tgt].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    score = round(r2_score(y_test, model.predict(X_test)), 4)

    feat_repr = repr(feat_cols)
    code = "\n".join([
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.linear_model import LinearRegression",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error",
        "",
        'df = pd.read_csv("your_dataset.csv")',
        "",
        f"X = df[{feat_repr}]",
        f'y = df["{tgt}"]',
        "",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=42",
        ")",
        "",
        "model = LinearRegression()",
        "model.fit(X_train, y_train)",
        "",
        "y_pred = model.predict(X_test)",
        "",
        'print("R2 Score :", round(r2_score(y_test, y_pred), 4))',
        'print("MAE      :", round(mean_absolute_error(y_test, y_pred), 2))',
        'print("RMSE     :", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))',
        'print("Intercept:", round(model.intercept_, 4))',
        'print("Coefficients:")',
        f"for col, coef in zip({feat_repr}, model.coef_):",
        '    print(f"  {col}: {round(coef, 4)}")',
    ])

    answer = (
        f"Linear Regression on {len(X_train)} rows predicting {tgt}. "
        f"Test R2: {score}\n\n"
        f"```python\n{code}\n```"
    )
    return answer


# ── LLM setup ─────────────────────────────────────────────────────────────────

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
3. Use only: pandas (pd), plotly.express (px), plotly.graph_objects (go), numpy (np), sklearn (scikit-learn).
4. Never call fig.show(). Never use matplotlib or seaborn.
5. `answer` must be plain text - NO HTML entities, NO angle brackets. Write "less than 0.7" not "< 0.7".
6. When a chart is produced, set answer to ONE short sentence describing what it shows.
7. Use df.columns to find column names - never hardcode or guess.
8. NEVER refuse ML, statistics, regression, classification, or coding questions.
   Only set answer = "I can only analyse your dataset." and chart = None for questions with
   ZERO connection to data, ML, statistics, or coding (e.g. "write me a poem").

QUESTION TYPES - handle ALL of these with real data:

- overview / describe / summary:
    rows = len(df)
    cols = len(df.columns)
    col_list = ", ".join(df.columns.tolist())
    num_summary = df.describe().to_string()
    answer = "This dataset has " + str(rows) + " rows and " + str(cols) + " columns. Columns: " + col_list + ". Numeric summary: " + num_summary
    chart = None

- feature importance / which features matter:
    num_df = df.select_dtypes(include='number')
    tgt = num_df.columns[-1]
    corr = num_df.corr()[tgt].drop(tgt).abs().sort_values(ascending=True)
    fig = px.bar(x=corr.values.tolist(), y=corr.index.tolist(), orientation='h', title='Feature Importance (correlation with ' + tgt + ')')
    chart = fig.to_dict()
    top3 = corr.sort_values(ascending=False).head(3).index.tolist()
    answer = "Top features correlated with " + tgt + ": " + ", ".join(top3)

- distribution of X: px.histogram(df, x=col). chart = fig.to_dict().
- scatter X vs Y: px.scatter(df, x=col1, y=col2). chart = fig.to_dict().
- correlation / heatmap: go.Heatmap on df.select_dtypes('number').corr(). chart = fig.to_dict().
- missing values / nulls: count nulls per column, bar chart of columns with nulls.
- bar / count / value counts: px.bar or px.pie on value_counts().
- stats / describe: df.describe() as readable string. chart = None.
- top N / highest / lowest: sort and show relevant rows/columns.
- outliers: use IQR on numeric columns, report count and show box plot.
- any other data question: write code to answer it from df directly."""


def _build_messages(columns: str, schema: str, shape: str, sample: str, question: str):
    user_text = (
        f"DataFrame info:\nColumns: {columns}\nDtypes:\n{schema}\n"
        f"Shape: {shape}\nSample (first 5 rows):\n{sample}\n\n"
        f"User question: {question}\n\nPython code only:"
    )
    return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_text)]


def _clean_answer(raw: str) -> str:
    import html
    import re as _re
    raw = html.unescape(raw).strip()
    if "```" in raw:
        return raw
    if not any(raw.lstrip().startswith(kw) for kw in ("import ", "from ", "def ", "#")):
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            raw = raw[1:-1].strip()
        for pat in (
            r"^here is (the |a |an )?\w[^.]*plot[^.]*\.\s*",
            r"^here is (the |a |an )?\w[^.]*chart[^.]*\.\s*",
            r"^the (scatter|bar|line|pie|histogram|heatmap) plot[^.]*\.\s*",
        ):
            raw = _re.sub(pat, "", raw, flags=_re.IGNORECASE).strip()
    return raw or "Done."


# ── Route ─────────────────────────────────────────────────────────────────────

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

    columns = ", ".join(df.columns.tolist())
    schema  = "\n".join([f"  {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
    sample  = df.head(5).to_string(index=False)
    shape   = f"{df.shape[0]} rows x {df.shape[1]} columns"
    try:
        sample += "\n\nNumeric summary:\n" + df.describe(include="number").to_string()
    except Exception:
        pass

    # Short-circuit for code requests — no chart, just code block
    if _is_code_request(request.question):
        try:
            answer = _run_regression_and_build_answer(df)
            return ChatResponse(answer=answer, chart=None, code=None)
        except Exception as e:
            logger.error(f"Regression shortcut failed: {e}\n{traceback.format_exc()}")

    llm = _create_llm()
    response = llm_invoke_with_retry(llm, _build_messages(
        columns=columns, schema=schema, sample=sample, shape=shape, question=request.question
    ))

    raw = response.content.strip()
    logger.info(f"LLM raw response:\n{raw}")

    fence = re.search(r"```(?:python)?\n([\s\S]*?)```", raw)
    if fence:
        raw_code = fence.group(1).strip()
    elif any(kw in raw for kw in ["df[", "df.", "px.", "go.", "pd.", "np.", "answer =", "answer=", "chart =", "chart="]):
        raw_code = raw
    else:
        raw_code = ""

    logger.info(f"Executing code:\n{raw_code}")

    import plotly.graph_objects as go
    import plotly.express as px

    local_vars = {
        "df": df, "pd": pd, "px": px, "go": go, "np": np,
        "answer": "", "chart": None,
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

    if not answer:
        if exec_error or raw_code == "":
            try:
                retry_messages = [
                    SystemMessage(content=(
                        "You are a dataset analyst. Answer the user's question using ONLY the actual data provided. "
                        "Give concrete numbers, column names, and statistics. Plain text only — no HTML, no markdown."
                    )),
                    HumanMessage(content=f"Dataset columns: {columns}\nShape: {shape}\nSample:\n{sample}\n\nQuestion: {request.question}")
                ]
                retry_resp = llm_invoke_with_retry(llm, retry_messages)
                answer = retry_resp.content.strip()
                chart  = None
            except Exception:
                answer = f"Could not analyse the dataset. Try: show distribution of {df.columns[0]}, or scatter plot of X vs Y."
        else:
            answer = "Analysis complete."

    answer = _clean_answer(answer)

    # Strip chart if the question is not a visualization request
    if not _is_viz_request(request.question):
        chart = None

    if chart is not None:
        if hasattr(chart, "to_dict"):
            chart = chart.to_dict()
        chart = json.loads(json.dumps(chart, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))

    return ChatResponse(answer=answer, chart=chart, code=raw_code or None)
