from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class PipelineRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None

class ChatRequest(BaseModel):
    dataset_id: str
    question: str

class PipelineStatusResponse(BaseModel):
    dataset_id: str
    status: str
    progress: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    chart: Optional[Dict[str, Any]] = None
    code: Optional[str] = None
