import os
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .model import IcdSuggester
from .rag import RagExplainer

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
app = FastAPI(title="ICD-10 Code Suggester", version="2.0.0")
_model = IcdSuggester.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else IcdSuggester().fit()
_rag = RagExplainer()


class NoteIn(BaseModel):
    note: str = Field(..., min_length=10, max_length=10_000)
    top_k: int = Field(3, ge=1, le=5)


@app.get("/health")
def health():
    return {"status": "ok", "metrics": _model.metrics, "llm_enabled": _rag.llm_enabled}


@app.post("/suggest")
def suggest(body: NoteIn):
    # The raw note is never logged or stored; only the redacted text is returned.
    return _model.suggest(body.note, body.top_k)


@app.post("/suggest/explain")
def suggest_explain(body: NoteIn):
    """Classifier + retrieval cross-check + grounded explanation (RAG)."""
    result = _model.suggest(body.note, body.top_k)
    result["rag"] = _rag.analyse(result["redacted_note"], result["suggestions"])
    return result
