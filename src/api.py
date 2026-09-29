from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .graph import build_rag_graph

app = FastAPI(
    title="Agentic AI eBook — Grounded RAG API",
    version="1.0.0",
    description="Strictly grounded RAG chatbot using LangGraph and Pinecone.",
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class RetrievedChunk(BaseModel):
    page: int
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: list[RetrievedChunk]
    confidence_score: float
    grounded: bool


graph = None


def get_graph():
    global graph
    if graph is None:
        graph = build_rag_graph()
    return graph


@app.get("/health")
def health():
    return {"status": "ok", "service": "agentic-ai-rag"}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    try:
        result = get_graph().invoke(
            {
                "question": request.query,
                "context": [],
                "answer": "",
                "confidence_score": 0.0,
            }
        )
        return QueryResponse(
            final_answer=result.get("answer", ""),
            retrieved_context=result.get("context", []),
            confidence_score=float(result.get("confidence_score", 0.0)),
            grounded=bool(result.get("grounded", False)),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
