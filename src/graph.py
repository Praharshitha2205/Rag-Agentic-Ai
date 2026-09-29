from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from transformers import pipeline

from .config import settings
from .vector_store import get_vector_store


class AgentState(TypedDict, total=False):
    question: str
    context: list[dict]
    answer: str
    confidence_score: float
    grounded: bool


SYSTEM_PROMPT = """
You are a strict document-grounded assistant.
Answer the user's question using ONLY the supplied context from the Agentic AI eBook.
Do not use outside knowledge.

If the context does not contain enough information to answer, respond exactly:
I cannot answer that based on the provided document.

Keep the answer concise and directly answer the question.
"""


# Free local language model.
# It runs on your computer and does not use OpenAI API credits.
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
)

def _retrieve(state: AgentState) -> AgentState:
    store = get_vector_store()

    results = store.similarity_search_with_score(
        state["question"],
        k=settings.top_k,
    )

    context = []
    scores = []

    for doc, score in results:
        score = float(score)
        scores.append(score)

        context.append(
            {
                "text": doc.page_content,
                "page": int(doc.metadata.get("page", 0)),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "score": round(score, 4),
            }
        )

    confidence = max(scores) if scores else 0.0

    return {
        "context": context,
        "confidence_score": round(confidence, 4),
    }


def _generate(state: AgentState) -> AgentState:
    context = state.get("context", [])
    confidence = state.get("confidence_score", 0.0)

    if not context or confidence < settings.min_retrieval_score:
        return {
            "answer": "I cannot answer that based on the provided document.",
            "grounded": False,
        }

    formatted_context = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in context
    )

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{formatted_context}

Question:
{state["question"]}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False,
    )

    answer = result[0]["generated_text"].strip()

    return {
        "answer": answer,
        "grounded": True,
    }


def build_rag_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()