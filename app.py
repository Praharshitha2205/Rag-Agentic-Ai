from __future__ import annotations

import streamlit as st

from src.config import settings
from src.graph import build_rag_graph

st.set_page_config(
    page_title="Agentic AI eBook RAG",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI eBook — Grounded RAG Chatbot")
st.caption("Answers are generated only from retrieved passages of the supplied eBook.")

with st.sidebar:
    st.header("Retrieval")
    st.write(f"Top-k: **{settings.top_k}**")
    st.write(f"Minimum relevance: **{settings.min_retrieval_score:.2f}**")
    st.write(f"Embedding: **{settings.embedding_model}**")
    st.write(f"LLM: **{settings.chat_model}**")
    st.divider()
    st.markdown("**Grounding behavior**")
    st.write("If relevant context cannot be retrieved, the chatbot refuses to answer instead of using outside knowledge.")

@st.cache_resource(show_spinner=False)
def load_graph():
    return build_rag_graph()

query = st.chat_input("Ask something about Agentic AI...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving evidence and generating a grounded answer..."):
                result = load_graph().invoke(
                    {
                        "question": query,
                        "context": [],
                        "answer": "",
                        "confidence_score": 0.0,
                    }
                )

            confidence = float(result.get("confidence_score", 0.0))
            grounded = bool(result.get("grounded", False))

            st.markdown(result.get("answer", ""))
            st.metric("Retrieval confidence", f"{confidence:.3f}")

            if grounded:
                st.success("Answer generated from retrieved document context.")
            else:
                st.warning("The document did not provide enough relevant context to answer this question.")

            st.subheader("Retrieved context")
            for i, item in enumerate(result.get("context", []), start=1):
                with st.expander(
                    f"Chunk {i} · Page {item['page']} · relevance {item['score']:.3f}"
                ):
                    st.caption(item["chunk_id"])
                    st.write(item["text"])
        except Exception as exc:
            st.error(str(exc))
            st.info("Check that .env is configured and the Pinecone index has been ingested.")
