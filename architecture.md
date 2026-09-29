# Architecture

```text
                 ┌──────────────────────────────┐
                 │      Agentic AI eBook PDF    │
                 └──────────────┬───────────────┘
                                │
                         PyPDFLoader
                                │
                     RecursiveTextSplitter
                         1000 / 200
                                │
                     OpenAI Embeddings
                   text-embedding-3-small
                                │
                                ▼
                    ┌────────────────────┐
                    │      Pinecone      │
                    │  cosine similarity │
                    └─────────┬──────────┘
                              │
                         top-k retrieval
                              │
User ── POST /chat ──► LangGraph StateGraph
                              │
                       ┌──────▼──────┐
                       │   retrieve   │
                       └──────┬──────┘
                              │ context + scores
                       ┌──────▼──────┐
                       │   generate   │
                       │ strict LLM  │
                       └──────┬──────┘
                              │
                    answer + chunks + score
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
              FastAPI                 Streamlit
```

## Design decisions

1. **Page-aware chunks:** page numbers are retained in metadata so reviewers can trace answers to the source.
2. **Retrieval score:** the displayed confidence is the top Pinecone cosine similarity, not an invented LLM confidence probability.
3. **Strict refusal:** when retrieval is empty or below the configured relevance threshold, the system refuses before calling the LLM.
4. **Temperature 0:** generation is deterministic and constrained by the system prompt.
5. **No outside knowledge:** the LLM prompt explicitly prohibits using model memory or internet knowledge.
6. **Two interfaces:** FastAPI satisfies the API requirement; Streamlit gives reviewers a simple visual demo.
