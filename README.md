# Agentic AI eBook — Grounded RAG Chatbot

A Python-native Retrieval-Augmented Generation (RAG) chatbot built for the AI assignment. It uses **PyPDFLoader + RecursiveCharacterTextSplitter + OpenAI embeddings + Pinecone + LangGraph + OpenAI chat model + FastAPI + Streamlit**.

## What this implementation demonstrates

- PDF ingestion and page-aware chunking
- 1000-character chunks with 200-character overlap
- `text-embedding-3-small` embeddings (1536 dimensions)
- Pinecone cosine-similarity vector search
- LangGraph `retrieve → generate` workflow
- Strict grounding: the LLM is instructed to use only retrieved context
- Early refusal when retrieval confidence is below a threshold
- Transparent answer + retrieved chunks + relevance score
- Both FastAPI and Streamlit interfaces
- Sample benchmark queries, including an out-of-domain refusal test

## 1. Prerequisites

- Python 3.10+
- 8 GB RAM recommended
- OpenAI API key
- Pinecone API key

## 2. Create the environment

### Windows PowerShell

```powershell
cd rag-agentic-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Configure API keys

Copy `.env.example` to `.env` and fill in:

```text
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=agentic-ai-index
PINECONE_NAMESPACE=agentic-ai
```

Do **not** commit `.env` to GitHub.

## 4. Add the source PDF

Put the assignment eBook here:

```text
data/Ebook-Agentic-AI.pdf
```

The repository does not redistribute the source PDF. If network access is available, you can run:

```powershell
python scripts/download_pdf.py
```

Then verify the file exists at `data/Ebook-Agentic-AI.pdf`.

## 5. Create the Pinecone index and ingest the document

Run:

```powershell
python scripts/ingest.py
```

The script creates the Pinecone serverless index if it does not exist, then loads the PDF, splits it, embeds the chunks, and upserts them.

## 6. Run the Streamlit UI

```powershell
streamlit run app.py
```

Open the displayed local URL, normally:

```text
http://localhost:8501
```

## 7. Run the FastAPI backend

In a second terminal:

```powershell
uvicorn src.api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

Example response shape:

```json
{
  "final_answer": "...",
  "retrieved_context": [
    {
      "page": 12,
      "chunk_id": "page_12_chunk_15",
      "score": 0.87,
      "text": "..."
    }
  ],
  "confidence_score": 0.87,
  "grounded": true
}
```

## 8. Benchmark / validation queries

Use the questions in `sample_queries.txt`.

The first five should retrieve relevant eBook passages. The FIFA World Cup question is deliberately outside the source document and should produce:

```text
I cannot answer that based on the provided document.
```

This demonstrates the assignment's strict-grounding requirement.

## 9. Project structure

```text
rag-agentic-ai/
├── app.py
├── architecture.md
├── requirements.txt
├── .env.example
├── .gitignore
├── sample_queries.txt
├── data/
│   ├── README.md
│   └── Ebook-Agentic-AI.pdf       # local source file, not committed by default
├── scripts/
│   ├── download_pdf.py
│   └── ingest.py
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── graph.py
│   ├── ingestion.py
│   └── vector_store.py
└── tests/
    └── test_grounding.py
```

## 10. Interview explanation

### Why RAG?

The chatbot must answer from a specific eBook rather than relying on the model's general knowledge. RAG separates knowledge retrieval from answer generation: the system first finds relevant passages, then gives those passages to the LLM as its evidence.

### Why embeddings?

Embeddings turn text into numerical vectors so semantically similar questions and document chunks can be compared even when they do not use exactly the same words.

### Why Pinecone?

Pinecone stores the embedding vectors and supports fast similarity search. The application retrieves the most relevant chunks for each user question.

### Why LangGraph?

LangGraph makes the RAG workflow explicit as stateful nodes. In this implementation the state moves from `retrieve` to `generate`, carrying the question, retrieved context, answer, and retrieval score.

### How is grounding enforced?

There are two layers:

1. A retrieval threshold prevents generation when there is not enough relevant evidence.
2. The system prompt tells the LLM to use only the retrieved context and to refuse unsupported questions.

### What does confidence mean?

The displayed value is **retrieval confidence**, derived from the highest Pinecone similarity score. It is intentionally not presented as a probability that the generated answer is correct.

## 11. GitHub submission checklist

Before submitting:

- [ ] `.env` is NOT committed.
- [ ] Source PDF is present locally and ingestion succeeds.
- [ ] Pinecone index is populated.
- [ ] Streamlit UI works.
- [ ] FastAPI `/chat` works.
- [ ] Retrieved chunks and page numbers are visible.
- [ ] Confidence/relevance score is visible.
- [ ] 5–6 benchmark queries have been tested.
- [ ] The out-of-domain FIFA question is refused.
- [ ] README contains setup and execution commands.
- [ ] GitHub repository is Public, as requested by the assignment.

## 12. Git commands

```powershell
git init
git add .
git commit -m "Build grounded Agentic AI RAG chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/agentic-ai-rag-chatbot.git
git push -u origin main
```

Never commit API keys.
