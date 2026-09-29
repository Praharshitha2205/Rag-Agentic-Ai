# Agentic AI eBook — Grounded RAG Chatbot

A Python-based **Retrieval-Augmented Generation (RAG) chatbot** that answers questions from an Agentic AI eBook using semantic search and document-grounded generation.

The system retrieves relevant passages from the eBook before generating an answer, helping prevent unsupported responses.

## What This Project Demonstrates

* PDF document ingestion and page-aware chunking
* Semantic search using vector embeddings
* Local Hugging Face embeddings with `all-MiniLM-L6-v2`
* Pinecone vector database for similarity search
* LangGraph `retrieve → generate` workflow
* Local LLM generation using `Qwen2.5-0.5B-Instruct`
* Retrieval confidence threshold for unsupported questions
* Strict document grounding
* Streamlit chatbot interface
* FastAPI backend
* Retrieved context, page numbers, and relevance scores
* Out-of-domain question handling

##  Architecture

```text
User Question
      ↓
Query Embedding
      ↓
Pinecone Similarity Search
      ↓
Relevant Document Chunks
      ↓
Retrieval Confidence Check
      ↓
Qwen2.5-0.5B-Instruct
      ↓
Grounded Answer
```

##  Tech Stack

* **Python**
* **LangGraph**
* **LangChain**
* **Hugging Face Transformers**
* **Sentence Transformers**
* **Pinecone**
* **FastAPI**
* **Streamlit**
* **PyPDF**
* **Git/GitHub**

## 📁 Project Structure

```text
rag-agentic-ai/
├── app.py
├── architecture.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── sample_queries.txt
│
├── data/
│   ├── README.md
│   └── Ebook-Agentic-AI.pdf
│
├── scripts/
│   ├── download_pdf.py
│   └── ingest.py
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── graph.py
│   ├── ingestion.py
│   └── vector_store.py
│
└── tests/
    └── test_grounding.py
```

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Praharshitha2205/Rag-Agentic-Ai.git
cd Rag-Agentic-Ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

Add your Pinecone configuration:

```text
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=agentic-ai-index
PINECONE_NAMESPACE=agentic-ai
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

**Never commit `.env` or API keys to GitHub.**

## 📄 Document Ingestion

Download or place the source eBook inside:

```text
data/Ebook-Agentic-AI.pdf
```

Then run:

```bash
python scripts/ingest.py
```

The ingestion pipeline:

1. Loads the PDF
2. Splits the document into chunks
3. Generates local embeddings
4. Stores the vectors in Pinecone
5. Preserves page and chunk metadata

##  Run the Streamlit Application

```bash
streamlit run app.py
```

Then open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

## 🔌 Run the FastAPI Backend

```bash
uvicorn src.api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

##  How RAG Works in This Project

The chatbot does not directly generate an answer from the user's question.

Instead:

**1. Retrieve**

The user's question is converted into an embedding and compared against document embeddings stored in Pinecone.

**2. Validate**

The retrieved similarity score is checked against a minimum confidence threshold.

**3. Generate**

The most relevant document chunks are provided to the local Qwen model as context.

**4. Ground**

The model is instructed to answer using only the retrieved document context.

If sufficient evidence is not found, the system returns:

```text
I cannot answer that based on the provided document.
```

##  Grounding & Hallucination Control

The project uses two main mechanisms:

* A retrieval similarity threshold prevents generation when relevant evidence is insufficient.
* The generation prompt instructs the model to use only retrieved document context.

This makes the chatbot focused on the knowledge contained in the source document.

##  Sample Queries

Example questions are available in:

```text
sample_queries.txt
```

The test set includes both relevant questions and an out-of-domain question to verify that the system refuses unsupported answers.

## 🎯 Key Learning Outcomes

Through this project, I worked with:

* Retrieval-Augmented Generation
* Vector databases
* Semantic embeddings
* LangGraph workflows
* Document ingestion pipelines
* FastAPI
* Streamlit
* Local LLM inference
* Grounded AI responses
* Environment and API-key management

## 👩‍💻 Author

**Praharshitha**

GitHub: [@Praharshitha2205](https://github.com/Praharshitha2205)
