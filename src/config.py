from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    index_name: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    namespace: str = os.getenv("PINECONE_NAMESPACE", "agentic-ai")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    top_k: int = int(os.getenv("TOP_K", "4"))
    min_retrieval_score: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.25"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    pdf_path: Path = ROOT_DIR / "data" / "Ebook-Agentic-AI.pdf"

    def validate_api_keys(self) -> None:
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise RuntimeError(
                "Missing environment variables: " + ", ".join(missing) + ". "
                "Copy .env.example to .env and fill in the keys."
            )


settings = Settings()
