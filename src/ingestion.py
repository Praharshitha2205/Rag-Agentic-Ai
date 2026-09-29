from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


def load_and_chunk_pdf(pdf_path: str | Path | None = None) -> list[Document]:
    path = Path(pdf_path or settings.pdf_path)
    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found at {path}. Place Ebook-Agentic-AI.pdf in data/ first."
        )

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):
        page_number = int(chunk.metadata.get("page", 0)) + 1
        chunk.metadata.update(
            {
                "source": path.name,
                "page": page_number,
                "chunk_id": f"page_{page_number}_chunk_{index}",
            }
        )
        chunk.page_content = " ".join(chunk.page_content.split())

    return [chunk for chunk in chunks if chunk.page_content.strip()]
