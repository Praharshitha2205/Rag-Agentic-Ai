from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.ingestion import load_and_chunk_pdf  # noqa: E402
from src.vector_store import ingest_documents  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI eBook into Pinecone.")
    parser.add_argument("--pdf", default=str(settings.pdf_path))
    args = parser.parse_args()

    settings.validate_api_keys()
    documents = load_and_chunk_pdf(args.pdf)
    count = ingest_documents(documents)
    print(f"Ingestion complete: {count} chunks upserted into '{settings.index_name}'.")


if __name__ == "__main__":
    main()
