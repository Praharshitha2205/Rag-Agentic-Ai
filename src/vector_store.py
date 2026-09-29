from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from .config import settings


# all-MiniLM-L6-v2 produces 384-dimensional embeddings
EMBEDDING_DIMENSION = 384


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index() -> None:
    pc = get_pinecone_client()
    existing = set(pc.list_indexes().names())

    if settings.index_name not in existing:
        pc.create_index(
            name=settings.index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )


def get_vector_store() -> PineconeVectorStore:
    ensure_index()

    return PineconeVectorStore(
        index_name=settings.index_name,
        embedding=get_embeddings(),
        namespace=settings.namespace,
    )


def ingest_documents(documents) -> int:
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    return len(documents)


def similarity_search_with_scores(query: str, k: int | None = None):
    vector_store = get_vector_store()

    return vector_store.similarity_search_with_score(
        query,
        k=k or settings.top_k,
    )