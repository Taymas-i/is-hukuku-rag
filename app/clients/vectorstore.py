from functools import lru_cache

from langchain_chroma import Chroma

from app.clients.embeddings import get_embeddings
from app.config import settings


@lru_cache
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
