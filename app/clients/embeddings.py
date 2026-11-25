from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.clients.device import get_device, get_dtype
from app.config import settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    device = get_device(settings.EMBEDDING_DEVICE)
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": device, "model_kwargs": {"torch_dtype": get_dtype(device)}},
        encode_kwargs={"normalize_embeddings": True},
    )
