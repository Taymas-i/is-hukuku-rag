import torch

from app.config import settings


def get_device(preferred: str | None = None) -> str:
    device = preferred or settings.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def get_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32
