from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.clients.device import get_device, get_dtype
from app.config import settings


class Reranker:
    def __init__(self, model_name: str):
        device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=get_dtype(device))
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def score(self, query: str, passages: list[str], batch_size: int = 8) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            enc = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(self.model.device)
            logits = self.model(**enc).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).tolist())
        return scores


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(settings.RERANKER_MODEL)
