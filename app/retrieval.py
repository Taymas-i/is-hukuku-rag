import json
import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.clients.reranker import get_reranker
from app.clients.vectorstore import get_vectorstore
from app.config import settings

Scored = list[tuple[Document, float]]


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str) -> list[str]:
    return [w[:5] for w in re.findall(r"\w+", turkish_lower(text))]


@lru_cache
def load_corpus() -> list[Document]:
    docs = []
    with open(settings.CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(page_content=row["text"], metadata=row["metadata"]))
    return docs


@lru_cache
def get_bm25() -> BM25Okapi:
    return BM25Okapi([tokenize(d.page_content) for d in load_corpus()])


def search_bm25(query: str, k: int) -> Scored:
    docs = load_corpus()
    scores = get_bm25().get_scores(tokenize(query))
    top = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], float(scores[i])) for i in top]


def search_dense(query: str, k: int) -> Scored:
    return get_vectorstore().similarity_search_with_relevance_scores(query, k=k)


def search_hybrid(query: str, k: int, rrf_k: int = 60) -> Scored:
    n = settings.CANDIDATES
    fused: dict[str, float] = {}
    by_id: dict[str, Document] = {}
    for results in (search_bm25(query, n), search_dense(query, n)):
        for rank, (doc, _) in enumerate(results):
            doc_id = doc.metadata["id"]
            by_id[doc_id] = doc
            fused[doc_id] = fused.get(doc_id, 0.0) + 1 / (rrf_k + rank + 1)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(by_id[d], fused[d]) for d in top]


def search_reranked(query: str, k: int) -> Scored:
    candidates = [doc for doc, _ in search_hybrid(query, settings.CANDIDATES)]
    scores = get_reranker().score(query, [d.page_content for d in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def best_sentences(query: str, text: str, n: int = 2) -> list[str]:
    body = text.split(": ", 1)[-1]
    parts = re.split(r"(?<=\.)\s+|\s+(?=[a-zçğış]\)\s)", body)
    sentences = [s.strip() for s in parts if len(s.strip()) > 25]
    if not sentences:
        return []
    scores = get_reranker().score(query, sentences)
    top = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:n]
    return [s if len(s) <= 300 else s[:300].rsplit(" ", 1)[0] + "…" for s in (sentences[i] for i in sorted(top))]


METHODS = {
    "BM25": search_bm25,
    "Dense (bge-m3)": search_dense,
    "Hibrit (RRF)": search_hybrid,
    "Hibrit + Reranker": search_reranked,
}


def source_label(doc: Document) -> str:
    m = doc.metadata
    return f"{m['kanun_no']} s. {m['kanun_adi']} m.{m['madde_no']}"
