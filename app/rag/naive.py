from app.clients.llm import get_llm
from app.config import settings
from app.rag.nodes import REFUSAL_PHRASE, build_answer_prompt, cited_docs
from app.retrieval import search_dense, source_label, turkish_lower


def naive_rag(question: str) -> dict:
    docs = [d for d, _ in search_dense(question, settings.TOP_K)]
    answer = get_llm().invoke(build_answer_prompt(question, docs)).content
    refused = turkish_lower(REFUSAL_PHRASE.rstrip(".")) in turkish_lower(answer)
    return {
        "answer": answer,
        "sources": [] if refused else list(dict.fromkeys(source_label(d) for d in cited_docs(answer, docs))),
        "final_status": "not_found" if refused else "answered",
    }
