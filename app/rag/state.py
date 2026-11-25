from typing import Literal, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    rewritten_question: str
    retrieved_docs: list[Document]
    retrieved_scores: list[float]
    relevant_docs: list[Document]
    relevant_scores: list[float]
    answer: str
    sources: list[str]
    is_grounded: bool
    answers_question: bool

    generation_count: int
    rewrite_retries: int

    final_status: Literal["answered", "not_found"]
