from functools import lru_cache

from langgraph.graph import END, StateGraph

from app.rag.nodes import (
    check_answer,
    finalize,
    generate,
    grade_documents,
    retrieve,
    transform_query,
)
from app.rag.state import RAGState

MAX_GENERATIONS = 2
MAX_REWRITE_RETRIES = 1


def route_after_grade(state: RAGState) -> str:
    return "generate" if state["relevant_docs"] else "finalize"


def route_after_check(state: RAGState) -> str:
    if state["is_grounded"] and state["answers_question"]:
        return "finalize"
    if not state["is_grounded"] and state["generation_count"] < MAX_GENERATIONS:
        return "generate"
    if not state["answers_question"] and state["rewrite_retries"] < MAX_REWRITE_RETRIES:
        return "transform_query"
    return "finalize"


def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.add_node("check_answer", check_answer)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "finalize": "finalize"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "check_answer")
    workflow.add_conditional_edges(
        "check_answer",
        route_after_check,
        {"finalize": "finalize", "generate": "generate", "transform_query": "transform_query"},
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


@lru_cache
def get_rag_graph():
    return build_rag_graph()


def build_initial_state(question: str) -> RAGState:
    return RAGState(
        question=question,
        rewritten_question=question,
        retrieved_docs=[],
        retrieved_scores=[],
        relevant_docs=[],
        relevant_scores=[],
        answer="",
        sources=[],
        is_grounded=False,
        answers_question=False,
        generation_count=0,
        rewrite_retries=0,
        final_status="answered",
    )
