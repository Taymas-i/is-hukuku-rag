from functools import lru_cache

from app.config import settings


def groq_model(model: str, temperature: float):
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=settings.GROQ_API_KEY,
        max_retries=1,
        reasoning_format="hidden" if "qwen" in model else None,
    )


def with_groq_fallback(build, temperature: float = 0.0):
    from groq import RateLimitError

    primary = build(groq_model(settings.GROQ_MODEL, temperature))
    fallback = build(groq_model(settings.GROQ_FALLBACK_MODEL, temperature))
    return primary.with_fallbacks([fallback], exceptions_to_handle=(RateLimitError,))


@lru_cache
def get_llm(temperature: float = 0.0):
    if settings.LLM_PROVIDER == "groq":
        return with_groq_fallback(lambda llm: llm, temperature)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.LLM_MODEL,
        temperature=temperature,
        num_ctx=settings.LLM_NUM_CTX,
        keep_alive="30m",
        reasoning=False if settings.LLM_MODEL.startswith("qwen3") else None,
    )


def invoke_structured(schema, prompt: str, temperature: float = 0.0):
    if settings.LLM_PROVIDER == "groq":
        return with_groq_fallback(lambda llm: llm.with_structured_output(schema), temperature).invoke(prompt)
    llm = get_llm(temperature=temperature).with_structured_output(schema)
    return llm.invoke(prompt)
