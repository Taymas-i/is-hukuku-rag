import logging
import re

from pydantic import BaseModel, Field

from app.clients.llm import get_llm, invoke_structured
from app.config import settings
from app.rag.state import RAGState
from app.retrieval import search_reranked, source_label

logger = logging.getLogger(__name__)

NOT_FOUND_ANSWER = (
    "Bu soru İş Kanunu kapsamında görünmüyor ya da ilgili bir madde bulamadım. "
    "Bu konuda cevap veremiyorum."
)
REFUSAL_PHRASE = "Bu bilgi verilen maddelerde yok."


class RewrittenQuery(BaseModel):
    rewritten_question: str = Field(description="Kanun maddelerinde aramaya daha uygun, yeniden yazılmış soru")


class AnswerCheck(BaseModel):
    is_grounded: bool = Field(description="Cevaptaki her bilgi verilen kanun maddelerinde geçiyor mu")
    answers_question: bool = Field(description="Cevap kullanıcının sorusunu gerçekten cevaplıyor mu")


def cited_docs(answer: str, docs) -> list:
    cited = set(re.findall(r"m\.\s*(\d+)", answer))
    picked = [d for d in docs if str(d.metadata["madde_no"]) in cited]
    return picked or list(docs)


def format_context(docs) -> str:
    return "\n\n---\n\n".join(f"[{source_label(d)}]\n{d.page_content}" for d in docs)


ASSISTANT_PROMPT = (
    "Sen, Türk iş hukuku konusunda çalışanlara ve işverenlere yardımcı olan bir hukuk asistanısın. "
    "Görevin, kullanıcının sorusunu YALNIZCA aşağıdaki kanun maddelerine dayanarak, onun durumuna "
    "uyarlayarak cevaplamak.\n\n"
    "Cevap düzeni (Markdown):\n"
    "1. İlk paragrafta soruyu kullanıcının durumuna uygulayarak tam bir cümleyle doğrudan cevapla. "
    "Kullanıcı çalışma süresi, yaş, saat gibi bir bilgi verdiyse hesabı onun bilgisiyle yap ve bunu "
    "cümlede belirt. Paragrafın tamamını değil, sadece sonucu kalın yaz. Örnek: 'İki yıldır "
    "çalıştığınız için işvereniniz sözleşmenizi feshetmeden en az **altı hafta** önce size "
    "bildirmelidir (m.17).'\n"
    "2. Ayrı bir satırda **Açıklama** yazıp altında ilgili kuralı sade bir dille 1-3 cümlede anlat.\n"
    "3. Sadece maddelerde açıkça yazıyorsa ayrı bir satırda **Bilmeniz gerekenler** yazıp altında "
    "istisnaları, şartları veya hakları 1-3 maddelik liste halinde yaz. Yoksa bu bölümü hiç yazma.\n"
    "Başlıklar için # veya ### kullanma, sadece kalın yazı kullan.\n\n"
    "Kurallar:\n"
    "- Maddelerde olmayan hiçbir bilgiyi ekleme: başka kanun, mahkeme kararı, tahmini tutar veya genel "
    "tavsiye yok.\n"
    "- Her bilgiden sonra parantez içinde madde numarasını yaz, ör. (m.17).\n"
    "- Süre, gün, oran ve tutarları maddeden aynen al. Maddede yazıyla yazılmış sayıları "
    "(ör. 'ikiyüzyetmiş saat') asla rakama çevirme, maddede yazdığı gibi yazıyla yaz.\n"
    "- Bilmeniz gerekenler bölümüne sadece soruyla doğrudan ilgili bilgileri yaz.\n"
    "- Cevap kullanıcının vermediği bir bilgiye bağlıysa, maddedeki seçenekleri kısa bir liste halinde "
    "göster ('... ise ..., ... ise ...').\n"
    "- Samimi ama profesyonel ol, 'siz' diye hitap et, toplam 170 kelimeyi geçme.\n"
    "- Maddelerden bazıları soruyla ilgisiz olabilir, onları kullanma.\n"
    f"- Sorunun cevabı maddelerde yoksa sadece şunu yaz: \"{REFUSAL_PHRASE}\"\n\n"
)


def build_answer_prompt(question: str, docs) -> str:
    if settings.ASSISTANT_MODE:
        return f"{ASSISTANT_PROMPT}Kanun maddeleri:\n{format_context(docs)}\n\nSoru: {question}"
    return (
        "Sen Türk iş hukuku konusunda yardımcı bir asistansın. Aşağıdaki kanun maddelerini "
        "kullanarak soruyu Türkçe, sade ve kısa cevapla.\n"
        "Kurallar:\n"
        "- SADECE verilen maddelerdeki bilgiyi kullan, genel bilgiyle veya tahminle doldurma.\n"
        "- Kullandığın her bilgi için parantez içinde madde numarasını yaz, ör. (m.53).\n"
        "- Süre, gün, oran ve tutarları maddede yazdığı gibi aynen aktar; aralıkların sınırlarına "
        "(ör. 'beş yıl dahil') dikkat et.\n"
        "- Maddelerden bazıları soruyla ilgisiz olabilir, onları kullanma.\n"
        f"- Cevap maddelerde yoksa sadece şunu yaz: \"{REFUSAL_PHRASE}\"\n\n"
        f"Kanun maddeleri:\n{format_context(docs)}\n\n"
        f"Soru: {question}"
    )


def retrieve(state: RAGState) -> dict:
    ranked = search_reranked(state["rewritten_question"], settings.TOP_K)
    return {
        "retrieved_docs": [d for d, _ in ranked],
        "retrieved_scores": [s for _, s in ranked],
    }


def grade_documents(state: RAGState) -> dict:
    scores = state["retrieved_scores"]
    if not scores or scores[0] < settings.RELEVANCE_THRESHOLD:
        return {"relevant_docs": [], "relevant_scores": []}
    return {"relevant_docs": state["retrieved_docs"], "relevant_scores": scores}


def transform_query(state: RAGState) -> dict:
    prompt = (
        "Aşağıdaki soru İş Kanunu maddeleri içinde arama yapmak için kullanılacak ama iyi sonuç vermedi. "
        "Soruyu kanunda geçmesi muhtemel hukuki terimlerle (ör. 'kıdem tazminatı', 'ihbar süresi', "
        "'fazla çalışma', 'yıllık ücretli izin', 'iş sözleşmesinin feshi') yeniden yaz, anlamını değiştirme.\n\n"
        f"Orijinal soru: {state['question']}\n"
        f"Son denenen sorgu: {state['rewritten_question']}"
    )
    result = invoke_structured(RewrittenQuery, prompt)
    return {
        "rewritten_question": result.rewritten_question,
        "rewrite_retries": state["rewrite_retries"] + 1,
    }


def generate(state: RAGState) -> dict:
    response = get_llm().invoke(build_answer_prompt(state["question"], state["relevant_docs"]))
    return {
        "answer": response.content,
        "sources": list(dict.fromkeys(source_label(d) for d in cited_docs(response.content, state["relevant_docs"]))),
        "generation_count": state["generation_count"] + 1,
    }


def check_answer(state: RAGState) -> dict:
    docs = state["relevant_docs"]
    number_check = ""
    if settings.ASSISTANT_MODE:
        docs = cited_docs(state["answer"], docs)
        number_check = (
            "Cevaptaki her sayıyı (süre, gün, saat, oran, tutar) maddedeki ifadeyle tek tek karşılaştır; "
            "biri bile farklıysa is_grounded false olmalı.\n"
        )
    prompt = (
        f"Kanun maddeleri:\n{format_context(docs)}\n\n"
        f"Soru: {state['question']}\n\n"
        f"Üretilen cevap:\n{state['answer']}\n\n"
        "İki şeyi değerlendir:\n"
        "1) is_grounded: Cevaptaki her iddia yukarıdaki maddelerde geçiyor mu? "
        "Maddelerde olmayan bir süre, oran, tutar veya hak varsa false.\n"
        f"{number_check}"
        "2) answers_question: Cevap sorulan soruyu gerçekten karşılıyor mu? "
        f"'{REFUSAL_PHRASE}' gibi bir ret cevabıysa false."
    )
    result = invoke_structured(AnswerCheck, prompt)
    return {"is_grounded": result.is_grounded, "answers_question": result.answers_question}


def finalize(state: RAGState) -> dict:
    answered = (
        bool(state.get("relevant_docs"))
        and state.get("is_grounded", False)
        and state.get("answers_question", False)
        and REFUSAL_PHRASE.rstrip(".") not in state.get("answer", "")
    )
    logger.info(
        "rag_finalized status=%s generations=%d rewrites=%d",
        "answered" if answered else "not_found",
        state.get("generation_count", 0),
        state.get("rewrite_retries", 0),
    )
    if answered:
        return {"final_status": "answered"}
    return {"final_status": "not_found", "answer": NOT_FOUND_ANSWER, "sources": []}
