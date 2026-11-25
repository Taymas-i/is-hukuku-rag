import time

import pandas as pd
import streamlit as st

from app.config import ROOT, settings
from app.rag.graph import build_initial_state, get_rag_graph
from app.retrieval import best_sentences, load_corpus, source_label

EXAMPLES = [
    "Gece en fazla kaç saat çalıştırılabilirim?",
    "Haftada en fazla kaç saat çalıştırılabilirim?",
    "İşe başlayalı 2 yıl oldu, işverenim beni çıkarmak isterse kaç hafta önceden haber vermeli?",
    "15 yaşındaki bir çocuk çalıştırılabilir mi?",
]

TOPICS = "yıllık izin, çalışma süreleri, ücret, fazla mesai, iş sözleşmesinin feshi, ihbar süreleri, doğum izni, çocuk işçiler"

st.set_page_config(page_title="İş Hukuku Asistanı", page_icon="⚖️")

if settings.LLM_PROVIDER == "groq":
    settings.ASSISTANT_MODE = True
MODEL_NAME = settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq" else f"{settings.LLM_MODEL} (yerel)"


@st.cache_resource(show_spinner="Modeller yükleniyor (ilk açılışta yaklaşık 1 dakika sürer)...")
def load_graph():
    from app.retrieval import search_reranked

    search_reranked("yıllık izin", 1)
    return get_rag_graph()


@st.cache_data
def load_results():
    try:
        e2e = pd.read_csv(ROOT / "results" / "deney3_uctan_uca.csv", index_col=0)
        detail = pd.read_csv(ROOT / "results" / "deney3_detay.csv")
        articles = pd.read_csv(ROOT / "results" / "deney2_madde_alaka.csv", index_col=0)
    except FileNotFoundError:
        return None
    ours = detail[(detail.sistem == "Self-Corrective RAG") & (detail.kapsam == 0)]
    return {
        "accuracy": e2e.loc["Self-Corrective RAG", "Accuracy"],
        "baseline": e2e.loc["Basit RAG", "Accuracy"],
        "rejected": f"{(ours.durum == 'not_found').sum()} / {len(ours)}",
        "recall1": articles.loc["Hibrit + Reranker", "Recall@1"],
    }


def sidebar():
    with st.sidebar:
        st.header("Nasıl çalışır?")
        st.markdown(
            "1. **Arama:** Soru, İş Kanunu'nun tüm maddeleri içinde kelime ve anlam bazlı aranır.\n"
            "2. **Sıralama:** Reranker modeli en alakalı 5 maddeyi seçer ve puanlar.\n"
            "3. **Kapsam kontrolü:** Hiçbir madde yeterince alakalı değilse soru reddedilir.\n"
            "4. **Cevap:** Yapay zekâ sadece bu maddeleri okuyarak cevap yazar.\n"
            "5. **Doğrulama:** Cevabın maddelerde gerçekten yazıp yazmadığı kontrol edilir."
        )
        st.divider()
        st.header("Sistem")
        st.markdown(
            f"- **Kaynak:** 4857 sayılı İş Kanunu ({len(load_corpus())} parça)\n"
            f"- **Dil modeli:** {MODEL_NAME}\n"
            "- **Arama:** BM25 + bge-m3 + bge-reranker\n"
            "- **Akış:** LangGraph"
        )
        results = load_results()
        if results:
            st.divider()
            st.header("Test sonuçları")
            st.metric("Genel doğruluk", f"%{results['accuracy'] * 100:.1f}",
                      f"+{(results['accuracy'] - results['baseline']) * 100:.1f} basit RAG'e göre")
            st.metric("Konu dışı soruları reddetme", results["rejected"])
            st.metric("Doğru maddeyi ilk sırada bulma", f"%{results['recall1'] * 100:.0f}")
        st.divider()
        if st.button("Sohbeti temizle", use_container_width=True):
            st.session_state.history = []
            st.rerun()


def render_steps(container, steps):
    for icon, text, seconds in steps:
        container.markdown(f"{icon} {text} · `{seconds:.1f} sn`")


def render_sources(sources):
    for src in sources:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"**📜 {src['label']}**")
            right.link_button("Resmî metin", src["url"], use_container_width=True)
            st.progress(min(max(src["score"], 0.0), 1.0), text=f"Alaka puanı: {src['score']:.2f}")
            if src["quote"]:
                st.markdown("**Kanun ne diyor?**")
                for sentence in src["quote"]:
                    st.markdown(f"> {sentence}")
            with st.expander("Maddenin tamamı"):
                st.caption(src["text"])


def render_message(msg):
    if msg.get("refused"):
        st.warning(msg["content"], icon="🚫")
        st.caption(f"Sorabileceğiniz konulara örnek: {TOPICS}.")
    else:
        st.markdown(msg["content"])
    if msg.get("steps"):
        with st.expander(f"⚙️ Sistem adımları · toplam {msg['elapsed']:.1f} sn"):
            render_steps(st, msg["steps"])
    render_sources(msg.get("sources", []))


def describe_step(node, update):
    if node == "retrieve":
        return "🔎", "Kanun maddeleri arandı (hibrit arama + reranker)"
    if node == "grade_documents":
        scores = update.get("relevant_scores") or []
        if scores:
            return "✅", f"Kapsam kontrolü geçti, en yüksek alaka puanı {scores[0]:.2f}"
        return "⛔", "Yeterince alakalı madde bulunamadı"
    if node == "transform_query":
        return "✏️", "Soru kanun terimleriyle yeniden yazıldı"
    if node == "generate":
        return "🤖", "Cevap yazıldı"
    if node == "check_answer":
        if update.get("is_grounded") and update.get("answers_question"):
            return "🛡️", "Cevap kanun maddelerine karşı doğrulandı"
        return "⚠️", "Cevap doğrulanamadı, tekrar deneniyor"
    return "🏁", "Tamamlandı"


NEXT_LABELS = {
    "retrieve": "Alaka kontrolü yapılıyor...",
    "grade_documents": "Cevap yazılıyor...",
    "transform_query": "Kanun maddeleri yeniden aranıyor...",
    "generate": "Cevap kanuna karşı doğrulanıyor...",
    "check_answer": "Sonuçlanıyor...",
}


def ask(question):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})

    with st.chat_message("assistant", avatar="⚖️"):
        status = st.status("Kanun maddeleri aranıyor...")
        placeholder = st.empty()
        text, step, final, steps = "", None, {}, []
        start = last = time.perf_counter()

        try:
            for mode, chunk in graph.stream(build_initial_state(question), stream_mode=["messages", "updates"]):
                if mode == "messages":
                    msg, meta = chunk
                    if meta.get("langgraph_node") != "generate":
                        continue
                    if meta.get("langgraph_step") != step:
                        step, text = meta.get("langgraph_step"), ""
                    text += msg.content
                    placeholder.markdown(text + "▌")
                    continue
                for node, update in chunk.items():
                    update = update or {}
                    final.update(update)
                    now = time.perf_counter()
                    if node != "finalize":
                        icon, desc = describe_step(node, update)
                        steps.append((icon, desc, now - last))
                        status.markdown(f"{icon} {desc} · `{now - last:.1f} sn`")
                    last = now
                    status.update(label=NEXT_LABELS.get(node, "Sonuçlanıyor..."))
        except Exception as error:
            status.update(label="Bir sorun oluştu", state="error")
            placeholder.empty()
            if "rate" in str(error).lower() or "429" in str(error):
                st.error("Dil modelinin kullanım sınırına ulaşıldı. Birkaç saniye bekleyip tekrar deneyin.", icon="⏳")
            else:
                st.error(f"Beklenmeyen bir hata oluştu: {error}", icon="⚠️")
            st.session_state.history.pop()
            return

        elapsed = time.perf_counter() - start
        refused = final.get("final_status") != "answered"
        placeholder.empty()
        status.update(
            label=f"{'Reddedildi' if refused else 'Cevaplandı'} · {elapsed:.1f} sn",
            state="error" if refused else "complete",
            expanded=False,
        )

        sources = []
        if not refused:
            for doc, score in zip(final.get("relevant_docs", []), final.get("relevant_scores", [])):
                label = source_label(doc)
                if label in final.get("sources", []) and label not in [s["label"] for s in sources]:
                    sources.append({
                        "label": label,
                        "score": score,
                        "url": doc.metadata.get("url", "https://www.mevzuat.gov.tr"),
                        "quote": best_sentences(question, doc.page_content),
                        "text": doc.page_content,
                    })

        message = {"role": "assistant", "content": final["answer"], "refused": refused, "sources": sources}
        if refused:
            st.warning(message["content"], icon="🚫")
            st.caption(f"Sorabileceğiniz konulara örnek: {TOPICS}.")
        else:
            st.markdown(message["content"])
        render_sources(sources)

    st.session_state.history.append({**message, "steps": steps, "elapsed": elapsed})


graph = load_graph()
st.session_state.setdefault("history", [])
sidebar()

st.title("⚖️ İş Hukuku Asistanı")
st.caption(
    "4857 sayılı İş Kanunu'na dayanarak cevap verir ve kaynağını gösterir. "
    "Hukuki danışmanlık yerine geçmez."
)

for msg in st.session_state.history:
    with st.chat_message(msg["role"], avatar="⚖️" if msg["role"] == "assistant" else None):
        if msg["role"] == "assistant":
            render_message(msg)
        else:
            st.markdown(msg["content"])

question = st.chat_input("İş hukukuyla ilgili sorunuzu yazın...") or st.session_state.pop("pending", None)

if not st.session_state.history and not question:
    st.markdown("**Örnek sorular:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending = example
            st.rerun()

if question:
    ask(question)
