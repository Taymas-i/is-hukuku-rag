import json
import random

import pandas as pd
from pydantic import BaseModel, Field

from app.clients.llm import invoke_structured
from app.config import ROOT, settings
from app.retrieval import turkish_lower

OUT = ROOT / "eval" / "questions.csv"
N_IN_SCOPE = 200
N_OUT_SCOPE = 100
SEED = 42

LABOR_WORDS = ["işçi", "işveren", "ücret", "kıdem", "ihbar", "iş sözleşme", "çalış", "izin", "mesai", "sendika"]

OUT_SCOPE_TOPICS = [
    "boşanma ve nafaka", "miras ve vasiyet", "trafik cezaları", "kira artışı ve tahliye",
    "tüketici hakları ve ayıplı mal", "vergi ve beyanname", "hırsızlık ve dolandırıcılık suçları",
    "tapu ve gayrimenkul satışı", "askerlik", "üniversite sınavı ve eğitim",
    "yemek tarifi", "futbol", "hava durumu", "bilgisayar ve telefon sorunları",
]


class Questions(BaseModel):
    sorular: list[str] = Field(description="Üretilen sorular")


def load_qa_benchmark() -> pd.DataFrame:
    path = next((ROOT / "data" / "raw").rglob("*qa*.csv"))
    return pd.read_csv(path)


def is_labor_law(row) -> bool:
    name = turkish_lower(f"{row.get('kanun_adi', '')} {row.get('kaynak', '')}")
    return "iş kanunu" in name


def has_labor_words(text: str) -> bool:
    t = turkish_lower(text)
    return any(w in t for w in LABOR_WORDS)


def norm_no(value) -> str:
    s = str(value).strip()
    return s[:-2] if s.endswith(".0") else s


def in_scope_questions(corpus: list[dict], qa: pd.DataFrame) -> list[dict]:
    ids = {r["metadata"]["madde_id"] for r in corpus}
    out = []

    for _, r in qa.iterrows():
        if not is_labor_law(r) or str(r.get("is_active", True)).lower() in ("false", "0"):
            continue
        law = "1475" if "1475" in str(r.get("kanun_adi", "")) + str(r.get("kaynak", "")) else "4857"
        gold = f"{law}-{norm_no(r['madde_no'])}"
        if gold in ids:
            out.append({"soru": r["soru"], "kapsam": 1, "gold_ids": gold, "kaynak": "qa_benchmark"})
    print(f"qa_benchmark'tan {len(out)} İş Kanunu sorusu")

    articles = [r for r in corpus if len(r["text"]) > 300]
    per_article = max(1, -(-(N_IN_SCOPE - len(out)) // len(articles)))
    for i, r in enumerate(articles, 1):
        prompt = (
            f"Aşağıdaki kanun maddesini oku ve bir işçinin veya işverenin günlük dille sorabileceği, "
            f"SADECE bu maddeyle cevaplanabilen {per_article} farklı soru yaz.\n"
            "Kurallar: Soruda madde numarası veya kanun adı geçmesin. Somut bir durum anlatsın "
            "(ör. '3 yıldır çalışıyorum, işten çıkarılırsam ne kadar önceden haber verilmeli?').\n\n"
            f"Madde:\n{r['text'][:3000]}"
        )
        try:
            qs = invoke_structured(Questions, prompt, temperature=0.7).sorular[:per_article]
        except Exception as e:
            print(f"  atlandı {r['metadata']['id']}: {e}")
            continue
        out += [{"soru": q, "kapsam": 1, "gold_ids": r["metadata"]["madde_id"], "kaynak": "llm"} for q in qs]
        print(f"  [{i}/{len(articles)}] {r['metadata']['id']}: {len(qs)} soru")

    random.Random(SEED).shuffle(out)
    return out[:N_IN_SCOPE]


def out_scope_questions(qa: pd.DataFrame) -> list[dict]:
    others = [
        q for _, r in qa.iterrows()
        if not is_labor_law(r) and not has_labor_words(q := str(r["soru"]))
    ]
    random.Random(SEED).shuffle(others)
    out = [{"soru": q, "kapsam": 0, "gold_ids": "", "kaynak": "qa_benchmark"} for q in others[:70]]

    n_left = N_OUT_SCOPE - len(out)
    per_topic = -(-n_left // len(OUT_SCOPE_TOPICS))
    for topic in OUT_SCOPE_TOPICS:
        prompt = f"'{topic}' konusunda sıradan bir kişinin sorabileceği {per_topic} farklı Türkçe soru yaz."
        qs = invoke_structured(Questions, prompt, temperature=0.9).sorular[:per_topic]
        out += [
            {"soru": q, "kapsam": 0, "gold_ids": "", "kaynak": "llm"}
            for q in qs if not has_labor_words(q)
        ]
    return out[:N_OUT_SCOPE]


def main():
    with open(settings.CORPUS_PATH, encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]
    qa = load_qa_benchmark()

    rows = in_scope_questions(corpus, qa) + out_scope_questions(qa)
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))
    df["kontrol_edildi"] = ""
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n{len(df)} soru -> {OUT.relative_to(ROOT)}")
    print(df.groupby(["kapsam", "kaynak"]).size())


if __name__ == "__main__":
    main()
