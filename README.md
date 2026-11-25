# Türk İş Hukuku için Kendini Denetleyen RAG Tabanlı Soru-Cevap Sistemi

4857 sayılı İş Kanunu (ve 1475 s. Kanun m.14 – kıdem tazminatı) üzerine, kaynak madde gösteren,
kapsam dışı soruları reddeden Türkçe soru-cevap sistemi.

## Mimari

```
Soru → Hibrit arama (BM25 + bge-m3) → Reranker (bge-reranker-v2-m3) → eşik kontrolü
         │                                                        │
         │                                  en iyi skor < eşik → RET (kapsam dışı)
         ▼
      LLM cevap (qwen2.5:7b, Ollama) → Kontrol (uydurma var mı? soruyu karşılıyor mu?)
                                            │ uydurma → yeniden üret (1 kez)
                                            │ karşılamıyor → soruyu yeniden yaz, tekrar ara (1 kez)
                                            ▼
                                     Cevap + kaynak maddeler
```

| Bileşen | Seçim |
|---|---|
| Veri | HuggingFace `mtntasci/turkish-legal-rag` (kaynak: mevzuat.gov.tr), madde bazlı |
| Kelime araması | BM25, Türkçe için ilk-5-harf kök bulma (F5 stemming) |
| Vektör araması | `BAAI/bge-m3` + ChromaDB (kosinüs) |
| Birleştirme | Reciprocal Rank Fusion (RRF) |
| Yeniden sıralama | `BAAI/bge-reranker-v2-m3` (cross-encoder, GPU) |
| LLM | `qwen2.5:7b` (Ollama, yerel) veya Groq |
| Akış | LangGraph |
| Arayüz | Streamlit |

## Kurulum

```bash
conda activate ai_project
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50 serisi GPU için
ollama pull qwen2.5:7b
```

## Çalıştırma (sırasıyla)

```bash
python -m scripts.ingest            # 1) İş Kanunu'nu indir, Chroma'ya yükle (bir kere)
python -m scripts.smoke_test        # 2) 8 örnek soruyla hızlı test
streamlit run streamlit_app.py      # 3) Arayüz
python -m eval.build_questions      # 4) 300 soruluk test seti -> eval/questions.csv (sonra ELLE KONTROL)
python -m eval.evaluate             # 5) Deney 1 + 2: tablolar ve grafikler -> results/
python -m eval.evaluate --e2e       # 6) + Deney 3: uçtan uca (yavaş, ~30-60 dk)
```

## Deneyler ve metrikler

Sorular %30 dev / %70 test olarak ayrılır. Eşikler dev setinde seçilir, metrikler test setinde raporlanır.

| Deney | Soru | Metrikler | Karşılaştırılanlar |
|---|---|---|---|
| 1. Kapsam tespiti | Soru İş Kanunu ile cevaplanabilir mi? | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion matrix | BM25, Dense, Hibrit, Hibrit + Reranker |
| 2. Madde alakası | Getirilen madde doğru madde mi? | + Recall@1, Recall@5, MRR | Aynı 4 yöntem |
| 3. Uçtan uca | Sistem doğru cevapladı / reddetti mi? | Accuracy, F1, doğru kaynak oranı, süre | Basit RAG vs Self-Corrective RAG |

Çıktılar: `results/deney*_*.csv|md` (tablolar), `results/deney*_roc.png`, `results/deney*_confusion.png`,
`results/deney3_detay.csv` (her cevap tek tek, halüsinasyon incelemesi için).

## Klasör yapısı

```
app/
  config.py            ayarlar (.env ile değiştirilebilir)
  clients/             LLM, embedding, reranker, Chroma
  retrieval.py         4 arama yöntemi
  rag/                 LangGraph akışı (state, nodes, graph) + karşılaştırma için basit RAG
scripts/ingest.py      veri yükleme
scripts/smoke_test.py  hızlı test
eval/                  soru seti üretimi + değerlendirme
streamlit_app.py       arayüz
```
