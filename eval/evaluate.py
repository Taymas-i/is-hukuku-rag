import argparse
import json
import time

import app
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from app.config import ROOT, settings
from app.retrieval import METHODS

QUESTIONS = ROOT / "eval" / "questions.csv"
RESULTS = ROOT / "results"
SEED = 42


def best_threshold(y_true, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def binary_metrics(y_true, y_pred, scores=None) -> dict:
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if scores is not None:
        m["ROC-AUC"] = roc_auc_score(y_true, scores)
    return m


def run_retrieval(questions: pd.DataFrame) -> dict[str, list[list[tuple[str, float]]]]:
    cache = RESULTS / "retrieval_cache.json"
    key = f"{settings.TOP_K}|{settings.CANDIDATES}|" + "\n".join(questions["soru"])
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            print("  (önbellekten okundu: results/retrieval_cache.json)")
            return saved["runs"]

    runs = {}
    for name, search in METHODS.items():
        t0 = time.perf_counter()
        runs[name] = [
            [(d.metadata["madde_id"], float(s)) for d, s in search(q, settings.TOP_K)]
            for q in questions["soru"]
        ]
        print(f"  {name}: {time.perf_counter() - t0:.1f} sn", flush=True)
    cache.write_text(json.dumps({"key": key, "runs": runs}, ensure_ascii=False), encoding="utf-8")
    return runs


def plot_roc(curves: dict, title: str, path):
    plt.figure(figsize=(6, 5))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, s):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("Yanlış Pozitif Oranı")
    plt.ylabel("Doğru Pozitif Oranı")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrices(cms: dict, labels: list[str], title: str, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 3.6))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str):
    df = df.round(3)
    df.to_csv(RESULTS / f"{name}.csv", encoding="utf-8-sig")
    (RESULTS / f"{name}.md").write_text(df.to_markdown(), encoding="utf-8")
    print(f"\n### {name}\n{df.to_markdown()}")


def experiment_scope(runs, dev, test):
    rows, curves, cms, thresholds = {}, {}, {}, {}
    for name, run in runs.items():
        top1 = np.array([r[0][1] if r else 0.0 for r in run])
        th = best_threshold(dev["kapsam"], top1[dev.index])
        y, s = test["kapsam"].to_numpy(), top1[test.index]
        pred = (s >= th).astype(int)
        rows[name] = {**binary_metrics(y, pred, s), "Eşik": th}
        curves[name] = (y, s)
        cms[name] = confusion_matrix(y, pred, labels=[1, 0])
        thresholds[name] = th
    save_table(pd.DataFrame(rows).T, "deney1_kapsam_tespiti")
    plot_roc(curves, "Deney 1: Kapsam tespiti ROC", RESULTS / "deney1_roc.png")
    plot_confusion_matrices(cms, ["Kapsam içi", "Kapsam dışı"], "Deney 1: Kapsam tespiti",
                            RESULTS / "deney1_confusion.png")
    return thresholds


def pairs_for(run, questions, idx):
    y, s = [], []
    for i in idx:
        gold = set(str(questions.at[i, "gold_ids"]).split(";"))
        for doc_id, score in run[i]:
            y.append(int(doc_id in gold))
            s.append(score)
    return np.array(y), np.array(s)


def experiment_articles(runs, questions, dev, test):
    dev_idx = dev.index[dev["kapsam"] == 1]
    test_idx = test.index[test["kapsam"] == 1]
    rows, curves, cms = {}, {}, {}
    for name, run in runs.items():
        th = best_threshold(*pairs_for(run, questions, dev_idx))
        y, s = pairs_for(run, questions, test_idx)
        pred = (s >= th).astype(int)

        ranks = []
        for i in test_idx:
            gold = set(str(questions.at[i, "gold_ids"]).split(";"))
            ids = [doc_id for doc_id, _ in run[i]]
            ranks.append(next((r + 1 for r, d in enumerate(ids) if d in gold), None))
        rows[name] = {
            **binary_metrics(y, pred, s),
            "Recall@1": np.mean([r == 1 for r in ranks]),
            f"Recall@{settings.TOP_K}": np.mean([r is not None for r in ranks]),
            "MRR": np.mean([1 / r if r else 0 for r in ranks]),
        }
        curves[name] = (y, s)
        cms[name] = confusion_matrix(y, pred, labels=[1, 0])
    save_table(pd.DataFrame(rows).T, "deney2_madde_alaka")
    plot_roc(curves, "Deney 2: Madde alaka ROC", RESULTS / "deney2_roc.png")
    plot_confusion_matrices(cms, ["Alakalı", "Alakasız"], "Deney 2: Madde alaka",
                            RESULTS / "deney2_confusion.png")


def experiment_end_to_end(test):
    from app.rag.graph import build_initial_state, get_rag_graph
    from app.rag.naive import naive_rag

    graph = get_rag_graph()
    systems = {
        "Basit RAG": naive_rag,
        "Self-Corrective RAG": lambda q: graph.invoke(build_initial_state(q)),
    }
    details, rows, cms = [], {}, {}
    y = test["kapsam"].to_numpy()
    for name, run in systems.items():
        preds, correct_source, times = [], [], []
        for n, (_, q) in enumerate(test.iterrows(), 1):
            t0 = time.perf_counter()
            out = run(q["soru"])
            times.append(time.perf_counter() - t0)
            answered = int(out["final_status"] == "answered")
            preds.append(answered)
            gold_madde = {g.split("-")[1] for g in str(q["gold_ids"]).split(";") if "-" in g}
            cited = {s.rsplit("m.", 1)[-1] for s in out["sources"]}
            if q["kapsam"] == 1 and answered:
                correct_source.append(bool(gold_madde & cited))
            details.append({"sistem": name, "id": q["id"], "soru": q["soru"], "kapsam": q["kapsam"],
                            "gold_ids": q["gold_ids"], "durum": out["final_status"],
                            "cevap": out["answer"], "kaynaklar": "; ".join(out["sources"]),
                            "sure_sn": round(times[-1], 2)})
            print(f"  {name} [{n}/{len(test)}]", end="\r")
        rows[name] = {
            **binary_metrics(y, preds),
            "Doğru kaynak oranı": np.mean(correct_source) if correct_source else 0.0,
            "Ort. süre (sn)": np.mean(times),
        }
        cms[name] = confusion_matrix(y, preds, labels=[1, 0])
    save_table(pd.DataFrame(rows).T, "deney3_uctan_uca")
    plot_confusion_matrices(cms, ["Cevapla", "Reddet"], "Deney 3: Uçtan uca",
                            RESULTS / "deney3_confusion.png")
    pd.DataFrame(details).to_csv(RESULTS / "deney3_detay.csv", index=False, encoding="utf-8-sig")
    print("\nCevapların tek tek incelenmesi için: results/deney3_detay.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true", help="Deney 3'ü de çalıştır (LLM, yavaş)")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    questions = pd.read_csv(QUESTIONS).reset_index(drop=True)
    dev, test = train_test_split(questions, test_size=0.7, stratify=questions["kapsam"], random_state=SEED)
    print(f"{len(questions)} soru: dev={len(dev)}, test={len(test)}")

    print("Arama yöntemleri çalışıyor...")
    runs = run_retrieval(questions)
    thresholds = experiment_scope(runs, dev, test)
    experiment_articles(runs, questions, dev, test)

    print(
        f"\nÖneri: .env içine RELEVANCE_THRESHOLD={thresholds['Hibrit + Reranker']:.3f} "
        "yazarak sistemin eşiğini dev setine göre ayarlayın."
    )
    if args.e2e:
        experiment_end_to_end(test)
    print(f"\nTablolar ve grafikler: {RESULTS}")


if __name__ == "__main__":
    main()
