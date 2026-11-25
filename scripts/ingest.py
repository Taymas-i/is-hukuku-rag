import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from langchain_core.documents import Document

from app.config import ROOT, settings

REPO = "mtntasci/turkish-legal-rag"
RAW_DIR = ROOT / "data" / "raw"

LAWS = {
    "4857": {"kanun_adi": "İş Kanunu", "maddeler": None},
    "1475": {"kanun_adi": "İş Kanunu (eski)", "maddeler": {"14"}},
}


def download_csvs() -> dict[str, Path]:
    paths = {}
    for f in list_repo_files(REPO, repo_type="dataset"):
        if f.endswith(".csv"):
            paths[Path(f).stem] = Path(hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW_DIR))
    return paths


def norm_no(value) -> str:
    s = str(value).strip()
    return s[:-2] if s.endswith(".0") else s


def is_repealed(text: str) -> bool:
    return len(text) < 250 and "mülga" in text.lower()


def build_corpus(corpus_csv: Path) -> list[dict]:
    df = pd.read_csv(corpus_csv)
    df["kanun_no"] = df["kanun_no"].map(norm_no)
    df["madde_no"] = df["madde_no"].map(norm_no)

    rows = []
    seen = set()
    for _, r in df[df["kanun_no"].isin(LAWS)].iterrows():
        law = LAWS[r["kanun_no"]]
        if law["maddeler"] and r["madde_no"] not in law["maddeler"]:
            continue
        text = re.sub(r"\s+", " ", str(r["context"])).strip()
        chunk = str(r["context_key"]).rsplit("chunk-", 1)[-1]
        madde_id = f"{r['kanun_no']}-{r['madde_no']}"
        doc_id = f"{madde_id}#{chunk}"
        if doc_id in seen or is_repealed(text):
            continue
        seen.add(doc_id)
        rows.append({
            "text": f"{law['kanun_adi']} Madde {r['madde_no']}: {text}",
            "metadata": {
                "id": doc_id,
                "madde_id": madde_id,
                "kanun_no": r["kanun_no"],
                "kanun_adi": law["kanun_adi"],
                "madde_no": r["madde_no"],
                "url": str(r.get("url", "")),
            },
        })
    return rows


def main():
    csvs = download_csvs()
    print("İndirilen dosyalar:", {k: str(v.relative_to(ROOT)) for k, v in csvs.items()})
    rows = build_corpus(csvs["statute_chunks"])
    Path(settings.CORPUS_PATH).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    n_madde = len({r["metadata"]["madde_id"] for r in rows})
    print(f"{n_madde} madde, {len(rows)} parça -> {Path(settings.CORPUS_PATH).relative_to(ROOT)}")

    from app.clients.vectorstore import get_vectorstore

    vs = get_vectorstore()
    vs.reset_collection()
    vs.add_documents(
        [Document(page_content=r["text"], metadata=r["metadata"]) for r in rows],
        ids=[r["metadata"]["id"] for r in rows],
    )
    print(f"Chroma'ya yazıldı: {settings.CHROMA_COLLECTION}")


if __name__ == "__main__":
    main()
