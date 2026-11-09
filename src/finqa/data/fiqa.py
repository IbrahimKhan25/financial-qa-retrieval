"""Download and load BEIR FiQA-2018 (corpus, queries, qrels per split)."""
import csv
import hashlib
import json
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
SPLITS = ("train", "dev", "test")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir: str | Path = "data/raw") -> Path:
    """Fetch and extract the zip into data_dir/fiqa. Returns the dataset dir. Idempotent."""
    data_dir = Path(data_dir)
    out = data_dir / "fiqa"
    if (out / "corpus.jsonl").exists():
        return out
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "fiqa.zip"
    if not zip_path.exists():
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL, context=ctx) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f)
    print(f"fiqa.zip sha256={_sha256(zip_path)} bytes={zip_path.stat().st_size}")
    root = data_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():  # reject path traversal before extracting anything
            if not (root / name).resolve().is_relative_to(root):
                raise ValueError(f"unsafe path in zip: {name}")
        zf.extractall(data_dir)
    return out


def load_corpus(dataset_dir: str | Path) -> dict[str, dict]:
    """doc_id -> {"title": str, "text": str}"""
    corpus = {}
    with open(Path(dataset_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[str(d["_id"])] = {"title": d.get("title", ""), "text": d["text"]}
    return corpus


def load_queries(dataset_dir: str | Path) -> dict[str, str]:
    queries = {}
    with open(Path(dataset_dir) / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[str(d["_id"])] = d["text"]
    return queries


def load_qrels(dataset_dir: str | Path, split: str) -> dict[str, dict[str, int]]:
    """query_id -> {doc_id: relevance}"""
    qrels: dict[str, dict[str, int]] = {}
    with open(Path(dataset_dir) / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def doc_text(doc: dict) -> str:
    """Text fed to retrievers (BEIR convention: title + text)."""
    return f"{doc['title']} {doc['text']}".strip() if doc["title"] else doc["text"]


def split_stats(dataset_dir: str | Path) -> dict:
    """Sizes needed to verify the split before using it."""
    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    stats = {"n_docs": len(corpus), "n_queries_total": len(queries)}
    for s in SPLITS:
        q = load_qrels(dataset_dir, s)
        stats[s] = {"n_queries": len(q), "n_pairs": sum(len(v) for v in q.values())}
    return stats
