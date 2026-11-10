"""Helpers that write serving artifacts (used by the build script and the tests). No PyTorch."""
from pathlib import Path

import numpy as np

from finqa.retrieval.bm25 import BM25Retriever


def write_text_store(texts: list[str], directory: Path) -> None:
    """Concatenate UTF-8 texts into texts.bin and record byte offsets, for memory-mapped reads."""
    offsets = [0]
    with open(directory / "texts.bin", "wb") as f:
        for t in texts:
            b = t.encode("utf-8")
            f.write(b)
            offsets.append(offsets[-1] + len(b))
    np.save(directory / "text_offsets.npy", np.array(offsets, dtype=np.int64))


def save_bm25(doc_ids: list[str], doc_texts: list[str], directory: Path) -> None:
    """Same tokenisation and parameters as the offline evaluation (en stopwords, no stemming, k1 1.5, b 0.75)."""
    BM25Retriever(doc_ids, doc_texts).index.save(str(directory / "bm25"))
