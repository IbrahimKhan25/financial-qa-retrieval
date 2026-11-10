"""Serving artifacts: embeddings, BM25 index, memory-mapped document texts, ONNX models."""
import json
import mmap
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np

from finqa.serve.onnx_models import QuestionEncoder, Reranker


class TextStore:
    """Document texts in one memory-mapped file plus an offsets array, so RAM stays small."""

    def __init__(self, directory: Path):
        self.offsets = np.load(directory / "text_offsets.npy")
        self._file = open(directory / "texts.bin", "rb")  # noqa: SIM115 (kept open for the mmap's lifetime)
        self._blob = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, i: int, limit: int | None = None) -> str:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        if limit is not None:
            b = min(b, a + limit * 4)  # at most 4 bytes per character
        text = self._blob[a:b].decode("utf-8", errors="ignore")
        return text[:limit] if limit is not None else text


@dataclass
class Artifacts:
    manifest: dict
    doc_ids: list[str]
    index_of: dict[str, int]
    embeddings: np.ndarray
    bm25: bm25s.BM25
    texts: TextStore
    encoder: QuestionEncoder
    reranker: Reranker | None


def load_artifacts(directory: str | Path, threads: int = 4) -> Artifacts:
    d = Path(directory)
    manifest = json.loads((d / "manifest.json").read_text())
    doc_ids = json.loads((d / "doc_ids.json").read_text())
    ce = d / "cross_encoder" / "model.onnx"
    return Artifacts(
        manifest=manifest, doc_ids=doc_ids, index_of={x: i for i, x in enumerate(doc_ids)},
        embeddings=np.load(d / "doc_embeddings.npy"), bm25=bm25s.BM25.load(str(d / "bm25"), load_corpus=False),
        texts=TextStore(d),
        encoder=QuestionEncoder(d / "bi_encoder" / "model.onnx", d / "bi_encoder" / "tokenizer.json", threads),
        reranker=Reranker(ce, d / "cross_encoder" / "tokenizer.json", threads) if ce.exists() else None)
