"""BM25 over the full corpus (bm25s: Lucene-style scoring, no stemming)."""
import bm25s
import numpy as np


class BM25Retriever:
    def __init__(self, doc_ids: list[str], doc_texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = doc_ids
        self.params = {"k1": k1, "b": b, "stopwords": "en", "stemmer": None}
        self.index = bm25s.BM25(k1=k1, b=b)
        self.index.index(bm25s.tokenize(doc_texts, stopwords="en", show_progress=False),
                         show_progress=False)

    def search(self, query_texts: list[str], k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Returns (doc_index [n_q, k], scores [n_q, k]) best-first."""
        tokens = bm25s.tokenize(query_texts, stopwords="en", show_progress=False)
        idx, scores = self.index.retrieve(tokens, k=k, show_progress=False)
        return idx, scores
