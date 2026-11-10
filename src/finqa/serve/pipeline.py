"""The retrieval pipeline served by the API: encode, dense search, BM25, fusion, optional rerank."""
from time import perf_counter

import bm25s
import numpy as np

from finqa.retrieval.hybrid import weighted_scored
from finqa.serve.store import Artifacts

SNIPPET_CHARS = 300


class Retriever:
    def __init__(self, art: Artifacts, cand_k: int = 100):
        self.art = art
        self.cand_k = cand_k
        self.default_alpha = float(art.manifest.get("alpha", 0.8))

    def _dense(self, q: np.ndarray, c: int) -> list[tuple[str, float]]:
        scores = self.art.embeddings @ q
        idx = np.argpartition(-scores, c - 1)[:c]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.art.doc_ids[i], float(scores[i])) for i in idx]

    def _bm25(self, query: str, c: int) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        if not tokens.ids[0]:  # every word was a stopword: nothing for BM25 to match
            return []
        idx, sc = self.art.bm25.retrieve(tokens, k=c, show_progress=False)
        return [(self.art.doc_ids[i], float(s)) for i, s in zip(idx[0], sc[0])]

    def search(self, query: str, k: int = 10, mode: str = "hybrid", alpha: float | None = None,
               rerank: bool = False, rerank_depth: int = 20) -> dict:
        t0 = perf_counter()
        c = min(self.cand_k, len(self.art.doc_ids))
        timings: dict[str, float] = {}

        q = self.art.encoder.encode([query])[0]
        timings["encode_ms"] = (perf_counter() - t0) * 1000

        t = perf_counter()
        dense = self._dense(q, c) if mode in ("dense", "hybrid") else []
        timings["dense_ms"] = (perf_counter() - t) * 1000
        t = perf_counter()
        lexical = self._bm25(query, c) if mode in ("bm25", "hybrid") else []
        timings["bm25_ms"] = (perf_counter() - t) * 1000

        t = perf_counter()
        if mode == "dense":
            ranked = dense
        elif mode == "bm25":
            ranked = lexical
        else:
            ranked = weighted_scored(dense, lexical, self.default_alpha if alpha is None else alpha, top=c)
        timings["fuse_ms"] = (perf_counter() - t) * 1000

        rerank_scores: dict[str, float] = {}
        t = perf_counter()
        if rerank:
            depth = min(rerank_depth, len(ranked))
            head = ranked[:depth]
            sc = self.art.reranker.score(query, [self.art.texts.get(self.art.index_of[d]) for d, _ in head])
            rerank_scores = {d: float(s) for (d, _), s in zip(head, sc)}
            order = sorted(range(depth), key=lambda i: -sc[i])  # stable, as in the offline rerank_top
            ranked = [head[i] for i in order] + ranked[depth:]
        timings["rerank_ms"] = (perf_counter() - t) * 1000

        hits = [{"doc_id": d, "score": s, "rerank_score": rerank_scores.get(d),
                 "snippet": self.art.texts.get(self.art.index_of[d], limit=SNIPPET_CHARS)} for d, s in ranked[:k]]
        timings["total_ms"] = (perf_counter() - t0) * 1000
        return {"hits": hits, "timings_ms": {k_: round(v, 3) for k_, v in timings.items()}}
