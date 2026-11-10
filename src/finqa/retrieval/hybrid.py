"""Hybrid retrieval: fuse a lexical (BM25) ranking with a dense ranking.

Inputs are per-query lists of (doc_id, score) in best-first order. Two fusions:
  - rrf: reciprocal rank fusion, uses ranks only, so no score scaling is needed.
  - weighted: per-list min-max scaling to [floor, 1], then alpha * dense + (1 - alpha) * bm25.
"""

FLOOR = 1e-3  # any retrieved doc scores at least this, so "retrieved" always beats "not retrieved"


def rrf(rankings: list[list[str]], k: int = 60, top: int = 100) -> list[str]:
    """Each doc scores sum over lists of 1 / (k + rank), rank starting at 1."""
    scores: dict[str, float] = {}
    for docs in rankings:
        for rank, d in enumerate(docs, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])[:top]  # stable: ties keep first-seen order


def minmax(items: list[tuple[str, float]], floor: float = FLOOR) -> dict[str, float]:
    """Scale scores to [floor, 1]: the worst retrieved doc gets floor, the best gets 1."""
    if not items:
        return {}
    vals = [s for _, s in items]
    lo, span = min(vals), max(vals) - min(vals)
    return {d: floor + (1 - floor) * ((s - lo) / span if span > 0 else 1.0) for d, s in items}


def weighted(dense: list[tuple[str, float]], lexical: list[tuple[str, float]],
             alpha: float, top: int = 100) -> list[str]:
    """alpha = 1 reproduces the dense ranking and alpha = 0 the lexical one.

    A doc missing from one list scores 0 from that system.
    """
    nd, nl = minmax(dense), minmax(lexical)
    docs = list(dict.fromkeys([d for d, _ in dense] + [d for d, _ in lexical]))
    fused = {d: alpha * nd.get(d, 0.0) + (1 - alpha) * nl.get(d, 0.0) for d in docs}
    return sorted(docs, key=lambda d: -fused[d])[:top]
