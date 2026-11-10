"""Rerank-depth logic."""


def rerank_top(candidates: list[str], scores: list[float], depth: int) -> list[str]:
    """Reorder the first `depth` candidates by score (best first, stable); keep the rest in order.

    Keeping the tail means a depth-20 rerank still returns a full list, so Recall@100 stays
    comparable across depths. Quality at depth d is what a system that only paid to score d
    pairs per query would achieve.
    """
    depth = min(depth, len(candidates))
    head = sorted(range(depth), key=lambda i: -scores[i])
    return [candidates[i] for i in head] + list(candidates[depth:])
