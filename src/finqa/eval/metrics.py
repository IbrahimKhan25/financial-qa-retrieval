"""Per-query retrieval metrics. Definitions follow trec_eval (linear gain), as BEIR does."""
import math

# A ranking maps query_id -> doc_ids ordered best-first. Qrels map query_id -> {doc_id: relevance}.
Ranking = dict[str, list[str]]
Qrels = dict[str, dict[str, int]]

METRICS = ("ndcg@10", "recall@100", "mrr@10")


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((r for r in rel.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int = 100) -> float:
    relevant = {d for d, r in rel.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def per_query(ranking: Ranking, qrels: Qrels) -> dict[str, dict[str, float]]:
    """Scores for every query in qrels. A query missing from the ranking scores 0 (not skipped)."""
    out = {}
    for qid, rel in qrels.items():
        ranked = ranking.get(qid, [])
        out[qid] = {
            "ndcg@10": ndcg_at_k(ranked, rel),
            "recall@100": recall_at_k(ranked, rel),
            "mrr@10": mrr_at_k(ranked, rel),
        }
    return out


def aggregate(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    n = len(scores)
    return {m: sum(s[m] for s in scores.values()) / n for m in METRICS}
