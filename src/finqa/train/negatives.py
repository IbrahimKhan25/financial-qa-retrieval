"""Hard-negative mining with a leakage guard.

A hard negative is a document a retriever ranks highly for a train query but that is not
labelled relevant. Two protections:
  1. dev and test positives are never allowed in the pool, so a held-out answer can never
     be taught as "wrong" (no leakage);
  2. we skip the very top ranks, where unlabeled-but-good answers (false negatives) are most
     likely, and sample from a window below them.
"""


def forbidden_doc_ids(*held_out_qrels: dict[str, dict[str, int]]) -> set[str]:
    """Every doc that is a positive for any held-out (dev or test) query."""
    return {d for qrels in held_out_qrels for rel in qrels.values() for d, r in rel.items() if r > 0}


def mine_hard_negatives(ranked: dict[str, list[str]], train_qrels: dict[str, dict[str, int]],
                        forbidden: set[str], window: tuple[int, int] = (10, 100),
                        max_per_query: int = 20) -> dict[str, list[str]]:
    """ranked: query_id -> doc ids best-first. Returns query_id -> candidate negatives."""
    lo, hi = window
    out: dict[str, list[str]] = {}
    for qid, docs in ranked.items():
        own = {d for d, r in train_qrels.get(qid, {}).items() if r > 0}
        cands = [d for d in docs[lo:hi] if d not in own and d not in forbidden]
        out[qid] = cands[:max_per_query]
    return out
