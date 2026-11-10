"""Training pairs, batching, and hard-negative sampling."""
import json
import random
from pathlib import Path


def positives_of(qrels: dict[str, dict[str, int]]) -> dict[str, set[str]]:
    return {q: {d for d, r in rel.items() if r > 0} for q, rel in qrels.items()}


def build_pairs(qrels: dict[str, dict[str, int]]) -> list[tuple[str, str]]:
    """One (query_id, positive_doc_id) pair per relevant judgement, in a fixed order."""
    return [(q, d) for q in sorted(qrels) for d in sorted(qrels[q]) if qrels[q][d] > 0]


def make_batches(pairs: list[tuple[str, str]], batch_size: int, seed: int) -> list[list[int]]:
    """Shuffle, then pack into batches so no query appears twice in a batch.

    A query with several positives would otherwise meet its own other positive as an
    in-batch negative. Each example goes in the first batch that has room and no copy of
    its query.
    """
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    batches: list[list[int]] = []
    seen: list[set[str]] = []
    for i in order:
        q = pairs[i][0]
        for b, s in zip(batches, seen):
            if len(b) < batch_size and q not in s:
                b.append(i)
                s.add(q)
                break
        else:
            batches.append([i])
            seen.append({q})
    return batches


def load_negatives(paths: list[str]) -> dict[str, list[str]]:
    """Union of several mined-negative files (query_id -> doc ids), order kept, no duplicates."""
    merged: dict[str, list[str]] = {}
    for p in paths:
        negs = json.loads(Path(p).read_text())["negatives"]
        for q, docs in negs.items():
            cur = merged.setdefault(q, [])
            cur.extend(d for d in docs if d not in cur)
    return merged


def sample_group(qid: str, pos_doc: str, negatives: dict[str, list[str]], positives: set[str],
                 all_doc_ids: list[str], rng: random.Random, n_neg: int, n_random: int = 0) -> list[str]:
    """[positive] + n_neg distinct negatives for the cross-encoder's group softmax (n_random of them random).

    Mined negatives are used first; any shortfall is filled with random corpus docs. No negative
    is ever a known positive of the query.
    """
    pool = [d for d in negatives.get(qid, []) if d not in positives]
    negs = rng.sample(pool, min(n_neg - n_random, len(pool)))  # the rest are random (easy) corpus docs
    while len(negs) < n_neg:
        d = rng.choice(all_doc_ids)
        if d not in positives and d not in negs:
            negs.append(d)
    return [pos_doc, *negs]


def sample_negative(qid: str, negatives: dict[str, list[str]], positives: set[str],
                    all_doc_ids: list[str], rng: random.Random) -> str:
    """A mined hard negative if the query has any, else a random corpus doc that is not a positive."""
    pool = negatives.get(qid)
    if pool:
        return rng.choice(pool)
    while True:
        d = rng.choice(all_doc_ids)
        if d not in positives:
            return d
