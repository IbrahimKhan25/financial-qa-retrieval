"""Error analysis on dev: which questions does the served hybrid retriever fail on, and what do they look like?

A failure is a question with no relevant document in the top 10 (NDCG@10 = 0). The script counts failures, splits
them by how far down the best relevant document sits, and writes a seeded random sample with the question, the
relevant documents and the top results, for reading and labelling by hand. Dev split only.

  uv run python scripts/error_analysis.py
"""
import argparse
import json
import random
from pathlib import Path
from statistics import mean

from finqa.data import fiqa
from finqa.eval import metrics

SYSTEMS = {"hybrid": "dev_hybrid_D.json", "dense": "dev_dense_D.json", "bm25": "dev_bm25.json"}


def best_rank(ranked: list[str], rel: dict[str, int]) -> int | None:
    """1-based rank of the highest-placed relevant document, None if it is not in the list."""
    ranks = [ranked.index(d) + 1 for d in rel if d in ranked]
    return min(ranks) if ranks else None


def bucket(rank: int | None) -> str:
    if rank is None:
        return "not in top 100"
    return "rank 11-20" if rank <= 20 else "rank 21-100"


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " ..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/error_analysis_dev.json")
    ap.add_argument("--labels", default="results/error_analysis_dev_labels.json")
    args = ap.parse_args()

    d = fiqa.download()
    queries, qrels, corpus = fiqa.load_queries(d), fiqa.load_qrels(d, "dev"), fiqa.load_corpus(d)
    runs = {name: json.loads((Path(args.candidates) / f).read_text())["candidates"] for name, f in SYSTEMS.items()}
    scores = metrics.per_query(runs["hybrid"], qrels)
    failed = sorted(q for q in scores if scores[q]["ndcg@10"] == 0)
    ok = [q for q in scores if q not in set(failed)]

    def words(q):
        return len(queries[q].split())

    buckets: dict[str, int] = {}
    for q in failed:
        b = bucket(best_rank(runs["hybrid"][q], qrels[q]))
        buckets[b] = buckets.get(b, 0) + 1
    summary = {
        "split": "dev", "system": "hybrid (alpha 0.8), top 100", "n_queries": len(scores), "n_failed": len(failed),
        "failed_by_best_relevant_rank": buckets,
        "failed_but_dense_alone_has_a_hit_in_top10": sum((best_rank(runs["dense"][q], qrels[q]) or 999) <= 10 for q in failed),
        "failed_but_bm25_alone_has_a_hit_in_top10": sum((best_rank(runs["bm25"][q], qrels[q]) or 999) <= 10 for q in failed),
        "mean_relevant_docs": {"failed": mean(len(qrels[q]) for q in failed), "not_failed": mean(len(qrels[q]) for q in ok)},
        "mean_question_words": {"failed": mean(words(q) for q in failed), "not_failed": mean(words(q) for q in ok)},
    }

    picked = random.Random(args.seed).sample(failed, min(args.sample, len(failed)))
    sample = []
    for q in picked:
        sample.append({
            "query_id": q, "question": queries[q],
            "best_relevant_rank": {name: best_rank(runs[name][q], qrels[q]) for name in SYSTEMS},
            "relevant": [{"doc_id": doc, "words": len(fiqa.doc_text(corpus[doc]).split()),
                          "text": clip(fiqa.doc_text(corpus[doc]), 700)} for doc in qrels[q]],
            "top3": [{"doc_id": doc, "text": clip(fiqa.doc_text(corpus[doc]), 400)} for doc in runs["hybrid"][q][:3]],
        })
    # Causes assigned by reading the sample are kept in their own file, so rerunning this script never loses them.
    labels_path = Path(args.labels)
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())["labels"]
        missing = [s["query_id"] for s in sample if s["query_id"] not in labels]
        if missing:
            raise SystemExit(f"{labels_path} has no label for sampled queries {missing} (different seed or sample size?)")
        counts: dict[str, int] = {}
        for s in sample:
            s["cause"], s["cause_note"] = labels[s["query_id"]]
            counts[s["cause"]] = counts.get(s["cause"], 0) + 1
        summary["sample_causes"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    Path(args.out).write_text(json.dumps({"summary": summary, "sample_seed": args.seed, "sample": sample}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
