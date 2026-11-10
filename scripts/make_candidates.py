"""Build first-stage candidate lists (top-100) for the DEV queries and report candidate recall.

Candidate recall at each rerank depth is the ceiling for a reranker, so this is how the first
stage is chosen (on dev). Test is not allowed.
"""
import argparse
import hashlib
import json
from pathlib import Path

from finqa.data import fiqa
from finqa.eval import metrics
from finqa.retrieval.first_stage import bm25_lists, dense_lists, hybrid_lists
from finqa.train.model import pick_device
from finqa.utils.seed import set_seed

DEPTHS = (10, 20, 50, 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-stage", choices=["dense", "hybrid"], required=True)
    ap.add_argument("--checkpoint", default="checkpoints/bi_encoder_D_mined")
    ap.add_argument("--alpha", type=float, default=0.8, help="hybrid weight on dense (chosen on dev)")
    ap.add_argument("--split", default="dev", choices=["dev"])  # test is deliberately not allowed
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    device = pick_device()

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    qrels = fiqa.load_qrels(d, args.split)
    doc_ids = list(corpus)
    doc_texts = [fiqa.doc_text(corpus[i]) for i in doc_ids]
    qids = sorted(qrels)
    q_texts = [queries[q] for q in qids]

    h = hashlib.sha256((Path(args.checkpoint) / "model.safetensors").read_bytes()).hexdigest()[:10]
    dense = dense_lists(args.checkpoint, f"finetuned-{h}", doc_ids, doc_texts, q_texts, device)
    if args.first_stage == "dense":
        lists = dense
    else:
        lists = hybrid_lists(dense, bm25_lists(doc_ids, doc_texts, q_texts), args.alpha)
    cands = {q: [x if isinstance(x, str) else x[0] for x in lst] for q, lst in zip(qids, lists)}

    recall = {}
    for k in DEPTHS:
        recall[k] = sum(metrics.recall_at_k(cands[q][:k], qrels[q], k) for q in qids) / len(qids)
    ndcg = metrics.aggregate(metrics.per_query(cands, qrels))["ndcg@10"]
    meta = {"first_stage": args.first_stage, "checkpoint": args.checkpoint, "checkpoint_sha256_10": h,
            "alpha": args.alpha if args.first_stage == "hybrid" else None, "split": args.split,
            "n_queries": len(qids), "candidate_recall_at": {str(k): round(v, 4) for k, v in recall.items()},
            "ndcg@10_of_candidate_order": round(ndcg, 4)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "candidates": cands}))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
