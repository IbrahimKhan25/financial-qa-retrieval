"""Mine hard negatives for the TRAIN queries, from BM25 or from a fine-tuned checkpoint.

Dev and test positives are excluded from the pool (see finqa.train.negatives). Writes
{"meta": ..., "negatives": {query_id: [doc ids]}}.
"""
import argparse
import json
from pathlib import Path

from finqa.data import fiqa
from finqa.eval.dense_eval import search
from finqa.retrieval.bm25 import BM25Retriever
from finqa.train.model import BiEncoder, pick_device
from finqa.train.negatives import forbidden_doc_ids, mine_hard_negatives
from finqa.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bm25", "dense"], required=True)
    ap.add_argument("--checkpoint", help="BiEncoder dir (required for --source dense)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=100, help="candidates retrieved per query")
    ap.add_argument("--window", type=int, nargs=2, default=[10, 100], metavar=("LO", "HI"),
                    help="sample from ranks LO..HI (skips the top, where false negatives hide)")
    ap.add_argument("--max-per-query", type=int, default=20)
    ap.add_argument("--exclude-train-positives", action="store_true",
                    help="also drop docs that are positives for ANY train query (false-negative filter)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    train_qrels = fiqa.load_qrels(d, "train")
    held_out = [fiqa.load_qrels(d, "dev"), fiqa.load_qrels(d, "test")]
    forbidden = forbidden_doc_ids(*held_out)
    if args.exclude_train_positives:
        forbidden |= forbidden_doc_ids(train_qrels)
    doc_ids = list(corpus)
    doc_texts = [fiqa.doc_text(corpus[i]) for i in doc_ids]
    qids = sorted(train_qrels)
    q_texts = [queries[q] for q in qids]

    if args.source == "bm25":
        idx, _ = BM25Retriever(doc_ids, doc_texts).search(q_texts, k=args.k)
    else:
        assert args.checkpoint, "--checkpoint is required for --source dense"
        device = pick_device()
        model = BiEncoder(args.checkpoint).to(device)
        idx = search(model.encode(q_texts), model.encode(doc_texts), k=args.k, device=device)
    ranked = {q: [doc_ids[j] for j in row] for q, row in zip(qids, idx)}

    negs = mine_hard_negatives(ranked, train_qrels, forbidden, tuple(args.window), args.max_per_query)

    pos_all = {d for rel in train_qrels.values() for d in rel}
    leaked = {d for docs in negs.values() for d in docs} & forbidden
    assert not leaked, f"{len(leaked)} forbidden docs in the negative pool"
    held_out_docs = forbidden_doc_ids(*held_out)
    assert not {d for docs in negs.values() for d in docs} & held_out_docs, "dev/test positive leaked"
    counts = [len(v) for v in negs.values()]
    meta = {"source": args.source, "checkpoint": args.checkpoint, "k": args.k,
            "window": args.window, "max_per_query": args.max_per_query, "seed": args.seed,
            "exclude_train_positives": args.exclude_train_positives,
            "n_queries": len(negs), "queries_with_negatives": sum(1 for c in counts if c),
            "mean_negatives_per_query": round(sum(counts) / len(counts), 2),
            "forbidden_docs_excluded": len(forbidden)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "negatives": negs}))
    print(json.dumps(meta, indent=1))
    print("negatives that are train positives of other queries:",
          sum(1 for docs in negs.values() for d in docs if d in pos_all))


if __name__ == "__main__":
    main()
