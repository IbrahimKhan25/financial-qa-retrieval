"""Golden check: the served pipeline must reproduce the offline dev numbers before anything is timed.

Runs all dev queries through Retriever in each mode and compares NDCG@10 with the manifest's expected values.
Exit code 1 on mismatch.
"""
import argparse
import json
import sys
from pathlib import Path

from finqa.data import fiqa
from finqa.eval import metrics
from finqa.serve.pipeline import Retriever
from finqa.serve.store import load_artifacts
from finqa.utils.hardware import describe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="serving_artifacts")
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/serving_parity_dev.json")
    args = ap.parse_args()

    art = load_artifacts(args.artifacts, args.threads)
    r = Retriever(art)
    d = fiqa.download()
    queries, qrels = fiqa.load_queries(d), fiqa.load_qrels(d, "dev")
    qids = sorted(qrels)
    expected = art.manifest["expected_dev_ndcg@10"]
    res, ok = {"split": "dev", "n_queries": len(qids), "tolerance": args.tol, "modes": {}, "hardware": describe(args.threads)}, True
    for mode in ("dense", "bm25", "hybrid"):
        ranking = {q: [h["doc_id"] for h in r.search(queries[q], k=100, mode=mode)["hits"]] for q in qids}
        got = metrics.aggregate(metrics.per_query(ranking, qrels))
        diff = got["ndcg@10"] - expected[mode]
        res["modes"][mode] = {"ndcg@10": got["ndcg@10"], "expected": expected[mode], "diff": diff, "recall@100": got["recall@100"], "mrr@10": got["mrr@10"]}
        flag = "OK" if abs(diff) <= args.tol else "MISMATCH"
        ok &= abs(diff) <= args.tol
        print(f"{mode:7s} ndcg@10 {got['ndcg@10']:.4f} expected {expected[mode]:.4f} diff {diff:+.4f}  {flag}")
    res["passed"] = ok
    Path(args.out).write_text(json.dumps(res, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
