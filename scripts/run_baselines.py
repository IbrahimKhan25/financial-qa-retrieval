"""Baselines (BM25, off-the-shelf bi-encoder) on one split, with bootstrap CIs.

Phase 1 runs on dev only. The test split is reserved for a single final evaluation script.
"""
import argparse
import json
import time
from pathlib import Path

from finqa.data import fiqa
from finqa.eval import metrics
from finqa.eval.stats import bootstrap_ci, paired_bootstrap
from finqa.retrieval.bm25 import BM25Retriever
from finqa.retrieval.dense import DenseRetriever
from finqa.utils.hardware import describe
from finqa.utils.seed import set_seed
from finqa.utils.tracking import log_metrics, start_run

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def to_ranking(qids, doc_ids, idx):
    return {q: [doc_ids[j] for j in row] for q, row in zip(qids, idx)}


def summarize(scores):
    qids = sorted(scores)
    out = {}
    for m in metrics.METRICS:
        out[m] = bootstrap_ci([scores[q][m] for q in qids])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev"])  # test is deliberately not allowed
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    qrels = fiqa.load_qrels(d, args.split)
    doc_ids = list(corpus)
    doc_texts = [fiqa.doc_text(corpus[i]) for i in doc_ids]
    qids = sorted(qrels)
    q_texts = [queries[q] for q in qids]
    print(f"{args.split}: {len(qids)} queries, {len(doc_ids)} docs")

    results = {"split": args.split, "seed": args.seed, "n_queries": len(qids), "systems": {}}

    with start_run("baselines", f"baselines-{args.split}", {"split": args.split, "seed": args.seed}):
        t = time.time()
        bm25 = BM25Retriever(doc_ids, doc_texts)
        idx, _ = bm25.search(q_texts, k=100)
        results["systems"]["bm25"] = {"params": bm25.params, "wall_s": round(time.time() - t, 1)}
        runs = {"bm25": to_ranking(qids, doc_ids, idx)}

        t = time.time()
        dense = DenseRetriever(MODEL)
        tok = dense.model.tokenizer
        max_len = dense.model.max_seq_length
        lens = [len(x) for x in tok(doc_texts, add_special_tokens=True, truncation=False)["input_ids"]]
        n_trunc = sum(1 for n in lens if n > max_len)
        d_emb = dense.encode_cached(doc_texts, f"data/cache/{MODEL.split('/')[-1]}_docs.npy")
        q_emb = dense.encode(q_texts)
        idx, _ = DenseRetriever.top_k(q_emb, d_emb, k=100)
        results["systems"]["minilm_zeroshot"] = {
            "model": MODEL, "device": dense.device, "max_seq_length": max_len,
            "docs_truncated": n_trunc, "docs_truncated_frac": round(n_trunc / len(lens), 4),
            "wall_s_incl_cache": round(time.time() - t, 1)}
        runs["minilm_zeroshot"] = to_ranking(qids, doc_ids, idx)

        per_sys = {}
        for name, ranking in runs.items():
            scores = metrics.per_query(ranking, qrels)
            per_sys[name] = scores
            results["systems"][name]["metrics"] = summarize(scores)
            results["systems"][name]["per_query"] = scores
            log_metrics({f"{name}/{m}": v["mean"] for m, v in results["systems"][name]["metrics"].items()})

        results["paired_minilm_vs_bm25"] = {
            m: paired_bootstrap([per_sys["bm25"][q][m] for q in qids],
                                [per_sys["minilm_zeroshot"][q][m] for q in qids])
            for m in metrics.METRICS}
        results["hardware"] = describe()

    out = Path("results") / f"baselines_{args.split}.json"
    out.write_text(json.dumps(results, indent=1))
    for name in runs:
        print(name)
        for m, v in results["systems"][name]["metrics"].items():
            print(f"  {m:11s} {v['mean']:.4f}  95% CI [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    print("paired (minilm - bm25):")
    for m, v in results["paired_minilm_vs_bm25"].items():
        print(f"  {m:11s} {v['diff']:+.4f}  95% CI [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  p={v['p_value']:.4f}")
    print("truncated docs:", results["systems"]["minilm_zeroshot"]["docs_truncated_frac"])


if __name__ == "__main__":
    main()
