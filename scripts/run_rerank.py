"""Rerank dev candidates with a cross-encoder: depth sweep (10/20/50/100) and CPU latency per depth.

Dev only. Each (query, doc) pair is scored once per cross-encoder and reused across depths and
first stages. Latency is measured on CPU with the thread count printed beside it; Docker numbers
come later. Test is not allowed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from finqa.data import fiqa
from finqa.eval import metrics
from finqa.eval.stats import bootstrap_ci, paired_bootstrap
from finqa.retrieval.rerank import rerank_top
from finqa.train.cross_encoder import CrossEncoder
from finqa.train.model import pick_device
from finqa.utils.hardware import describe
from finqa.utils.seed import set_seed

DEPTHS = (10, 20, 50, 100)


def score_pairs(ce, name_key, queries, text, cand_sets, cache_dir="data/cache"):
    """Score the union of (qid, doc) pairs across candidate sets, with an on-disk cache."""
    path = Path(cache_dir) / f"ce_scores_{name_key}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    need = sorted({(q, x) for cands in cand_sets.values() for q, lst in cands.items() for x in lst
                   if f"{q}|{x}" not in cache})
    if need:
        sc = ce.score([queries[q] for q, _ in need], [text[x] for _, x in need])
        cache.update({f"{q}|{x}": float(s) for (q, x), s in zip(need, sc)})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    return cache


def latency(ce, queries, text, cands, qids, threads, n_queries):
    """Per-query time to tokenize and score `depth` pairs in one batch, on CPU (ms)."""
    torch.set_num_threads(threads)
    out = {}
    for depth in DEPTHS:
        times = []
        for q in qids[:n_queries + 3]:
            qs, ds = [queries[q]] * depth, [text[x] for x in cands[q][:depth]]
            t = time.perf_counter()
            ce.score(qs, ds, batch_size=depth)
            times.append((time.perf_counter() - t) * 1000)
        times = np.array(times[3:])  # first 3 queries are warm-up
        out[depth] = {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                      "mean_ms": float(times.mean()), "n_queries": len(times)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--offshelf", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--finetuned", help="fine-tuned CrossEncoder dir (optional)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--latency-queries", type=int, default=100)
    ap.add_argument("--no-latency", action="store_true", help="skip the CPU latency benchmark")
    ap.add_argument("--out", default="results/rerank_dev.json")
    ap.add_argument("--split", default="dev", choices=["dev"])  # test is deliberately not allowed
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    device = pick_device()

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    qrels = fiqa.load_qrels(d, args.split)
    text = {i: fiqa.doc_text(corpus[i]) for i in corpus}
    qids = sorted(qrels)
    cand_sets = {n: json.loads(Path(p).read_text())["candidates"] for n, p in (a.split("=", 1) for a in args.candidates)}
    ce_specs = {"offshelf": args.offshelf}
    if args.finetuned:
        ce_specs["finetuned"] = args.finetuned

    res = {"split": args.split, "n_queries": len(qids), "depths": DEPTHS, "first_stages": {}, "paired": {}}
    per_q = {}
    for stage, cands in cand_sets.items():
        r0 = metrics.per_query({q: cands[q] for q in qids}, qrels)
        per_q[(stage, "none", 100)] = r0
        res["first_stages"][stage] = {"none": {"metrics": {m: bootstrap_ci([r0[q][m] for q in qids]) for m in metrics.METRICS},
                                               "per_query": r0}}
    for ce_name, spec in ce_specs.items():
        t0 = time.time()
        ce = CrossEncoder(spec).to(device)
        key = hashlib.sha256((Path(spec) / "model.safetensors").read_bytes()).hexdigest()[:10] \
            if Path(spec).exists() else spec.replace("/", "_")
        cache = score_pairs(ce, key, queries, text, cand_sets)
        print(f"scored with {ce_name} ({spec}) on {device} in {time.time() - t0:.0f}s", flush=True)
        for stage, cands in cand_sets.items():
            for depth in DEPTHS:
                ranking = {q: rerank_top(cands[q], [cache[f"{q}|{x}"] for x in cands[q]], depth) for q in qids}
                per = metrics.per_query(ranking, qrels)
                per_q[(stage, ce_name, depth)] = per
                entry = {"metrics": {m: bootstrap_ci([per[q][m] for q in qids]) for m in metrics.METRICS}}
                if depth == 100:
                    entry["per_query"] = per
                res["first_stages"][stage].setdefault(ce_name, {})[depth] = entry
        del ce

    def pair(a, b):
        return {m: paired_bootstrap([per_q[b][q][m] for q in qids], [per_q[a][q][m] for q in qids]) for m in metrics.METRICS}

    for stage in cand_sets:
        base = (stage, "none", 100)
        for ce_name in ce_specs:
            for depth in DEPTHS:
                res["paired"][f"{stage}: {ce_name} depth {depth} minus no reranking"] = pair((stage, ce_name, depth), base)
        if "finetuned" in ce_specs:
            res["paired"][f"{stage}: finetuned depth 100 minus offshelf depth 100"] = pair((stage, "finetuned", 100), (stage, "offshelf", 100))

    # CPU latency (architecture is identical for both cross-encoders; measure the fine-tuned one if present)
    if not args.no_latency:
        spec = ce_specs.get("finetuned", ce_specs["offshelf"])
        stage0 = next(iter(cand_sets))
        cpu_model = CrossEncoder(spec).to("cpu")
        res["latency_cpu"] = {"model": spec, "per_depth": latency(cpu_model, queries, text, cand_sets[stage0], qids,
                                                                    args.threads, args.latency_queries),
                              "precision": "fp32 PyTorch", "includes": "tokenization + forward, one query, one batch",
                              "hardware": {**describe(num_threads=args.threads), "torch": torch.__version__}}
    res["hardware"] = {**describe(), "scoring_device": device, "torch": torch.__version__}
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=1))

    print(f"\nwrote {out}\n")
    for stage in cand_sets:
        n = res["first_stages"][stage]["none"]["metrics"]
        print(f"[{stage}] no reranking: ndcg {n['ndcg@10']['mean']:.4f} recall@100 {n['recall@100']['mean']:.4f} mrr {n['mrr@10']['mean']:.4f}")
        for ce_name in ce_specs:
            for depth in DEPTHS:
                m = res["first_stages"][stage][ce_name][depth]["metrics"]
                print(f"[{stage}] {ce_name:9s} depth {depth:3d}: ndcg {m['ndcg@10']['mean']:.4f} [{m['ndcg@10']['ci_low']:.4f},"
                      f"{m['ndcg@10']['ci_high']:.4f}] mrr {m['mrr@10']['mean']:.4f} recall@100 {m['recall@100']['mean']:.4f}")
    if "latency_cpu" in res:
        print("\nCPU latency per query (ms):", {k: (round(v["p50_ms"]), round(v["p95_ms"])) for k, v in res["latency_cpu"]["per_depth"].items()},
              "threads", args.threads, res["latency_cpu"]["hardware"]["cpu_model"])


if __name__ == "__main__":
    main()
