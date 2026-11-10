"""Hybrid retrieval (BM25 + dense) on dev, with an alpha sweep and paired comparisons.

Dev only: alpha is tuned on dev (one parameter), so the selected alpha's dev score is slightly
optimistic; the whole alpha curve is saved so sensitivity can be shown. Test is never touched here.
Runs for two dense models: the fine-tuned checkpoint (row D) and the zero-shot MiniLM.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from finqa.data import fiqa
from finqa.eval import metrics
from finqa.eval.dense_eval import encode_cached, search_scores
from finqa.eval.stats import bootstrap_ci, paired_bootstrap
from finqa.retrieval.bm25 import BM25Retriever
from finqa.retrieval.hybrid import FLOOR, rrf, weighted
from finqa.train.model import BiEncoder, pick_device
from finqa.utils.hardware import describe
from finqa.utils.seed import set_seed

BASE = "sentence-transformers/all-MiniLM-L6-v2"
K, RRF_K = 100, 60
ALPHAS = [round(0.1 * i, 1) for i in range(11)]


def dense_lists(model_path, key, doc_texts, doc_ids, q_texts, device):
    model = BiEncoder(model_path).to(device)
    d_emb = encode_cached(model, doc_texts, key)
    q_emb = model.encode(q_texts)
    idx, sc = search_scores(q_emb, d_emb, k=K, device=device)
    del model
    return [[(doc_ids[j], float(s)) for j, s in zip(row, srow)] for row, srow in zip(idx, sc)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/bi_encoder_D_mined")
    ap.add_argument("--split", default="dev", choices=["dev"])  # test is deliberately not allowed
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
    print(f"{args.split}: {len(qids)} queries, {len(doc_ids)} docs, device={device}", flush=True)

    idx, sc = BM25Retriever(doc_ids, doc_texts).search(q_texts, k=K)
    bm25 = [[(doc_ids[j], float(s)) for j, s in zip(row, srow)] for row, srow in zip(idx, sc)]
    ckpt_hash = hashlib.sha256((Path(args.checkpoint) / "model.safetensors").read_bytes()).hexdigest()[:10]
    dense = {
        "D": dense_lists(args.checkpoint, f"finetuned-{ckpt_hash}", doc_texts, doc_ids, q_texts, device),
        "zs": dense_lists(BASE, "zeroshot-all-MiniLM-L6-v2", doc_texts, doc_ids, q_texts, device),
    }
    if device == "mps":
        torch.mps.empty_cache()

    runs = {"bm25": {q: [x for x, _ in lst] for q, lst in zip(qids, bm25)}}
    curve = {}
    for tag, lists in dense.items():
        runs[f"dense_{tag}"] = {q: [x for x, _ in lst] for q, lst in zip(qids, lists)}
        runs[f"rrf_{tag}"] = {q: rrf([[x for x, _ in dl], [x for x, _ in bl]], k=RRF_K, top=K)
                              for q, dl, bl in zip(qids, lists, bm25)}
        curve[tag] = []
        for a in ALPHAS:
            runs[f"weighted_{tag}_a{a}"] = {q: weighted(dl, bl, a, top=K)
                                            for q, dl, bl in zip(qids, lists, bm25)}
            agg = metrics.aggregate(metrics.per_query(runs[f"weighted_{tag}_a{a}"], qrels))
            curve[tag].append({"alpha": a, **agg})
    scores = {name: metrics.per_query(r, qrels) for name, r in runs.items()}

    best = {tag: max(curve[tag], key=lambda r: r["ndcg@10"])["alpha"] for tag in dense}
    selected = {"bm25": "bm25", "dense_D": "dense_D", "dense_zs": "dense_zs", "rrf_D": "rrf_D",
                "rrf_zs": "rrf_zs", f"weighted_D (alpha={best['D']})": f"weighted_D_a{best['D']}",
                f"weighted_zs (alpha={best['zs']})": f"weighted_zs_a{best['zs']}"}

    # Sanity: the pieces must reproduce numbers we already measured.
    base = json.loads(Path("results/baselines_dev.json").read_text())["systems"]
    train_d = json.loads(Path("results/train_bi_encoder_D_mined.json").read_text())
    ndcg = lambda name: metrics.aggregate(scores[name])["ndcg@10"]
    sanity = {"bm25_vs_phase1": ndcg("bm25") - base["bm25"]["metrics"]["ndcg@10"]["mean"],
              "dense_zs_vs_phase1": ndcg("dense_zs") - base["minilm_zeroshot"]["metrics"]["ndcg@10"]["mean"],
              "dense_D_vs_colab_row_D": ndcg("dense_D") - train_d["metrics"]["ndcg@10"]["mean"]}
    print("sanity (NDCG@10 differences vs earlier runs):", {k: round(v, 4) for k, v in sanity.items()}, flush=True)

    def paired(a, b):
        ka, kb = selected[a], selected[b]
        qs = sorted(scores[ka])
        return {m: paired_bootstrap([scores[kb][q][m] for q in qs], [scores[ka][q][m] for q in qs]) for m in metrics.METRICS}

    pairs = [("rrf_D", "dense_D"), (f"weighted_D (alpha={best['D']})", "dense_D"), ("rrf_D", "bm25"),
             (f"weighted_D (alpha={best['D']})", "bm25"), (f"weighted_D (alpha={best['D']})", "rrf_D"),
             ("rrf_zs", "dense_zs"), (f"weighted_zs (alpha={best['zs']})", "dense_zs")]
    result = {
        "split": args.split, "n_queries": len(qids), "seed": args.seed, "device": device,
        "config": {"k": K, "rrf_k": RRF_K, "alphas": ALPHAS, "minmax_floor": FLOOR,
                   "checkpoint": args.checkpoint, "checkpoint_sha256_10": ckpt_hash,
                   "alpha_selected_on": "dev ndcg@10 (optimistic by selection)"},
        "best_alpha": best, "alpha_curve": curve, "sanity_ndcg_diffs": sanity,
        "systems": {label: {"metrics": {m: bootstrap_ci([scores[key][q][m] for q in qids]) for m in metrics.METRICS},
                            "per_query": scores[key]} for label, key in selected.items()},
        "paired": {f"{a} minus {b}": paired(a, b) for a, b in pairs},
        "hardware": {**describe(), "device": device, "torch": torch.__version__},
    }
    out = Path("results/hybrid_dev.json")
    out.write_text(json.dumps(result, indent=1))

    print(f"\nwrote {out}\n")
    for label, s in result["systems"].items():
        m = s["metrics"]
        print(f"{label:28s} ndcg {m['ndcg@10']['mean']:.4f} [{m['ndcg@10']['ci_low']:.4f},{m['ndcg@10']['ci_high']:.4f}] "
              f"recall@100 {m['recall@100']['mean']:.4f} mrr {m['mrr@10']['mean']:.4f}")
    print("\npaired differences (first minus second):")
    for k, v in result["paired"].items():
        n, r = v["ndcg@10"], v["recall@100"]
        print(f"  {k:55s} ndcg {n['diff']:+.4f} [{n['ci_low']:+.4f},{n['ci_high']:+.4f}]  recall@100 {r['diff']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]")
    print("\nalpha curve (ndcg@10 | recall@100):")
    for tag, rows in curve.items():
        print(f"  {tag}: " + "  ".join(f"{r['alpha']}:{r['ndcg@10']:.3f}|{r['recall@100']:.3f}" for r in rows))


if __name__ == "__main__":
    main()
