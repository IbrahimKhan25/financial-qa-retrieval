"""The one final evaluation on the official TEST split. Run once.

Everything here was fixed on dev before this script existed: the systems, the checkpoint (row D) and the hybrid
weight (alpha = 0.8). Nothing is tuned or selected on test. The script refuses to run if its output file already
exists, so a second run cannot quietly replace the first.

  uv run python scripts/final_test_eval.py
"""
import hashlib
import json
from pathlib import Path

import torch

from finqa.data import fiqa
from finqa.eval import metrics
from finqa.eval.stats import bootstrap_ci, paired_bootstrap
from finqa.retrieval.first_stage import bm25_lists, dense_lists, hybrid_lists
from finqa.train.model import pick_device
from finqa.utils.hardware import describe
from finqa.utils.seed import set_seed
from finqa.utils.tracking import log_metrics, start_run

SPLIT = "test"
BASE = "sentence-transformers/all-MiniLM-L6-v2"
CHECKPOINT = "checkpoints/bi_encoder_D_mined"
ALPHA = 0.8  # chosen on dev (results/hybrid_dev.json)
SEED = 42
OUT = Path("results/final_test.json")
# (first, second): reported as first minus second
PAIRS = [("minilm_zeroshot", "bm25"), ("dense_D", "minilm_zeroshot"), ("hybrid_D", "dense_D"), ("hybrid_D", "bm25")]


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists: the test split has been evaluated. Not running again.")
    set_seed(SEED)
    device = pick_device()

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    qrels = fiqa.load_qrels(d, SPLIT)
    doc_ids = list(corpus)
    doc_texts = [fiqa.doc_text(corpus[i]) for i in doc_ids]
    qids = sorted(qrels)
    q_texts = [queries[q] for q in qids]
    print(f"{SPLIT}: {len(qids)} queries, {len(doc_ids)} docs, device={device}", flush=True)

    ckpt_hash = hashlib.sha256((Path(CHECKPOINT) / "model.safetensors").read_bytes()).hexdigest()[:10]
    bm25 = bm25_lists(doc_ids, doc_texts, q_texts)
    zs = dense_lists(BASE, "zeroshot-all-MiniLM-L6-v2", doc_ids, doc_texts, q_texts, device)
    dense = dense_lists(CHECKPOINT, f"finetuned-{ckpt_hash}", doc_ids, doc_texts, q_texts, device)
    hybrid = hybrid_lists(dense, bm25, ALPHA)

    def ids(lists):
        return {q: [x if isinstance(x, str) else x[0] for x in lst] for q, lst in zip(qids, lists)}

    runs = {"bm25": ids(bm25), "minilm_zeroshot": ids(zs), "dense_D": ids(dense), "hybrid_D": ids(hybrid)}
    scores = {name: metrics.per_query(r, qrels) for name, r in runs.items()}

    result = {
        "split": SPLIT, "n_queries": len(qids), "seed": SEED, "device": device,
        "config": {"k": 100, "alpha": ALPHA, "checkpoint": CHECKPOINT, "checkpoint_sha256_10": ckpt_hash,
                   "selection": "systems, checkpoint and alpha were fixed on dev; nothing was selected on test"},
        "systems": {name: {"metrics": {m: bootstrap_ci([s[q][m] for q in qids]) for m in metrics.METRICS},
                           "per_query": s} for name, s in scores.items()},
        "paired": {f"{a} minus {b}": {m: paired_bootstrap([scores[b][q][m] for q in qids],
                                                           [scores[a][q][m] for q in qids])
                                      for m in metrics.METRICS} for a, b in PAIRS},
        "hardware": {**describe(), "device": device, "torch": torch.__version__},
    }
    with start_run("final-test", "final-test", {"split": SPLIT, "seed": SEED, "alpha": ALPHA,
                                                "checkpoint_sha256_10": ckpt_hash}):
        for name, s in result["systems"].items():
            log_metrics({f"{name}/{m}": v["mean"] for m, v in s["metrics"].items()})
    OUT.write_text(json.dumps(result, indent=1))

    print(f"\nwrote {OUT}\n")
    for name, s in result["systems"].items():
        print(name)
        for m, v in s["metrics"].items():
            print(f"  {m:11s} {v['mean']:.4f}  95% CI [{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    print("\npaired differences (first minus second):")
    for k, v in result["paired"].items():
        print(f"  {k}")
        for m, x in v.items():
            print(f"    {m:11s} {x['diff']:+.4f}  95% CI [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]  p={x['p_value']:.4f}")


if __name__ == "__main__":
    main()
