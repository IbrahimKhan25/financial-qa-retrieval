"""Diagnose a cross-encoder on TRAIN groups: does it rank the labeled positive first among 1 + 7 negatives?

Two kinds of group: negatives drawn from the mined hard-negative files (the training distribution) and random
corpus documents (easy). A model that learned something useful but forgot basic relevance does better on the
first and collapses on the second. Train queries only; no dev or test data is used.
"""
import argparse
import json
import random
from pathlib import Path

from finqa.data import fiqa
from finqa.train.cross_encoder import CrossEncoder
from finqa.train.data import build_pairs, load_negatives, positives_of
from finqa.train.model import pick_device
from finqa.utils.hardware import describe

G = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--negatives", nargs="+", default=["data/negatives/ce_bm25.json", "data/negatives/ce_dense_D.json"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/reranker_diagnostic.json")
    args = ap.parse_args()

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    train = fiqa.load_qrels(d, "train")
    text = {i: fiqa.doc_text(corpus[i]) for i in corpus}
    pos_of, negs = positives_of(train), load_negatives(args.negatives)
    all_docs = list(corpus)
    rng = random.Random(args.seed)
    sample = rng.sample(build_pairs(train), args.n)
    device = pick_device()
    models = {n: CrossEncoder(p).to(device) for n, p in (a.split("=", 1) for a in args.models)}

    res = {"n_groups": args.n, "group_size": G, "seed": args.seed, "chance_top1": 1 / G, "split": "train", "results": {}}
    for kind in ("hard", "random"):
        qs, ds = [], []
        for q, p in sample:
            if kind == "hard":
                neg = rng.sample([x for x in negs[q] if x not in pos_of[q]], G - 1)
            else:
                neg = []
                while len(neg) < G - 1:
                    x = rng.choice(all_docs)
                    if x not in pos_of[q] and x not in neg:
                        neg.append(x)
            for x in [p, *neg]:
                qs.append(queries[q])
                ds.append(text[x])
        for name, m in models.items():
            s = m.score(qs, ds).reshape(-1, G)
            rank = (s > s[:, :1]).sum(1) + 1
            res["results"].setdefault(kind, {})[name] = {"positive_first": float((s.argmax(1) == 0).mean()),
                                                         "mean_reciprocal_rank": float((1 / rank).mean())}
            r = res["results"][kind][name]
            print(f"{kind:6s} negatives | {name:14s} | positive first {r['positive_first']:.3f} | MRR {r['mean_reciprocal_rank']:.3f}")
    res["hardware"] = {**describe(), "device": device}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
