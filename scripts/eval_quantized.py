"""Quality cost of ONNX / int8 on dev: numeric parity, retrieval quality, and rerank quality.

Reference is PyTorch fp32. Dev only; test is not touched.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from finqa.data import fiqa
from finqa.eval import metrics
from finqa.eval.dense_eval import encode_cached, search_scores
from finqa.eval.stats import bootstrap_ci, paired_bootstrap
from finqa.optimize.runtime import OnnxBiEncoder, OnnxCrossEncoder
from finqa.retrieval.rerank import rerank_top
from finqa.train.cross_encoder import CrossEncoder
from finqa.train.model import BiEncoder
from finqa.utils.hardware import describe
from finqa.utils.seed import set_seed


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="onnx_models")
    ap.add_argument("--bi-checkpoint", default="checkpoints/bi_encoder_D_mined")
    ap.add_argument("--ce", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--ce-name", default="offshelf")
    ap.add_argument("--candidates", default="data/candidates/dev_dense_D.json")
    ap.add_argument("--torch-ce-scores", default="data/cache/ce_scores_cross-encoder_ms-marco-MiniLM-L-6-v2.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/onnx_quality_dev.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    od = Path(args.onnx_dir)

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    qrels = fiqa.load_qrels(d, "dev")
    doc_ids = list(corpus)
    texts = [fiqa.doc_text(corpus[i]) for i in doc_ids]
    text = dict(zip(doc_ids, texts))
    qids = sorted(qrels)
    q_texts = [queries[q] for q in qids]
    cands = json.loads(Path(args.candidates).read_text())["candidates"]
    res = {"hardware": {**describe(num_threads=args.threads)}, "reference": "PyTorch fp32", "split": "dev"}

    bi_dir = od / "bi_encoder_D"
    torch_bi = BiEncoder(args.bi_checkpoint).to("cpu")
    bi = {k: OnnxBiEncoder(bi_dir / f"model_{k}.onnx", bi_dir, threads=args.threads) for k in ("fp32", "int8")}

    # ---- A. numeric parity of the bi-encoder
    sample = q_texts[:32] + texts[:32]
    ref = torch_bi.encode(sample)
    f32, i8 = bi["fp32"].encode(sample), bi["int8"].encode(sample)
    cos = (f32 * i8).sum(1)
    res["parity_bi_encoder"] = {"onnx_fp32_max_abs_diff_vs_torch": float(np.abs(ref - f32).max()),
                               "int8_cosine_to_fp32_mean": float(cos.mean()), "int8_cosine_to_fp32_min": float(cos.min()),
                               "n_texts": len(sample)}
    print("bi-encoder parity:", res["parity_bi_encoder"], flush=True)

    # ---- B. retrieval quality on dev
    key = f"finetuned-{hashlib.sha256((Path(args.bi_checkpoint) / 'model.safetensors').read_bytes()).hexdigest()[:10]}"
    d_fp32 = encode_cached(torch_bi, texts, key)  # cached from the hybrid run
    t0 = time.time()
    d_int8 = encode_cached(bi["int8"], texts, f"onnxint8-{key}")
    print(f"int8 document encoding took {time.time() - t0:.0f}s (0 if cached)", flush=True)
    q_emb = {"pytorch_fp32": torch_bi.encode(q_texts), "onnx_fp32": bi["fp32"].encode(q_texts),
             "onnx_int8": bi["int8"].encode(q_texts)}
    setups = {"pytorch fp32 (reference)": (q_emb["pytorch_fp32"], d_fp32),
              "onnx fp32 queries, fp32 docs": (q_emb["onnx_fp32"], d_fp32),
              "int8 queries, fp32 docs": (q_emb["onnx_int8"], d_fp32),
              "int8 queries, int8 docs": (q_emb["onnx_int8"], d_int8)}
    per = {}
    for name, (q, dd) in setups.items():
        idx, _ = search_scores(q, dd, k=100)
        per[name] = metrics.per_query({qq: [doc_ids[j] for j in row] for qq, row in zip(qids, idx)}, qrels)
    ref_name = "pytorch fp32 (reference)"
    res["retrieval"] = {}
    for name, p in per.items():
        entry = {"metrics": {m: bootstrap_ci([p[q][m] for q in qids]) for m in metrics.METRICS}}
        if name != ref_name:
            entry["paired_vs_reference"] = {m: paired_bootstrap([per[ref_name][q][m] for q in qids], [p[q][m] for q in qids])
                                            for m in metrics.METRICS}
        res["retrieval"][name] = entry
        m = entry["metrics"]
        print(f"retrieval {name:30s} ndcg {m['ndcg@10']['mean']:.4f} recall@100 {m['recall@100']['mean']:.4f} mrr {m['mrr@10']['mean']:.4f}", flush=True)

    # ---- C. reranker parity and rerank quality
    ce_dir = od / f"cross_encoder_{args.ce_name}"
    torch_ce = CrossEncoder(args.ce).to("cpu")
    ce = {k: OnnxCrossEncoder(ce_dir / f"model_{k}.onnx", ce_dir, threads=args.threads) for k in ("fp32", "int8")}
    pairs = [(queries[q], text[x]) for q in qids[:6] for x in cands[q][:25]]
    qs, ds = [p[0] for p in pairs], [p[1] for p in pairs]
    s_t, s_f, s_i = torch_ce.score(qs, ds), ce["fp32"].score(qs, ds), ce["int8"].score(qs, ds)
    res["parity_cross_encoder"] = {"onnx_fp32_max_abs_diff_vs_torch": float(np.abs(s_t - s_f).max()),
                                   "int8_pearson_vs_torch": corr(s_t, s_i), "int8_spearman_vs_torch": corr(ranks(s_t), ranks(s_i)),
                                   "n_pairs": len(pairs)}
    print("cross-encoder parity:", res["parity_cross_encoder"], flush=True)

    torch_scores = json.loads(Path(args.torch_ce_scores).read_text())  # fp32 scores from the Phase 4 run
    cache_path = Path(f"data/cache/ce_scores_onnxint8_{args.ce_name}.json")
    int8_scores = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    need = sorted({(q, x) for q in qids for x in cands[q] if f"{q}|{x}" not in int8_scores})
    if need:
        t0 = time.time()
        sc = ce["int8"].score([queries[q] for q, _ in need], [text[x] for _, x in need])
        int8_scores.update({f"{q}|{x}": float(s) for (q, x), s in zip(need, sc)})
        cache_path.write_text(json.dumps(int8_scores))
        print(f"int8 reranker scored {len(need)} pairs in {time.time() - t0:.0f}s", flush=True)
    res["rerank"] = {}
    for depth in (20, 100):
        pq = {}
        for name, sc in (("pytorch fp32 (reference)", torch_scores), ("onnx int8", int8_scores)):
            ranking = {q: rerank_top(cands[q], [sc[f"{q}|{x}"] for x in cands[q]], depth) for q in qids}
            pq[name] = metrics.per_query(ranking, qrels)
        res["rerank"][f"depth {depth}"] = {
            name: {"metrics": {m: bootstrap_ci([p[q][m] for q in qids]) for m in metrics.METRICS}} for name, p in pq.items()}
        res["rerank"][f"depth {depth}"]["onnx int8"]["paired_vs_reference"] = {
            m: paired_bootstrap([pq["pytorch fp32 (reference)"][q][m] for q in qids], [pq["onnx int8"][q][m] for q in qids])
            for m in metrics.METRICS}
        a, b = (res["rerank"][f"depth {depth}"][k]["metrics"]["ndcg@10"]["mean"] for k in ("pytorch fp32 (reference)", "onnx int8"))
        print(f"rerank depth {depth}: ndcg fp32 {a:.4f} int8 {b:.4f}", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
