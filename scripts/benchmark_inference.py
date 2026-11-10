"""Latency (p50/p95) and throughput: PyTorch fp32 vs ONNX fp32 vs ONNX int8, on CPU, dev inputs.

Hardware and thread count are recorded in the output. Timings include tokenization. Outside Docker;
the Docker and x86 numbers come later.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from finqa.data import fiqa
from finqa.optimize.runtime import OnnxBiEncoder, OnnxCrossEncoder
from finqa.train.cross_encoder import CrossEncoder
from finqa.train.model import BiEncoder
from finqa.utils.hardware import describe
from finqa.utils.seed import set_seed

DEPTHS = (10, 20, 50, 100)


def stats(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "n": len(a)}


def timed(fn, n_warmup: int = 3) -> list[float]:
    out = []
    for i, call in enumerate(fn):
        t = time.perf_counter()
        call()
        if i >= n_warmup:
            out.append((time.perf_counter() - t) * 1000)
    return out


def torch_mb(model) -> float:
    return round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="onnx_models")
    ap.add_argument("--bi-checkpoint", default="checkpoints/bi_encoder_D_mined")
    ap.add_argument("--ce", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--ce-name", default="offshelf")
    ap.add_argument("--candidates", default="data/candidates/dev_dense_D.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--out", default="results/onnx_benchmark.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    od = Path(args.onnx_dir)

    d = fiqa.download()
    corpus, queries = fiqa.load_corpus(d), fiqa.load_queries(d)
    qids = sorted(fiqa.load_qrels(d, "dev"))
    text = {i: fiqa.doc_text(corpus[i]) for i in corpus}
    cands = json.loads(Path(args.candidates).read_text())["candidates"]
    n = args.queries + 3
    q_texts = [queries[q] for q in qids[:n]]
    res = {"hardware": {**describe(num_threads=args.threads), "torch": torch.__version__},
           "note": "CPU, timings include tokenization, one query per call, outside Docker",
           "bi_encoder": {}, "cross_encoder": {}}

    # ---- bi-encoder: single-query latency, and document throughput
    bi_dir = od / "bi_encoder_D"
    torch_bi = BiEncoder(args.bi_checkpoint).to("cpu")
    backends = {"pytorch_fp32": (torch_bi, torch_mb(torch_bi)),
                "onnx_fp32": (OnnxBiEncoder(bi_dir / "model_fp32.onnx", bi_dir, threads=args.threads),
                              round((bi_dir / "model_fp32.onnx").stat().st_size / 1e6, 1)),
                "onnx_int8": (OnnxBiEncoder(bi_dir / "model_int8.onnx", bi_dir, threads=args.threads),
                              round((bi_dir / "model_int8.onnx").stat().st_size / 1e6, 1))}
    docs = [text[i] for i in list(corpus)[:512]]
    for name, (enc, mb) in backends.items():
        lat = stats(timed([(lambda t=t, enc=enc: enc.encode([t], batch_size=1)) for t in q_texts]))
        t0 = time.perf_counter()
        enc.encode(docs, batch_size=32)
        thr = len(docs) / (time.perf_counter() - t0)
        res["bi_encoder"][name] = {"single_query_latency": lat, "docs_per_second_batch32": thr, "model_mb": mb}
        print(f"bi-encoder {name:13s} p50 {lat['p50_ms']:.1f} ms p95 {lat['p95_ms']:.1f} ms | {thr:.0f} docs/s | {mb} MB", flush=True)

    # ---- cross-encoder: per-query latency by rerank depth, and pair throughput
    ce_dir = od / f"cross_encoder_{args.ce_name}"
    torch_ce = CrossEncoder(args.ce).to("cpu")
    ce_backends = {"pytorch_fp32": (torch_ce, torch_mb(torch_ce)),
                   "onnx_fp32": (OnnxCrossEncoder(ce_dir / "model_fp32.onnx", ce_dir, threads=args.threads),
                                 round((ce_dir / "model_fp32.onnx").stat().st_size / 1e6, 1)),
                   "onnx_int8": (OnnxCrossEncoder(ce_dir / "model_int8.onnx", ce_dir, threads=args.threads),
                                 round((ce_dir / "model_int8.onnx").stat().st_size / 1e6, 1))}
    pairs = [(queries[q], text[x]) for q in qids[:12] for x in cands[q]][:1024]
    for name, (ce, mb) in ce_backends.items():
        by_depth = {}
        for depth in DEPTHS:
            calls = [(lambda q=q, depth=depth, ce=ce: ce.score([queries[q]] * depth, [text[x] for x in cands[q][:depth]],
                                                       batch_size=depth)) for q in qids[:n]]
            by_depth[depth] = stats(timed(calls))
        t0 = time.perf_counter()
        ce.score([p[0] for p in pairs], [p[1] for p in pairs], batch_size=32)
        thr = len(pairs) / (time.perf_counter() - t0)
        res["cross_encoder"][name] = {"latency_by_depth": by_depth, "pairs_per_second_batch32": thr, "model_mb": mb}
        print(f"cross-encoder {name:13s} " + " ".join(f"d{k}:{v['p50_ms']:.0f}/{v['p95_ms']:.0f}ms" for k, v in by_depth.items())
              + f" | {thr:.0f} pairs/s | {mb} MB", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
