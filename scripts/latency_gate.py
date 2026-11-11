"""Latency gate: fail if this code's p95 search latency is much worse than a baseline version's.

Both versions run on the same machine, in the same job, against the same model files and the same dev questions,
in alternating order (base, head, head, base, ...) so that slow drift of the machine hits both equally.
Each run is a fresh Python process whose PYTHONPATH points at that version's src/ folder.

  uv run python scripts/latency_gate.py --base-src ../base/src --max-ratio 1.3
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def worker(args) -> None:
    """Time single searches with whichever finqa is first on PYTHONPATH. Imports nothing newer than Phase 6."""
    import random
    from time import perf_counter

    import numpy as np

    import finqa
    from finqa.data import fiqa
    from finqa.serve.pipeline import Retriever
    from finqa.serve.store import load_artifacts

    r = Retriever(load_artifacts(args.artifacts, args.threads))
    d = fiqa.download()
    queries, qrels = fiqa.load_queries(d), fiqa.load_qrels(d, "dev")
    qids = sorted(qrels)
    random.Random(args.seed).shuffle(qids)
    texts = [queries[q] for q in qids[: args.n + args.warmup]]
    for t in texts[: args.warmup]:
        r.search(t, k=10)
    ms = []
    for t in texts[args.warmup:]:
        t0 = perf_counter()
        r.search(t, k=10)
        ms.append((perf_counter() - t0) * 1000)
    print(json.dumps({"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
                      "mean_ms": float(np.mean(ms)), "n": len(ms), "code": str(Path(finqa.__file__).parent)}))


def run_worker(src: str, args) -> dict:
    env = {**os.environ, "PYTHONPATH": str(Path(src).resolve())}
    cmd = [sys.executable, __file__, "--worker", "--artifacts", args.artifacts, "--threads", str(args.threads),
           "--n", str(args.n), "--warmup", str(args.warmup), "--seed", str(args.seed)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    res = json.loads(out.stdout.strip().splitlines()[-1])
    if Path(res["code"]).resolve() != (Path(src) / "finqa").resolve():
        raise RuntimeError(f"expected to time {src}/finqa but Python imported {res['code']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--base-src", help="src/ folder of the baseline version")
    ap.add_argument("--head-src", default="src")
    ap.add_argument("--artifacts", default="serving_artifacts")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=300, help="timed searches per round")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ratio", type=float, default=1.3)
    ap.add_argument("--out", default="results/latency_gate.json")
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    from finqa.eval.latency import compare
    from finqa.utils.hardware import describe

    runs = {"base": [], "head": []}
    src = {"base": args.base_src, "head": args.head_src}
    for i in range(args.rounds):
        for name in (("base", "head") if i % 2 == 0 else ("head", "base")):
            res = run_worker(src[name], args)
            runs[name].append(res)
            print(f"round {i + 1} {name}: p50 {res['p50_ms']:.2f} ms  p95 {res['p95_ms']:.2f} ms", flush=True)

    verdict = compare([r["p95_ms"] for r in runs["base"]], [r["p95_ms"] for r in runs["head"]], args.max_ratio)
    flag = "PASS" if verdict["passed"] else "FAIL"
    line = (f"p95 base {verdict['base_p95_ms']:.2f} ms, head {verdict['head_p95_ms']:.2f} ms, "
            f"ratio {verdict['ratio']:.2f} (limit {args.max_ratio:.2f}): {flag}")
    print(line)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({**verdict, "rounds": runs, "searches_per_round": args.n,
                                          "hardware": describe(args.threads)}, indent=1))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### Latency gate\n\n{line}\n\nMedian of {args.rounds} rounds of {args.n} searches each, "
                    f"{args.threads} threads, {describe(args.threads)['cpu_model']}.\n")
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()
