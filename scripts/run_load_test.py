"""Load-test a running API with Locust and write results/load_test_<tag>.json.

The server must already be running (natively or in Docker). Queries are the 500 dev queries, picked
at random; this measures speed only, so no quality decision is made from it. Each scenario runs
warmup + duration seconds and only requests finishing after the warm-up are counted.

  uv run --group loadtest python scripts/run_load_test.py --host http://localhost:8000 --tag mac_native
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from finqa.data import fiqa
from finqa.utils.hardware import describe

# (name, concurrent users, rerank, rerank depth)
SCENARIOS = [
    ("retrieve_u1", 1, False, 0), ("retrieve_u4", 4, False, 0), ("retrieve_u16", 16, False, 0),
    ("retrieve_u64", 64, False, 0), ("rerank10_u1", 1, True, 10), ("rerank10_u4", 4, True, 10),
]


def run_scenario(host: str, users: int, rerank: bool, depth: int, warmup: float, duration: float, qfile: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "samples.json"
        env = {**os.environ, "FINQA_LT_QUERIES": qfile, "FINQA_LT_OUT": str(out),
               "FINQA_LT_RERANK": "1" if rerank else "0", "FINQA_LT_DEPTH": str(depth or 10)}
        t0 = time.time()
        subprocess.run([sys.executable, "-m", "locust", "-f", "loadtest/locustfile.py", "--headless", "--host", host,
                        "-u", str(users), "-r", str(users), "-t", f"{int(warmup + duration)}s", "--only-summary",
                        "--loglevel", "WARNING"], env=env, check=False, capture_output=True)
        samples = json.loads(out.read_text())
    kept = [(t, ms, ok) for t, ms, ok in samples if t >= t0 + warmup]
    if not kept:
        raise RuntimeError("no requests recorded after warm-up; is the server running?")
    ms = np.array([m for _, m, ok in kept if ok])
    window = max(t for t, _, _ in kept) - (t0 + warmup)
    failures = sum(1 for _, _, ok in kept if not ok)
    return {"users": users, "rerank": rerank, "rerank_depth": depth if rerank else None, "requests": len(kept),
            "failures": failures, "requests_per_second": len(ms) / window, "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)), "mean_ms": float(ms.mean())}


def ready(host: str) -> dict:
    with urllib.request.urlopen(f"{host}/ready", timeout=5) as r:
        return json.loads(r.read())


def throttle_delta(before: dict | None, after: dict | None) -> dict | None:
    """CPU throttling of the server's container during one scenario (None when it is not in a container)."""
    if not before or not after:
        return None
    periods = after["nr_periods"] - before["nr_periods"]
    throttled = after["nr_throttled"] - before["nr_throttled"]
    return {"periods": periods, "throttled_periods": throttled, "throttled_pct": 100 * throttled / periods if periods else 0.0,
            "throttled_ms": (after["throttled_usec"] - before["throttled_usec"]) / 1000,
            "cpu_seconds_used": (after["usage_usec"] - before["usage_usec"]) / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--tag", required=True, help="names the output file, e.g. mac_native or x86_docker")
    ap.add_argument("--warmup", type=float, default=5)
    ap.add_argument("--duration", type=float, default=20)
    args = ap.parse_args()

    server = ready(args.host)
    server.pop("cpu_throttle", None)
    d = fiqa.download()
    queries, qrels = fiqa.load_queries(d), fiqa.load_qrels(d, "dev")
    res = {"tag": args.tag, "server": server, "client_hardware": describe(),
           "note": "closed loop (no think time), k=10, hybrid; the Locust client runs on the same machine as the "
                   "server and competes for CPU; times are end to end over HTTP on localhost; one run per scenario",
           "warmup_s": args.warmup, "duration_s": args.duration, "scenarios": {}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump([queries[q] for q in sorted(qrels)], f)
    try:
        for name, users, rerank, depth in SCENARIOS:
            if rerank and not server.get("reranker"):
                continue
            before = ready(args.host).get("cpu_throttle")
            s = run_scenario(args.host, users, rerank, depth, args.warmup, args.duration, f.name)
            s["cpu_throttle"] = th = throttle_delta(before, ready(args.host).get("cpu_throttle"))
            res["scenarios"][name] = s
            print(f"{name:13s} users {users:3d}  {s['requests_per_second']:7.1f} req/s  p50 {s['p50_ms']:7.1f}  "
                  f"p95 {s['p95_ms']:7.1f}  p99 {s['p99_ms']:7.1f} ms  failures {s['failures']}"
                  + (f"  throttled {th['throttled_pct']:.0f}% of periods" if th else ""))
    finally:
        os.unlink(f.name)
    out = Path(f"results/load_test_{args.tag}.json")
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
