"""Locust user for POST /search. Use scripts/run_load_test.py, which sets the environment and reads the output.

Locust's own percentiles are rounded to whole milliseconds, too coarse for a 6 ms endpoint, so every
response time is also written unrounded to FINQA_LT_OUT and the percentiles are computed from that file.
"""
import json
import os
import random
import time
from pathlib import Path

from locust import FastHttpUser, constant, events, task

QUERIES = json.loads(Path(os.environ["FINQA_LT_QUERIES"]).read_text(encoding="utf-8"))
RERANK = os.environ.get("FINQA_LT_RERANK", "0") == "1"
DEPTH = int(os.environ.get("FINQA_LT_DEPTH", "10"))
SAMPLES: list[tuple[float, float, bool]] = []  # (finish time, response ms, ok)


@events.request.add_listener
def record(response_time, exception, **_):
    SAMPLES.append((time.time(), float(response_time), exception is None))


@events.quitting.add_listener
def dump(**_):
    with open(os.environ["FINQA_LT_OUT"], "w") as f:
        json.dump(SAMPLES, f)


class SearchUser(FastHttpUser):
    wait_time = constant(0)  # closed loop: each user sends its next request as soon as the last one returns

    def on_start(self):
        self.rng = random.Random(id(self))

    @task
    def search(self):
        body = {"query": self.rng.choice(QUERIES), "k": 10, "rerank": RERANK, "rerank_depth": DEPTH}
        self.client.post("/search", json=body, name="/search")
