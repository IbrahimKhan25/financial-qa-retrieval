import math

import numpy as np
import pytest

from finqa.eval import metrics
from finqa.eval.stats import bootstrap_ci, paired_bootstrap


def test_ndcg_hand_computed():
    rel = {"a": 1, "b": 1}
    # relevant at ranks 1 and 3: DCG = 1 + 1/log2(4) = 1.5; IDCG = 1 + 1/log2(3)
    got = metrics.ndcg_at_k(["a", "x", "b"], rel)
    assert got == pytest.approx(1.5 / (1 + 1 / math.log2(3)))


def test_mrr_and_recall():
    rel = {"a": 1, "b": 1}
    assert metrics.mrr_at_k(["x", "y", "a"], rel) == pytest.approx(1 / 3)
    assert metrics.mrr_at_k(["x"] * 10 + ["a"], rel) == 0.0  # outside top 10
    assert metrics.recall_at_k(["a", "x"], rel) == 0.5


def test_missing_query_scores_zero():
    scores = metrics.per_query({}, {"q": {"a": 1}})
    assert scores["q"] == {"ndcg@10": 0.0, "recall@100": 0.0, "mrr@10": 0.0}


def test_matches_pytrec_eval_on_random_data():
    import pytrec_eval

    rng = np.random.default_rng(0)
    docs = [f"d{i}" for i in range(300)]
    qrels, ranking, run = {}, {}, {}
    for q in range(50):
        qid = f"q{q}"
        qrels[qid] = {d: 1 for d in rng.choice(docs, size=rng.integers(1, 6), replace=False)}
        ranked = list(rng.choice(docs, size=150, replace=False))
        ranking[qid] = ranked
        run[qid] = {d: float(len(ranked) - i) for i, d in enumerate(ranked)}  # strictly descending
    ref = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "recall_100", "recip_rank"})
    ref_scores = ref.evaluate(run)
    ours = metrics.per_query(ranking, qrels)
    for qid in qrels:
        assert ours[qid]["ndcg@10"] == pytest.approx(ref_scores[qid]["ndcg_cut_10"])
        assert ours[qid]["recall@100"] == pytest.approx(ref_scores[qid]["recall_100"])
        # trec's recip_rank has no cutoff, so compare only when the first hit is in the top 10
        rr_full = ref_scores[qid]["recip_rank"]
        expected = rr_full if rr_full >= 0.1 else 0.0
        assert ours[qid]["mrr@10"] == pytest.approx(expected)


def test_bootstrap_ci_contains_mean_and_is_seeded():
    x = np.random.default_rng(1).random(200)
    r1, r2 = bootstrap_ci(x, seed=7), bootstrap_ci(x, seed=7)
    assert r1 == r2
    assert r1["ci_low"] <= r1["mean"] <= r1["ci_high"]


def test_paired_bootstrap_detects_real_gain_and_null():
    rng = np.random.default_rng(2)
    a = rng.random(300)
    better = paired_bootstrap(a, a + 0.1 + rng.normal(0, 0.05, 300))
    assert better["ci_low"] > 0 and better["p_value"] < 0.01
    null = paired_bootstrap(a, a + rng.normal(0, 0.05, 300))
    assert null["ci_low"] < 0 < null["ci_high"]
