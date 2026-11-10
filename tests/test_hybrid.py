import random

import pytest

from finqa.retrieval.hybrid import FLOOR, minmax, rrf, weighted


def test_rrf_matches_hand_computation():
    # k=60: y = 1/62 + 1/61, z = 1/63 + 1/62, x = 1/61, w = 1/63
    assert rrf([["x", "y", "z"], ["y", "z", "w"]], k=60) == ["y", "z", "x", "w"]


def test_rrf_respects_top():
    assert len(rrf([[f"a{i}" for i in range(50)], [f"b{i}" for i in range(50)]], top=30)) == 30


def test_minmax_range_and_floor():
    out = minmax([("a", 10.0), ("b", 5.0), ("c", 0.0)])
    assert out["a"] == pytest.approx(1.0)
    assert out["c"] == pytest.approx(FLOOR) and out["c"] > 0
    assert out["c"] < out["b"] < out["a"]


def test_minmax_constant_scores_do_not_divide_by_zero():
    assert set(minmax([("a", 3.0), ("b", 3.0)]).values()) == {1.0}


def _lists(seed):
    rng = random.Random(seed)
    docs = [f"d{i}" for i in range(150)]
    dense = [(d, 1.0 - 0.005 * i) for i, d in enumerate(rng.sample(docs, 100))]
    lex = [(d, 40.0 - 0.3 * i) for i, d in enumerate(rng.sample(docs, 100))]
    return dense, lex


@pytest.mark.parametrize("seed", range(5))
def test_alpha_extremes_reproduce_each_system_exactly(seed):
    dense, lex = _lists(seed)
    assert weighted(dense, lex, alpha=1.0) == [d for d, _ in dense]
    assert weighted(dense, lex, alpha=0.0) == [d for d, _ in lex]


def test_weighted_blend_can_promote_doc_both_systems_like():
    dense = [("a", 0.9), ("both", 0.8), ("c", 0.1)]
    lex = [("b", 9.0), ("both", 8.0), ("d", 1.0)]
    out = weighted(dense, lex, alpha=0.5)
    assert out[0] == "both"
