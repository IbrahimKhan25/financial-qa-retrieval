import random

from finqa.train.data import build_pairs, make_batches, sample_negative


def _pairs():
    qrels = {f"q{i}": {f"d{i}_{j}": 1 for j in range(1 + i % 3)} for i in range(40)}
    qrels["q0"]["zero"] = 0  # relevance 0 must not become a training pair
    return build_pairs(qrels)


def test_zero_relevance_is_not_a_pair():
    assert all(d != "zero" for _, d in _pairs())


def test_batches_never_repeat_a_query_and_cover_every_example_once():
    pairs = _pairs()
    batches = make_batches(pairs, batch_size=8, seed=1)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(pairs)))
    for b in batches:
        qids = [pairs[i][0] for i in b]
        assert len(qids) == len(set(qids)) and len(b) <= 8


def test_batches_are_seeded_and_change_between_epochs():
    pairs = _pairs()
    assert make_batches(pairs, 8, seed=5) == make_batches(pairs, 8, seed=5)
    assert make_batches(pairs, 8, seed=5) != make_batches(pairs, 8, seed=6)


def test_sample_negative_prefers_mined_and_falls_back_to_non_positive_random():
    rng = random.Random(0)
    assert sample_negative("q", {"q": ["n"]}, {"p"}, ["p", "n", "z"], rng) == "n"
    got = {sample_negative("q", {}, {"p"}, ["p", "n", "z"], rng) for _ in range(50)}
    assert "p" not in got
