from finqa.train.negatives import forbidden_doc_ids, mine_hard_negatives


def test_forbidden_is_union_of_positives_only():
    dev = {"d1": {"a": 1, "x": 0}}
    test = {"t1": {"b": 1}}
    assert forbidden_doc_ids(dev, test) == {"a", "b"}


def test_pool_excludes_own_positives_and_held_out_positives():
    ranked = {"q": ["p", "n1", "heldout", "n2", "n3", "n4"]}
    train = {"q": {"p": 1}}
    out = mine_hard_negatives(ranked, train, forbidden={"heldout"}, window=(0, 6))
    assert out["q"] == ["n1", "n2", "n3", "n4"]


def test_window_skips_top_ranks_and_respects_cap():
    ranked = {"q": [f"d{i}" for i in range(20)]}
    out = mine_hard_negatives(ranked, {"q": {}}, forbidden=set(), window=(5, 12), max_per_query=3)
    assert out["q"] == ["d5", "d6", "d7"]


def test_no_forbidden_doc_ever_reaches_any_pool():
    ranked = {f"q{i}": [f"d{j}" for j in range(30)] for i in range(5)}
    forbidden = {f"d{j}" for j in range(0, 30, 3)}
    out = mine_hard_negatives(ranked, {q: {} for q in ranked}, forbidden, window=(0, 30), max_per_query=50)
    assert not {d for docs in out.values() for d in docs} & forbidden
