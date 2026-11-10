import random

import numpy as np
import pytest
import torch

from finqa.retrieval.rerank import rerank_top
from finqa.train.cross_encoder import CrossEncoder
from finqa.train.data import sample_group
from finqa.train.losses import group_cross_entropy

# Plumbing tests use the locally cached MiniLM weights (a BERT) with a fresh 1-output head.
NAME = "sentence-transformers/all-MiniLM-L6-v2"


def test_group_loss_small_when_positive_wins_and_large_when_it_loses():
    good = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    bad = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
    assert group_cross_entropy(good).item() < 1e-3
    assert group_cross_entropy(bad).item() > 5


def test_group_loss_uninformative_is_log_g():
    assert abs(group_cross_entropy(torch.zeros(3, 8)).item() - np.log(8)) < 1e-5


def test_sample_group_puts_positive_first_and_never_uses_positives_as_negatives():
    rng = random.Random(0)
    positives = {"p", "p2"}
    negatives = {"q": ["n1", "p2", "n2", "n3"]}  # p2 is a positive and must be dropped
    for _ in range(30):
        g = sample_group("q", "p", negatives, positives, [f"d{i}" for i in range(50)] + ["p", "p2"], rng, 5)
        assert g[0] == "p" and len(g) == 6 and len(set(g)) == 6
        assert not (set(g[1:]) & positives)


def test_sample_group_falls_back_to_random_docs_when_no_mined_negatives():
    g = sample_group("q", "p", {}, {"p"}, ["p", "a", "b", "c", "d"], random.Random(1), 3)
    assert g[0] == "p" and set(g[1:]) <= {"a", "b", "c", "d"} and len(set(g)) == 4


def test_rerank_top_reorders_only_the_head_and_keeps_the_tail():
    cands = ["a", "b", "c", "d", "e"]
    scores = [0.1, 0.9, 0.5, 5.0, 5.0]  # d and e score highest but sit beyond depth 3
    assert rerank_top(cands, scores, depth=3) == ["b", "c", "a", "d", "e"]
    assert rerank_top(cands, scores, depth=5) == ["d", "e", "b", "c", "a"]  # stable on ties
    assert rerank_top(cands, scores, depth=99) == rerank_top(cands, scores, depth=5)


@pytest.fixture(scope="module")
def ce():
    return CrossEncoder(NAME, max_length=64)


def test_pair_tokenization_trims_the_document_not_the_question(ce):
    q = "what is a dividend"
    tok = ce.tokenize([q], ["word " * 500])
    ids = tok["input_ids"][0].tolist()
    assert len(ids) == 64
    q_ids = ce.tokenizer(q, add_special_tokens=False)["input_ids"]
    assert ids[1:1 + len(q_ids)] == q_ids  # the full question survives at the front


def test_score_shape_order_and_determinism(ce):
    qs = ["what is a dividend", "pay off mortgage early", "what is a dividend"]
    ds = ["a payment to shareholders", "interest rates matter", "a payment to shareholders"]
    s1, s2 = ce.score(qs, ds, batch_size=2), ce.score(qs, ds, batch_size=3)
    assert s1.shape == (3,) and np.allclose(s1, s2, atol=1e-4)
    assert abs(s1[0] - s1[2]) < 1e-4  # identical pairs score identically, whatever the batching


def test_score_raises_instead_of_returning_nan(ce, monkeypatch):
    monkeypatch.setattr(type(ce.model), "forward", lambda self, **kw: type("O", (), {"logits": torch.full((len(kw["input_ids"]), 1), float("nan"))})())
    with pytest.raises(RuntimeError, match="NaN"):
        ce.score(["q"], ["d"])


def test_real_reranker_gives_finite_correct_cpu_scores():
    # Regression for a real bug: weights loaded straight from the safetensors file made this model return
    # NaN on CPU (macOS arm64). Expected values were cross-checked against the GPU and ONNX scores.
    from huggingface_hub import try_to_load_from_cache

    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    if not isinstance(try_to_load_from_cache(name, "model.safetensors"), str):
        pytest.skip("reranker weights not cached")
    s = CrossEncoder(name).to("cpu").score(
        ["what is a dividend", "pay off mortgage early"],
        ["A payment made to shareholders out of company profits.",
         "Paying down debt guarantees a return equal to the loan interest rate."])
    assert np.allclose(s, [-2.023, -10.181], atol=0.01)
