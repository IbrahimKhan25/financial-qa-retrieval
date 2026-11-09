from pathlib import Path

import pytest

from finqa.data import fiqa

DATA = Path("data/raw/fiqa")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run scripts/download_data.py first")


def test_split_sizes():
    s = fiqa.split_stats(DATA)
    assert s["n_docs"] == 57638
    assert (s["train"]["n_queries"], s["dev"]["n_queries"], s["test"]["n_queries"]) == (
        5500, 500, 648)


def test_query_splits_are_disjoint():
    ids = {s: set(fiqa.load_qrels(DATA, s)) for s in fiqa.SPLITS}
    assert not ids["train"] & ids["dev"]
    assert not ids["train"] & ids["test"]
    assert not ids["dev"] & ids["test"]


def test_qrels_reference_existing_docs_and_queries():
    corpus, queries = fiqa.load_corpus(DATA), fiqa.load_queries(DATA)
    for s in fiqa.SPLITS:
        for qid, rel in fiqa.load_qrels(DATA, s).items():
            assert qid in queries
            assert set(rel) <= set(corpus)
