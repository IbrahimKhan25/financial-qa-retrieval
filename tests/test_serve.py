import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from finqa.optimize.export import export_bi_encoder, export_cross_encoder
from finqa.optimize.fuse import fuse_bert
from finqa.serve.app import create_app
from finqa.serve.build import save_bm25, write_text_store
from finqa.serve.onnx_models import QuestionEncoder
from finqa.train.cross_encoder import CrossEncoder
from finqa.train.model import BiEncoder

NAME = "sentence-transformers/all-MiniLM-L6-v2"  # cached stand-in; the cross-encoder gets a fresh random head
TOPICS = ["dividend", "mortgage", "bond", "index fund", "retirement account", "credit score", "inflation",
          "capital gains", "stock option", "savings account"]
TEMPLATES = ["How does a {t} work and what are the risks?", "A {t} can help you plan your finances over the long term.",
             "Many people ask whether a {t} is worth it.", "Tax rules for a {t} differ by country."]
# Different lengths give distinct BM25 scores (identical scores would be tie-broken differently by BM25 and by the fusion).
DOCS = [tpl.format(t=t) + " pad" * i for i, (t, tpl) in enumerate((t, tpl) for t in TOPICS for tpl in TEMPLATES)]
IDS = [f"d{i}" for i in range(len(DOCS))]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("art")
    for sub, model, export in (("bi_encoder", BiEncoder(NAME, attn_implementation="eager"), export_bi_encoder),
                               ("cross_encoder", CrossEncoder(NAME, attn_implementation="eager"), export_cross_encoder)):
        cfg = (model.backbone if sub == "bi_encoder" else model.model).config
        export(model, d / f"{sub}.eager.onnx")
        fuse_bert(d / f"{sub}.eager.onnx", d / sub / "model.onnx", cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers)
        model.tokenizer.save_pretrained(d / sub)
    emb = QuestionEncoder(d / "bi_encoder" / "model.onnx", d / "bi_encoder" / "tokenizer.json", 2).encode(DOCS)
    np.save(d / "doc_embeddings.npy", emb.astype(np.float32))
    (d / "doc_ids.json").write_text(json.dumps(IDS))
    write_text_store(DOCS, d)
    save_bm25(IDS, DOCS, d)
    (d / "manifest.json").write_text(json.dumps({"alpha": 0.8, "bi_encoder_sha256_10": "test", "built_at": "now"}))
    return d


@pytest.fixture(scope="module")
def client(artifacts):
    with TestClient(create_app(str(artifacts), threads=2)) as c:
        yield c


def search(client, **body):
    return client.post("/search", json=body)


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready").json()
    assert r["status"] == "ready" and r["docs"] == len(DOCS) and r["reranker"] is True


def test_search_returns_k_hits_in_non_increasing_score_order_with_timings(client):
    r = search(client, query="mortgage risks", k=5)
    assert r.status_code == 200
    body = r.json()
    assert len(body["hits"]) == 5 and body["mode"] == "hybrid"
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert all(h["snippet"] and h["rerank_score"] is None for h in body["hits"])
    assert {"encode_ms", "dense_ms", "bm25_ms", "fuse_ms", "rerank_ms", "total_ms"} <= set(body["timings_ms"])


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "   "}, {"query": "x", "k": 0}, {"query": "x", "k": 101},
                                  {"query": "x", "mode": "other"}, {"query": "x", "alpha": 1.5}, {"query": "x" * 2001}, {}])
def test_bad_requests_are_rejected_with_422(client, body):
    assert client.post("/search", json=body).status_code == 422


def test_alpha_extremes_reproduce_dense_and_bm25_orderings(client):
    ids = lambda **b: [h["doc_id"] for h in search(client, k=5, **b).json()["hits"]]
    q = "retirement account tax rules"
    assert ids(query=q, mode="hybrid", alpha=1.0) == ids(query=q, mode="dense")
    assert ids(query=q, mode="hybrid", alpha=0.0) == ids(query=q, mode="bm25")


def test_query_made_only_of_stopwords_still_works(client):
    r = search(client, query="the of and to", k=3)
    assert r.status_code == 200 and len(r.json()["hits"]) == 3


def test_rerank_reorders_only_the_head_and_scores_it(client):
    base = search(client, query="dividend risks", k=10).json()["hits"]
    rr = search(client, query="dividend risks", k=10, rerank=True, rerank_depth=5).json()["hits"]
    assert {h["doc_id"] for h in rr[:5]} == {h["doc_id"] for h in base[:5]}
    assert [h["doc_id"] for h in rr[5:]] == [h["doc_id"] for h in base[5:]]
    assert all(h["rerank_score"] is not None for h in rr[:5]) and all(h["rerank_score"] is None for h in rr[5:])


def test_rerank_without_a_reranker_is_a_400(artifacts, tmp_path):
    d = tmp_path / "no_ce"
    shutil.copytree(artifacts, d, ignore=shutil.ignore_patterns("cross_encoder"))
    with TestClient(create_app(str(d), threads=2)) as c:
        assert search(c, query="bond", rerank=True).status_code == 400
        assert search(c, query="bond").status_code == 200


def test_serving_code_never_imports_pytorch_or_transformers():
    code = "import sys, finqa.serve.app; sys.exit(1 if {'torch', 'transformers'} & set(sys.modules) else 0)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_tokenizers_library_matches_transformers_tokenization(artifacts):
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    ref = AutoTokenizer.from_pretrained(NAME)
    tok = Tokenizer.from_file(str(artifacts / "bi_encoder" / "tokenizer.json"))
    tok.enable_truncation(max_length=256)
    for text in ["Should I pay off my mortgage early?", "401k and IRA limits " * 100]:
        assert tok.encode(text).ids == ref(text, truncation=True, max_length=256)["input_ids"]
    q, doc = "what is a dividend", "A payment to shareholders. " * 200
    e = tok.encode(q, doc)
    r = ref(q, doc, truncation="longest_first", max_length=256)
    assert e.ids == r["input_ids"] and e.type_ids == r["token_type_ids"]


def test_throttle_delta_reports_share_of_paused_periods():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_load_test", "scripts/run_load_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    before = {"nr_periods": 100, "nr_throttled": 10, "throttled_usec": 1_000_000, "usage_usec": 5_000_000}
    after = {"nr_periods": 300, "nr_throttled": 110, "throttled_usec": 4_000_000, "usage_usec": 85_000_000}
    d = mod.throttle_delta(before, after)
    assert d["throttled_pct"] == 50.0 and d["throttled_ms"] == 3000.0 and d["cpu_seconds_used"] == 80.0
    assert mod.throttle_delta(None, after) is None


def test_spinning_can_be_switched_off(tmp_path, monkeypatch):
    from finqa.serve.onnx_models import make_session

    monkeypatch.setenv("FINQA_ORT_SPIN", "0")
    model = Path("serving_artifacts/bi_encoder/model.onnx")
    if not model.exists():
        pytest.skip("needs a built serving_artifacts/")
    assert make_session(model, 2).get_session_options().get_session_config_entry("session.intra_op.allow_spinning") == "0"
