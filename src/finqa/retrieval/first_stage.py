"""First-stage candidate lists (BM25, dense, hybrid) shared by the reranking scripts."""
from finqa.eval.dense_eval import encode_cached, search_scores
from finqa.retrieval.bm25 import BM25Retriever
from finqa.retrieval.hybrid import weighted
from finqa.train.model import BiEncoder


def bm25_lists(doc_ids, doc_texts, q_texts, k=100):
    idx, sc = BM25Retriever(doc_ids, doc_texts).search(q_texts, k=k)
    return [[(doc_ids[j], float(s)) for j, s in zip(row, srow)] for row, srow in zip(idx, sc)]


def dense_lists(model_path, key, doc_ids, doc_texts, q_texts, device, k=100):
    model = BiEncoder(model_path).to(device)
    d_emb = encode_cached(model, doc_texts, key)
    q_emb = model.encode(q_texts)
    idx, sc = search_scores(q_emb, d_emb, k=k, device=device)
    return [[(doc_ids[j], float(s)) for j, s in zip(row, srow)] for row, srow in zip(idx, sc)]


def hybrid_lists(dense, bm25, alpha, k=100):
    return [weighted(dl, bl, alpha, top=k) for dl, bl in zip(dense, bm25)]
