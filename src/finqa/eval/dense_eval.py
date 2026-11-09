"""Evaluate a BiEncoder on a split against the full corpus (no candidate shortcuts)."""
import numpy as np
import torch

from finqa.eval import metrics


def search(q_emb: np.ndarray, d_emb: np.ndarray, k: int = 100, chunk: int = 512,
           device: str = "cpu") -> np.ndarray:
    """Exact top-k doc indices per query by inner product (embeddings are L2-normalised)."""
    d = torch.from_numpy(d_emb).to(device)
    out = []
    for s in range(0, len(q_emb), chunk):
        q = torch.from_numpy(q_emb[s:s + chunk]).to(device)
        out.append(torch.topk(q @ d.T, k=k, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def evaluate(model, doc_ids: list[str], doc_texts: list[str], qids: list[str],
             q_texts: list[str], qrels: dict, k: int = 100, doc_batch: int = 256):
    """Returns (per-query scores, aggregate means)."""
    device = next(model.parameters()).device.type
    d_emb = model.encode(doc_texts, batch_size=doc_batch)
    q_emb = model.encode(q_texts)
    idx = search(q_emb, d_emb, k=k, device=device)
    ranking = {q: [doc_ids[j] for j in row] for q, row in zip(qids, idx)}
    scores = metrics.per_query(ranking, qrels)
    return scores, metrics.aggregate(scores)
