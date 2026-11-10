"""Evaluate a BiEncoder on a split against the full corpus (no candidate shortcuts)."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from finqa.eval import metrics


def search_scores(q_emb: np.ndarray, d_emb: np.ndarray, k: int = 100, chunk: int = 512,
                  device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k (doc indices, scores) per query by inner product (embeddings are L2-normalised)."""
    d = torch.from_numpy(d_emb).to(device)
    idx, sc = [], []
    for s in range(0, len(q_emb), chunk):
        q = torch.from_numpy(q_emb[s:s + chunk]).to(device)
        top = torch.topk(q @ d.T, k=k, dim=1)
        idx.append(top.indices.cpu().numpy())
        sc.append(top.values.float().cpu().numpy())
    return np.concatenate(idx), np.concatenate(sc)


def search(q_emb: np.ndarray, d_emb: np.ndarray, k: int = 100, chunk: int = 512,
           device: str = "cpu") -> np.ndarray:
    """Exact top-k doc indices per query by inner product."""
    return search_scores(q_emb, d_emb, k, chunk, device)[0]


def encode_cached(model, texts: list[str], key: str, cache_dir: str = "data/cache",
                  batch_size: int = 256) -> np.ndarray:
    """Encode once and cache. The file name carries a hash of the texts, so changed text can
    never silently reuse stale embeddings; `key` must identify the model weights."""
    h = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"{key}_{h}.npy"
    if path.exists():
        return np.load(path)
    emb = model.encode(texts, batch_size=batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


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
