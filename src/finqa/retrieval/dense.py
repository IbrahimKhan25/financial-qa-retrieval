"""Dense bi-encoder retrieval: encode corpus once (cached), exact dot-product top-k."""
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DenseRetriever:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)

    def encode_cached(self, texts: list[str], cache_path: str | Path, **kw) -> np.ndarray:
        cache_path = Path(cache_path)
        if cache_path.exists():
            emb = np.load(cache_path)
            if len(emb) == len(texts):
                return emb
        emb = self.encode(texts, **kw)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        return emb

    @staticmethod
    def top_k(query_emb: np.ndarray, doc_emb: np.ndarray, k: int = 100):
        """Exact inner-product search (embeddings are L2-normalized, so this is cosine)."""
        q, d = torch.from_numpy(query_emb), torch.from_numpy(doc_emb)
        scores, idx = torch.topk(q @ d.T, k=k, dim=1)
        return idx.numpy(), scores.numpy()
