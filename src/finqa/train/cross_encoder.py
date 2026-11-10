"""Cross-encoder: reads (query, document) together and outputs one relevance score.

Unlike the bi-encoder it cannot cache document vectors, so it only reranks a short candidate list.
"""
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoder(nn.Module):
    def __init__(self, name_or_path: str, max_length: int = 256, attn_implementation: str | None = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
        self.model = AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels=1, **kw)
        # Weights loaded straight from the safetensors file made the CPU forward pass return NaN for this
        # model (macOS arm64, torch 2.15); copying them into fresh memory fixes it. Disabling oneDNN did not.
        for t in [*self.model.parameters(), *self.model.buffers()]:
            t.data = t.data.clone()
        self.max_length = max_length

    def tokenize(self, queries: list[str], docs: list[str]) -> dict:
        # longest_first trims whichever side is longer, which in practice is the document.
        return self.tokenizer(queries, docs, padding=True, truncation="longest_first",
                              max_length=self.max_length, return_tensors="pt")

    def forward(self, **tok):
        return self.model(**tok).logits.squeeze(-1)

    @torch.no_grad()
    def score(self, queries: list[str], docs: list[str], batch_size: int = 128) -> np.ndarray:
        """One float score per (query, doc) pair, in input order."""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        use_amp = device.type == "cuda"
        order = np.argsort([-(len(q) + len(d)) for q, d in zip(queries, docs)])  # less padding
        out = np.zeros(len(queries), dtype=np.float32)
        for s in range(0, len(queries), batch_size):
            idx = order[s:s + batch_size]
            tok = {k: v.to(device) for k, v in
                   self.tokenize([queries[i] for i in idx], [docs[i] for i in idx]).items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out[idx] = self(**tok).float().cpu().numpy()
        self.train(was_training)
        if np.isnan(out).any():
            raise RuntimeError("cross-encoder produced NaN scores (see the weight-copy note in __init__)")
        return out

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
