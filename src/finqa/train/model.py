"""Bi-encoder: a transformer backbone, mean pooling over tokens, L2 normalisation.

This reproduces what sentence-transformers does for all-MiniLM-L6-v2, written out so the
training loop is ours (a test checks the embeddings match).
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BiEncoder(nn.Module):
    def __init__(self, name_or_path: str, max_length: int = 256):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.backbone = AutoModel.from_pretrained(name_or_path)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask, **_):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean over real tokens only
        return F.normalize(pooled, dim=-1)

    def tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                              return_tensors="pt")

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """L2-normalised embeddings, float32, in the input order."""
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        use_amp = device.type == "cuda"
        order = np.argsort([-len(t) for t in texts])  # similar lengths together = less padding
        out = np.zeros((len(texts), self.backbone.config.hidden_size), dtype=np.float32)
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            tok = {k: v.to(device) for k, v in self.tokenize([texts[i] for i in idx]).items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                emb = self(**tok)
            out[idx] = emb.float().cpu().numpy()
        self.train(was_training)
        return out

    def save(self, path: str) -> None:
        self.backbone.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
