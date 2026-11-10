"""ONNX Runtime wrappers with the same encode/score interface as the PyTorch models (CPU only)."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class _Session:
    def __init__(self, path: Path, threads: int):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}

    def run(self, tok: dict) -> np.ndarray:
        feed = {k: v.astype(np.int64) for k, v in tok.items() if k in self.names}
        return self.sess.run(None, feed)[0]


class OnnxBiEncoder:
    def __init__(self, onnx_path: Path, tokenizer_dir: Path, max_length: int = 256, threads: int = 4):
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.session = _Session(Path(onnx_path), threads)
        self.max_length = max_length

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        order = np.argsort([-len(t) for t in texts])
        out = None
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            tok = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="np")
            emb = self.session.run(dict(tok))
            if out is None:
                out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
            out[idx] = emb
        return out


class OnnxCrossEncoder:
    def __init__(self, onnx_path: Path, tokenizer_dir: Path, max_length: int = 256, threads: int = 4):
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.session = _Session(Path(onnx_path), threads)
        self.max_length = max_length

    def score(self, queries: list[str], docs: list[str], batch_size: int = 128) -> np.ndarray:
        order = np.argsort([-(len(q) + len(d)) for q, d in zip(queries, docs)])
        out = np.zeros(len(queries), dtype=np.float32)
        for s in range(0, len(queries), batch_size):
            idx = order[s:s + batch_size]
            tok = self.tokenizer([queries[i] for i in idx], [docs[i] for i in idx], padding=True,
                                 truncation="longest_first", max_length=self.max_length, return_tensors="np")
            out[idx] = self.session.run(dict(tok)).reshape(-1)
        return out
