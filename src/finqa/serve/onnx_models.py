"""ONNX Runtime encoders for serving. No PyTorch or transformers: tokenization uses the `tokenizers` library."""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def make_session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if os.environ.get("FINQA_ORT_SPIN", "1") == "0":
        # By default idle ORT threads busy-wait for the next job. Under a container CPU quota that
        # wasted CPU counts against the quota and can get the whole container paused.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def load_tokenizer(path: Path, max_length: int) -> Tokenizer:
    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length=max_length)  # longest_first, as in training and evaluation
    tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")  # pad to the longest in the batch
    return tok


class _Model:
    def __init__(self, model_path: Path, tokenizer_json: Path, threads: int, max_length: int = 256):
        self.session = make_session(model_path, threads)
        self.names = {i.name for i in self.session.get_inputs()}
        self.tok = load_tokenizer(tokenizer_json, max_length)

    def _run(self, encodings) -> np.ndarray:
        feed = {"input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64)}
        return self.session.run(None, {k: v for k, v in feed.items() if k in self.names})[0]


class QuestionEncoder(_Model):
    """Text -> L2-normalised embedding (pooling and normalisation are inside the exported graph)."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._run(self.tok.encode_batch(texts))


class Reranker(_Model):
    """(query, document) pairs -> one relevance score each."""

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        return self._run(self.tok.encode_batch([(query, d) for d in docs])).reshape(-1)
