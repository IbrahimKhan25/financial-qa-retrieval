"""ONNX Runtime BERT graph fusion: attention, embedding+LayerNorm, GELU and residual LayerNorm.

The default exporter emits PyTorch's built-in attention operator, which the fuser does not recognise, so
export with eager attention (see scripts/export_onnx.py) and fuse afterwards. Raises if attention was
not fused, so a silently unoptimized graph can never be mistaken for an optimized one.
"""
from pathlib import Path

from onnxruntime.transformers.optimizer import optimize_model


def fuse_bert(src: Path, dst: Path, num_heads: int, hidden_size: int, num_layers: int) -> dict:
    m = optimize_model(str(src), model_type="bert", num_heads=num_heads, hidden_size=hidden_size)
    stats = {k: v for k, v in m.get_fused_operator_statistics().items() if v}
    if stats.get("Attention", 0) != num_layers:
        raise RuntimeError(f"attention was not fused in every layer: {stats}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    m.save_model_to_file(str(dst))
    return stats
