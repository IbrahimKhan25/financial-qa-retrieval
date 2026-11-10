import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from finqa.optimize.export import export_bi_encoder, export_cross_encoder
from finqa.optimize.quantize import quantize_int8
from finqa.optimize.runtime import OnnxBiEncoder, OnnxCrossEncoder
from finqa.train.cross_encoder import CrossEncoder
from finqa.train.model import BiEncoder

# Plumbing tests use the locally cached MiniLM weights (a BERT); the cross-encoder gets a fresh random head.
NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = ["Should I pay off my mortgage early?", "What is a dividend?", "401k and IRA contribution limits " * 20,
         "bonds"]


@pytest.fixture(scope="module")
def bi(tmp_path_factory):
    d = tmp_path_factory.mktemp("bi")
    model = BiEncoder(NAME)
    model.tokenizer.save_pretrained(d)
    export_bi_encoder(model, d / "fp32.onnx")
    quantize_int8(d / "fp32.onnx", d / "int8.onnx")
    return model.to("cpu"), d


@pytest.fixture(scope="module")
def ce(tmp_path_factory):
    d = tmp_path_factory.mktemp("ce")
    model = CrossEncoder(NAME)
    model.tokenizer.save_pretrained(d)
    export_cross_encoder(model, d / "fp32.onnx")
    return model.to("cpu"), d


def test_bi_encoder_fp32_onnx_matches_pytorch(bi):
    model, d = bi
    ref = model.encode(TEXTS)
    got = OnnxBiEncoder(d / "fp32.onnx", d).encode(TEXTS)
    assert np.abs(ref - got).max() < 1e-4


def test_bi_encoder_onnx_handles_any_batch_size_and_keeps_order(bi):
    _, d = bi
    enc = OnnxBiEncoder(d / "fp32.onnx", d)
    full = enc.encode(TEXTS, batch_size=4)
    single = np.stack([enc.encode([t])[0] for t in TEXTS])
    assert np.abs(full - single).max() < 1e-4


def test_bi_encoder_int8_is_sane_and_smaller(bi):
    # Sanity only: int8 embeddings are not garbage and the file shrinks. Measured cosines to fp32 on
    # these texts were about 0.92 to 0.98, so how much that costs in retrieval quality is decided on
    # dev by scripts/eval_quantized.py, not by this threshold.
    _, d = bi
    f32 = OnnxBiEncoder(d / "fp32.onnx", d).encode(TEXTS)
    i8 = OnnxBiEncoder(d / "int8.onnx", d).encode(TEXTS)
    cos = (f32 * i8).sum(1)  # both are unit length
    assert cos.min() > 0.85
    assert (d / "int8.onnx").stat().st_size < 0.5 * (d / "fp32.onnx").stat().st_size


def test_cross_encoder_fp32_onnx_matches_pytorch(ce):
    model, d = ce
    qs, ds = ["what is a dividend", "pay off mortgage"], ["a payment to shareholders", "interest rates matter " * 40]
    ref = model.score(qs, ds)
    got = OnnxCrossEncoder(d / "fp32.onnx", d).score(qs, ds)
    assert np.abs(ref - got).max() < 1e-3


@pytest.fixture(scope="module")
def fused_bi(tmp_path_factory):
    from finqa.optimize.fuse import fuse_bert

    d = tmp_path_factory.mktemp("fused_bi")
    model = BiEncoder(NAME, attn_implementation="eager")
    model.tokenizer.save_pretrained(d)
    export_bi_encoder(model, d / "eager.onnx")
    cfg = model.backbone.config
    stats = fuse_bert(d / "eager.onnx", d / "fused_fp32.onnx", cfg.num_attention_heads, cfg.hidden_size,
                      cfg.num_hidden_layers)
    quantize_int8(d / "fused_fp32.onnx", d / "fused_int8.onnx")
    return BiEncoder(NAME).to("cpu"), d, stats, cfg.num_hidden_layers


def test_fusion_fuses_attention_in_every_layer(fused_bi):
    _, _, stats, layers = fused_bi
    assert stats["Attention"] == layers


def test_fused_fp32_matches_pytorch_and_fused_int8_is_sane(fused_bi):
    model, d, _, _ = fused_bi
    ref = model.encode(TEXTS)
    f32 = OnnxBiEncoder(d / "fused_fp32.onnx", d).encode(TEXTS)
    i8 = OnnxBiEncoder(d / "fused_int8.onnx", d).encode(TEXTS)
    assert np.abs(ref - f32).max() < 1e-4
    assert (f32 * i8).sum(1).min() > 0.85


def test_fusion_raises_when_attention_is_not_fused(bi, tmp_path):
    # The default (sdpa) export has an attention pattern the fuser does not match; that must be an error.
    from finqa.optimize.fuse import fuse_bert

    _, d = bi
    with pytest.raises(RuntimeError, match="attention was not fused"):
        fuse_bert(d / "fp32.onnx", tmp_path / "x.onnx", 12, 384, 6)
