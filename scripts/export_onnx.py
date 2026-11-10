"""Export the bi-encoder and cross-encoder to ONNX: default export, and eager-attention export + BERT fusion.

Per model directory: model_fp32.onnx and model_int8.onnx (default export), and model_fused_fp32.onnx and
model_fused_int8.onnx (attention/LayerNorm/GELU fused). Re-run with --ce <checkpoint dir> --ce-name finetuned
once the fine-tuned reranker exists. Output goes to a git-ignored folder.
"""
import argparse
import json
from pathlib import Path

import onnx
import onnxruntime
import torch

from finqa.optimize.export import OPSET, export_bi_encoder, export_cross_encoder
from finqa.optimize.fuse import fuse_bert
from finqa.optimize.quantize import quantize_int8
from finqa.train.cross_encoder import CrossEncoder
from finqa.train.model import BiEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi-checkpoint", default="checkpoints/bi_encoder_D_mined")
    ap.add_argument("--ce", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--ce-name", default="offshelf")
    ap.add_argument("--out", default="onnx_models")
    args = ap.parse_args()
    out = Path(args.out)

    jobs = [("bi_encoder_D", lambda **kw: BiEncoder(args.bi_checkpoint, **kw), export_bi_encoder, lambda m: m.backbone.config),
            (f"cross_encoder_{args.ce_name}", lambda **kw: CrossEncoder(args.ce, **kw), export_cross_encoder, lambda m: m.model.config)]
    meta = {"opset": OPSET, "torch": torch.__version__, "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__, "models": {}}
    for name, make, export, config_of in jobs:
        d = out / name
        model = make()
        model.tokenizer.save_pretrained(d)
        export(model, d / "model_fp32.onnx")                      # default (sdpa) export
        onnx.checker.check_model(str(d / "model_fp32.onnx"))
        quantize_int8(d / "model_fp32.onnx", d / "model_int8.onnx")
        eager = make(attn_implementation="eager")                  # explicit attention, which the fuser matches
        cfg = config_of(eager)
        export(eager, d / "model_eager_fp32.onnx")
        fused = fuse_bert(d / "model_eager_fp32.onnx", d / "model_fused_fp32.onnx", cfg.num_attention_heads,
                          cfg.hidden_size, cfg.num_hidden_layers)
        (d / "model_eager_fp32.onnx").unlink()
        quantize_int8(d / "model_fused_fp32.onnx", d / "model_fused_int8.onnx")
        mb = {k: round((d / f"model_{k}.onnx").stat().st_size / 1e6, 1) for k in ("fp32", "int8", "fused_fp32", "fused_int8")}
        meta["models"][name] = {"size_mb": mb, "fused_operators": fused}
        print(f"{name}: {mb} fused={fused}")
    (out / "export_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
