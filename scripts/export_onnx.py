"""Export the bi-encoder and cross-encoder to ONNX fp32 and dynamic int8 (into a git-ignored folder).

Re-run with --ce <checkpoint dir> --ce-name finetuned once the fine-tuned reranker exists.
"""
import argparse
import json
from pathlib import Path

import onnx
import onnxruntime
import torch

from finqa.optimize.export import OPSET, export_bi_encoder, export_cross_encoder
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

    jobs = [("bi_encoder_D", BiEncoder(args.bi_checkpoint), export_bi_encoder),
            (f"cross_encoder_{args.ce_name}", CrossEncoder(args.ce), export_cross_encoder)]
    meta = {"opset": OPSET, "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
            "models": {}}
    for name, model, export in jobs:
        d = out / name
        model.tokenizer.save_pretrained(d)
        export(model, d / "model_fp32.onnx")
        onnx.checker.check_model(str(d / "model_fp32.onnx"))
        quantize_int8(d / "model_fp32.onnx", d / "model_int8.onnx")
        mb = {k: round((d / f"model_{k}.onnx").stat().st_size / 1e6, 1) for k in ("fp32", "int8")}
        meta["models"][name] = {"size_mb": mb, "int8_over_fp32": round(mb["int8"] / mb["fp32"], 3)}
        print(f"{name}: fp32 {mb['fp32']} MB, int8 {mb['int8']} MB")
    (out / "export_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
