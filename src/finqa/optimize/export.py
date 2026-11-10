"""Export the bi-encoder and cross-encoder to ONNX (fp32) with dynamic batch and sequence length."""
from pathlib import Path

import torch
from torch import nn

from finqa.train.cross_encoder import CrossEncoder
from finqa.train.model import BiEncoder

OPSET = 17


class _CrossEncoderGraph(nn.Module):
    """Explicit-argument wrapper so the exported graph has stable input names."""

    def __init__(self, ce: CrossEncoder):
        super().__init__()
        self.ce = ce

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.ce(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


def _export(module: nn.Module, example: dict, input_names: list[str], output_name: str, path: Path) -> None:
    module.eval()
    dynamic = {n: {0: "batch", 1: "seq"} for n in input_names}
    dynamic[output_name] = {0: "batch"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module, tuple(example[n] for n in input_names), str(path),
                          input_names=input_names, output_names=[output_name], dynamic_axes=dynamic,
                          opset_version=OPSET, do_constant_folding=True, dynamo=False)


def export_bi_encoder(model: BiEncoder, path: Path) -> None:
    """Graph: input_ids, attention_mask -> L2-normalised embedding (pooling is inside the graph)."""
    model = model.to("cpu")
    tok = model.tokenize(["Should I pay off my mortgage early?", "What is a dividend and how is it taxed?"])
    _export(model, tok, ["input_ids", "attention_mask"], "embedding", path)


def export_cross_encoder(model: CrossEncoder, path: Path) -> None:
    """Graph: input_ids, attention_mask, token_type_ids -> one relevance logit per pair."""
    model = model.to("cpu")
    tok = model.tokenize(["pay off mortgage early", "what is a dividend"],
                         ["Paying down debt guarantees a return equal to the interest rate.", "A payment to shareholders."])
    _export(_CrossEncoderGraph(model), tok, ["input_ids", "attention_mask", "token_type_ids"], "logit", path)
