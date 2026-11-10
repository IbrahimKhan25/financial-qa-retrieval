"""Dynamic int8 quantization: weights stored as int8, activation scales computed at run time.

No calibration data is needed. Quality and speed are measured, not assumed (see scripts/).
"""
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_int8(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})  # needed for fused graphs
