"""Dynamic int8 quantization: weights stored as int8, activation scales computed at run time.

No calibration data is needed. Quality and speed are measured, not assumed (see scripts/).
"""
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_int8(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
