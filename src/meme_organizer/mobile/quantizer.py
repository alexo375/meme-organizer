"""Dedicated ONNX INT8 Quantizer for Mobile Delivery Bundles."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class OnnxQuantizer:
    """Applies dynamic INT8 quantization to FP32 ONNX models for mobile CPU inference."""

    @staticmethod
    def quantize_dynamic_int8(
        fp32_onnx_path: Path,
        int8_onnx_path: Path,
        cleanup_fp32: bool = True,
    ) -> Path:
        """Quantize an FP32 ONNX model into dynamic INT8 and verify loadability."""
        import onnxruntime
        from onnxruntime.quantization import QuantType, quantize_dynamic

        if not fp32_onnx_path.exists():
            raise FileNotFoundError(f"Исходная FP32 ONNX модель не найдена: {fp32_onnx_path}")

        int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Применение dynamic INT8 квантования (%s -> %s)...", fp32_onnx_path.name, int8_onnx_path.name)

        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=int8_onnx_path,
            weight_type=QuantType.QInt8,
        )

        if cleanup_fp32:
            fp32_onnx_path.unlink(missing_ok=True)
            data_file = fp32_onnx_path.with_name(f"{fp32_onnx_path.name}.data")
            data_file.unlink(missing_ok=True)

        # Verify quantized model loads cleanly on CPU
        _ = onnxruntime.InferenceSession(str(int8_onnx_path), providers=["CPUExecutionProvider"])
        size_mb = int8_onnx_path.stat().st_size / (1024 * 1024)
        logger.info("ONNX INT8 модель готова: %s (размер: %.2f МБ)", int8_onnx_path, size_mb)
        return int8_onnx_path
