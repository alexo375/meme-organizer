"""Модуль динамического квантования ONNX-моделей в формат INT8."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class OnnxQuantizer:
    """Применяет динамическое INT8-квантование к FP32 ONNX-моделям для CPU."""

    @staticmethod
    def quantize_dynamic_int8(
        fp32_onnx_path: Path,
        int8_onnx_path: Path,
        cleanup_fp32: bool = True,
    ) -> Path:
        """Квантует FP32 ONNX-модель в динамический INT8 и проверяет загрузку в CPUExecutionProvider.

        Args:
            fp32_onnx_path: Путь к исходной FP32 ONNX-модели.
            int8_onnx_path: Путь для сохранения квантованной INT8-модели.
            cleanup_fp32: Удалять ли исходный FP32-файл и связанный .data-файл после квантования.

        Returns:
            Путь к сохраненной INT8 ONNX-модели.

        Raises:
            FileNotFoundError: Если файл fp32_onnx_path не существует.

        """
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

        # Enforce ONNX IR version <= 9 for Android onnxruntime-android 1.17.1 compatibility
        import onnx
        model_proto = onnx.load(str(int8_onnx_path), load_external_data=True)
        if model_proto.ir_version > 9:
            model_proto.ir_version = 9
            onnx.save_model(model_proto, str(int8_onnx_path), save_as_external_data=False)

        if cleanup_fp32:
            fp32_onnx_path.unlink(missing_ok=True)
            data_file = fp32_onnx_path.with_name(f"{fp32_onnx_path.name}.data")
            data_file.unlink(missing_ok=True)

        # Verify quantized model loads cleanly on CPU
        _ = onnxruntime.InferenceSession(str(int8_onnx_path), providers=["CPUExecutionProvider"])
        size_mb = int8_onnx_path.stat().st_size / (1024 * 1024)
        logger.info("ONNX INT8 модель готова: %s (размер: %.2f МБ)", int8_onnx_path, size_mb)
        return int8_onnx_path
