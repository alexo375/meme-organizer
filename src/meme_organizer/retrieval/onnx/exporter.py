"""Модуль экспорта моделей в формат ONNX и квантования INT8."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class OnnxExporter:
    """Экспортирует HuggingFace Transformer в ONNX и применяет dynamic INT8 квантование."""

    @staticmethod
    def export_text_encoder(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: Path,
        base_name: str,
        max_length: int = 64,
    ) -> Path:
        """Экспортирует модель в ONNX и возвращает путь к квантованному файлу."""
        import onnxruntime
        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic

        output_dir.mkdir(parents=True, exist_ok=True)
        raw_onnx_path = output_dir / f"{base_name}_fp32.onnx"
        quant_onnx_path = output_dir / f"{base_name}_int8.onnx"

        if quant_onnx_path.exists():
            logger.info("Квантованная ONNX модель уже существует: %s", quant_onnx_path)
            return quant_onnx_path

        logger.info("Экспорт модели %s в формат ONNX...", base_name)
        model.eval()

        # Dummy inputs для трассировки
        dummy_text = "Тестовый запрос для экспорта"
        dummy_inputs = tokenizer(
            dummy_text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        }

        inputs_to_pass: tuple[torch.Tensor, ...] = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
        if "token_type_ids" in dummy_inputs:
            input_names.append("token_type_ids")
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
            inputs_to_pass = (
                dummy_inputs["input_ids"],
                dummy_inputs["attention_mask"],
                dummy_inputs["token_type_ids"],
            )

        # Экспорт через torch.onnx
        torch.onnx.export(
            model,
            inputs_to_pass,
            f=str(raw_onnx_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )

        logger.info("Применение dynamic INT8 квантования...")
        quantize_dynamic(
            model_input=raw_onnx_path,
            model_output=quant_onnx_path,
            weight_type=QuantType.QInt8,
        )

        # Удаляем промежуточный тяжелый fp32 onnx файл для экономии места
        if raw_onnx_path.exists():
            raw_onnx_path.unlink()

        # Проверка работоспособности через InferenceSession
        _ = onnxruntime.InferenceSession(str(quant_onnx_path), providers=["CPUExecutionProvider"])
        size_mb = quant_onnx_path.stat().st_size / (1024 * 1024)
        logger.info("ONNX INT8 модель готова: %s (размер: %.2f МБ)", quant_onnx_path, size_mb)

        return quant_onnx_path
