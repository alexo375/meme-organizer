"""Экспорт трансформеров PyTorch в формат FP32 ONNX с опциональным пулингом внутри графа."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class NormalizedEmbeddingPipeline(torch.nn.Module):
    """Оборачивает модель HuggingFace, добавляя mean-pooling и L2-нормализацию в граф.

    Вход: input_ids [batch, seq_len], attention_mask [batch, seq_len].
    Выход: sentence_embedding [batch, hidden_dim] с L2-нормой 1.0.
    """

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Выполняет прямой проход с mean-pooling и L2-нормализацией эмбеддингов."""
        if token_type_ids is not None:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        last_hidden_state = outputs.last_hidden_state  # [batch, seq, dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # L2-normalization for ready-to-use cosine similarity
        return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)


class OnnxExporter:
    """Экспортирует трансформер HuggingFace в модель формата FP32 ONNX."""

    @staticmethod
    def export_to_fp32_onnx(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: Path,
        base_name: str,
        max_length: int = 64,
        include_pooling: bool = True,
        overwrite: bool = False,
    ) -> Path:
        """Экспортирует модель и проверяет валидность ONNX-файла, возвращая путь к нему."""
        import onnxruntime

        output_dir.mkdir(parents=True, exist_ok=True)
        fp32_onnx_path = output_dir / f"{base_name}_fp32.onnx"

        if fp32_onnx_path.exists() and not overwrite:
            logger.info("FP32 ONNX модель уже существует: %s", fp32_onnx_path)
            return fp32_onnx_path

        logger.info("Экспорт модели %s в формат FP32 ONNX (include_pooling=%s)...", base_name, include_pooling)
        model.eval()

        export_model: torch.nn.Module = NormalizedEmbeddingPipeline(model) if include_pooling else model
        export_model.eval()

        dummy_text = "Тестовый запрос для экспорта"
        dummy_inputs = tokenizer(
            dummy_text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_names = ["input_ids", "attention_mask"]
        batch_dim = torch.export.Dim("batch_size", min=1)
        seq_dim = torch.export.Dim("sequence_length", min=1)
        dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] = (
            {0: batch_dim, 1: seq_dim},
            {0: batch_dim, 1: seq_dim},
        )

        output_names = ["sentence_embedding"] if include_pooling else ["last_hidden_state"]

        inputs_to_pass: tuple[torch.Tensor, ...] = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
        if "token_type_ids" in dummy_inputs and not include_pooling:
            input_names.append("token_type_ids")
            dynamic_shapes = (*dynamic_shapes, {0: batch_dim, 1: seq_dim})
            inputs_to_pass = (
                dummy_inputs["input_ids"],
                dummy_inputs["attention_mask"],
                dummy_inputs["token_type_ids"],
            )

        torch.onnx.export(
            export_model,
            inputs_to_pass,
            f=str(fp32_onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            do_constant_folding=True,
        )

        # Enforce ONNX IR version <= 9 for Android onnxruntime-android 1.17.1 compatibility
        import onnx
        model_proto = onnx.load(str(fp32_onnx_path), load_external_data=True)
        if model_proto.ir_version > 9:
            model_proto.ir_version = 9
            onnx.save_model(model_proto, str(fp32_onnx_path), save_as_external_data=False)

        _ = onnxruntime.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"])
        size_mb = fp32_onnx_path.stat().st_size / (1024 * 1024)
        logger.info("FP32 ONNX модель готова: %s (размер: %.2f МБ)", fp32_onnx_path, size_mb)

        return fp32_onnx_path

    @staticmethod
    def export_tokenizer_assets(tokenizer: PreTrainedTokenizerBase, output_dir: Path) -> list[Path]:
        """Сохраняет файлы токенизатора в целевую директорию и возвращает список путей к ним."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = tokenizer.save_pretrained(str(output_dir))
        return [Path(p) for p in saved_files]
