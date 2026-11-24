"""Module for exporting PyTorch Transformer models to ONNX INT8 with in-graph pooling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class NormalizedEmbeddingPipeline(torch.nn.Module):
    """Wraps HuggingFace model with Mean-Pooling and L2-Normalization in graph.

    Input: input_ids [batch, seq_len], attention_mask [batch, seq_len]
    Output: sentence_embedding [batch, hidden_dim] with L2 norm = 1.0
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
        """Forward pass with Mean Pooling and L2 Normalization."""
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
    """Exports HuggingFace Transformer model to ONNX with dynamic INT8 quantization."""

    @staticmethod
    def export_text_encoder(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: Path,
        base_name: str,
        max_length: int = 64,
        include_pooling: bool = True,
    ) -> Path:
        """Export transformer into ONNX INT8."""
        import onnxruntime
        from onnxruntime.quantization import QuantType, quantize_dynamic

        output_dir.mkdir(parents=True, exist_ok=True)
        raw_onnx_path = output_dir / f"{base_name}_fp32.onnx"
        quant_onnx_path = output_dir / f"{base_name}_int8.onnx"

        if quant_onnx_path.exists():
            logger.info("Квантованная ONNX модель уже существует: %s", quant_onnx_path)
            return quant_onnx_path

        logger.info("Экспорт модели %s в формат ONNX (include_pooling=%s)...", base_name, include_pooling)
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
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }

        if include_pooling:
            output_names = ["sentence_embedding"]
            dynamic_axes["sentence_embedding"] = {0: "batch_size"}
        else:
            output_names = ["last_hidden_state"]
            dynamic_axes["last_hidden_state"] = {0: "batch_size", 1: "sequence_length"}

        inputs_to_pass: tuple[torch.Tensor, ...] = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
        if "token_type_ids" in dummy_inputs and not include_pooling:
            input_names.append("token_type_ids")
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
            inputs_to_pass = (
                dummy_inputs["input_ids"],
                dummy_inputs["attention_mask"],
                dummy_inputs["token_type_ids"],
            )

        torch.onnx.export(
            export_model,
            inputs_to_pass,
            f=str(raw_onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=18,
            do_constant_folding=True,
        )

        logger.info("Применение dynamic INT8 квантования...")
        quantize_dynamic(
            model_input=raw_onnx_path,
            model_output=quant_onnx_path,
            weight_type=QuantType.QInt8,
        )

        # Удаляем промежуточный тяжелый fp32 onnx файл и данные
        if raw_onnx_path.exists():
            raw_onnx_path.unlink()
        fp32_data = output_dir / f"{base_name}_fp32.onnx.data"
        if fp32_data.exists():
            fp32_data.unlink()

        _ = onnxruntime.InferenceSession(str(quant_onnx_path), providers=["CPUExecutionProvider"])
        size_mb = quant_onnx_path.stat().st_size / (1024 * 1024)
        logger.info("ONNX INT8 модель готова: %s (размер: %.2f МБ)", quant_onnx_path, size_mb)

        return quant_onnx_path

    @staticmethod
    def export_tokenizer_assets(tokenizer: PreTrainedTokenizerBase, output_dir: Path) -> list[Path]:
        """Save all tokenizer files needed by mobile client or HF Tokenizers."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = tokenizer.save_pretrained(str(output_dir))
        return [Path(p) for p in saved_files]
