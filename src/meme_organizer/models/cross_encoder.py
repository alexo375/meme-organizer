"""Cross-Encoder Reranker using on-device ONNX INT8 runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cointegrated/rubert-tiny-bilingual-nli"
DEFAULT_MODELS_DIR = Path(".cache/models")


class CrossEncoderReranker:
    """Lightweight Cross-Encoder for second-stage candidate re-ranking."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        models_dir: Path = DEFAULT_MODELS_DIR,
    ) -> None:
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.safe_name = model_name.replace("/", "_")
        self.onnx_path = self.models_dir / "onnx" / f"{self.safe_name}_int8.onnx"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.session: ort.InferenceSession | None = None

    def export_and_quantize(self) -> Path:
        """Export Cross-Encoder to ONNX and quantize to INT8."""
        if self.onnx_path.exists():
            logger.info("ONNX INT8 кросс-энкодер уже существует: %s", self.onnx_path)
            return self.onnx_path

        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from transformers import AutoModelForSequenceClassification

        self.models_dir.mkdir(parents=True, exist_ok=True)
        onnx_dir = self.models_dir / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        fp32_path = onnx_dir / f"{self.safe_name}_fp32.onnx"

        logger.info("Экспорт кросс-энкодера %s в ONNX FP32...", self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()

        dummy_tokens = self.tokenizer(
            "тестовый запрос",
            "тестовый документ",
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )

        input_names = ["input_ids", "attention_mask"]
        inputs: tuple[Any, ...] = (dummy_tokens["input_ids"], dummy_tokens["attention_mask"])
        dynamic_axes: dict[str, dict[int, str]] = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        }
        if "token_type_ids" in dummy_tokens:
            input_names.append("token_type_ids")
            inputs = (
                dummy_tokens["input_ids"],
                dummy_tokens["attention_mask"],
                dummy_tokens["token_type_ids"],
            )
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "seq_len"}

        torch.onnx.export(
            model,
            inputs,
            str(fp32_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )

        logger.info("Квантование кросс-энкодера в INT8...")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(self.onnx_path),
            weight_type=QuantType.QInt8,
        )

        if fp32_path.exists():
            fp32_path.unlink()
        fp32_data = onnx_dir / f"{self.safe_name}_fp32.onnx.data"
        if fp32_data.exists():
            fp32_data.unlink()

        size_mb = self.onnx_path.stat().st_size / (1024 * 1024)
        logger.info("ONNX INT8 кросс-энкодер готов: %s (%.2f МБ)", self.onnx_path, size_mb)
        return self.onnx_path

    def _ensure_session(self) -> ort.InferenceSession:
        if self.session is None:
            self.export_and_quantize()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])
        return self.session

    def predict_pair_scores(self, pairs: list[tuple[str, str]], max_length: int = 128) -> np.ndarray:
        """Score (query, document) pairs using NLI entailment logit."""
        session = self._ensure_session()
        queries = [p[0] for p in pairs]
        passages = [p[1] for p in pairs]

        encoded = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in encoded:
            session_input_names = [inp.name for inp in session.get_inputs()]
            if "token_type_ids" in session_input_names:
                ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

        outputs = session.run(None, ort_inputs)
        logits = outputs[0]

        # 0: not_entailment, 1: entailment
        if logits.ndim == 2:
            if logits.shape[1] == 1:
                return np.asarray(logits.squeeze(-1))
            if logits.shape[1] == 2:
                return np.asarray(logits[:, 1])
            if logits.shape[1] == 3:
                return np.asarray(logits[:, 2])
        return np.asarray(logits.squeeze())
