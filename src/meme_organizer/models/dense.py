"""Dense Text Bi-Encoder Retriever (Zero-Shot & ONNX Mobile)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from meme_organizer.core.contracts import BaseRetriever
from meme_organizer.models.onnx_exporter import OnnxExporter

logger = logging.getLogger(__name__)


class DenseTextRetriever(BaseRetriever):
    """Dense Bi-Encoder retriever running on CPU or ONNX INT8."""

    def __init__(
        self,
        model_name: str,
        hf_model_id: str,
        max_length: int = 64,
        batch_size: int = 32,
        query_prefix: str = "",
        passage_prefix: str = "",
        onnx_dir: Path = Path(".cache/models/onnx"),
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        self.hf_model_id = hf_model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.onnx_dir = onnx_dir

        self._tokenizer: Any = None
        self._model: Any = None
        self._corpus_embeddings: np.ndarray | None = None
        self._onnx_path: Path | None = None

    def load_or_build(self) -> None:
        """Load tokenizer and weights, and export/cache ONNX INT8."""
        logger.info("Загрузка модели %s (%s)...", self.model_name, self.hf_model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        self._model = AutoModel.from_pretrained(self.hf_model_id)

        # Экспорт в ONNX INT8
        safe_name = self.model_name.replace("/", "_")
        self._onnx_path = OnnxExporter.export_text_encoder(
            model=self._model,
            tokenizer=self._tokenizer,
            output_dir=self.onnx_dir,
            base_name=safe_name,
            max_length=self.max_length,
        )

    def _mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        self._model.eval()

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                outputs = self._model(**encoded)
                embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
                # L2 normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> np.ndarray:
        """Encode meme text into dense vectors."""
        logger.info("Кодирование корпуса (%d документов) моделью %s...", len(corpus_docs), self.model_name)
        texts: list[str] = []
        for doc in corpus_docs:
            combined = (
                f"{doc.get('user_caption', '')} "
                f"{doc.get('rich_description', '')} "
                f"{doc.get('transcript_or_text', '')} "
                f"{' '.join(doc.get('tags', []))}"
            ).strip()
            texts.append(f"{self.passage_prefix}{combined}")

        self._corpus_embeddings = self._encode_texts(texts)
        return self._corpus_embeddings

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode queries into normalized dense vectors."""
        prefixed_queries = [f"{self.query_prefix}{q.strip()}" for q in queries]
        return self._encode_texts(prefixed_queries)

    def predict_scores(self, query_repr: np.ndarray, corpus_repr: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix via dot product of normalized vectors."""
        return np.asarray(np.dot(query_repr, corpus_repr.T))

    def get_onnx_paths(self) -> list[Path]:
        """Return cached ONNX model file path."""
        if self._onnx_path and self._onnx_path.exists():
            return [self._onnx_path]
        return []

    def get_model_size_mb(self) -> float:
        """Disk size of quantized ONNX model."""
        if self._onnx_path and self._onnx_path.exists():
            return self._onnx_path.stat().st_size / (1024 * 1024)
        return 0.0
