"""Универсальный адаптер для плотных текстовых Bi-Encoder моделей (Transformers / Sentence-Transformers)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from meme_organizer.retrieval.base import BaseRetrieverAdapter
from meme_organizer.retrieval.onnx.exporter import OnnxExporter

logger = logging.getLogger(__name__)


class DenseTextRetrieverAdapter(BaseRetrieverAdapter):
    """Адаптер для Dense Bi-Encoder моделей (например, rubert-tiny2 или multilingual-e5-small)."""

    def __init__(
        self,
        model_name: str,
        hf_model_id: str,
        max_length: int = 64,
        batch_size: int = 32,
        query_prefix: str = "",
        passage_prefix: str = "",
        onnx_dir: Path = Path("models/onnx"),
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
        """Загружает модель и токенизатор из HuggingFace и экспортирует ONNX INT8."""
        logger.info("Загрузка модели %s (%s)...", self.model_name, self.hf_model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        self._model = AutoModel.from_pretrained(self.hf_model_id)
        self._model.eval()

        # Автоматический экспорт и замер веса мобильного артефакта
        try:
            self._onnx_path = OnnxExporter.export_text_encoder(
                model=self._model,
                tokenizer=self._tokenizer,
                output_dir=self.onnx_dir,
                base_name=self.model_name.replace("/", "_"),
                max_length=self.max_length,
            )
        except Exception:
            logger.exception("Ошибка экспорта в ONNX, продолжаем на PyTorch")

    @staticmethod
    def _mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Усредняет токены с учетом маски внимания (Mean Pooling)."""
        token_embeddings = model_output[0]  # First element contains hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Кодирует список текстов в нормализованные эмбеддинги."""
        all_embeddings: list[np.ndarray] = []

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
                # L2 нормализация векторов к единичной длине
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        if not all_embeddings:
            return np.empty((0, 312), dtype=np.float32)

        return np.vstack(all_embeddings)

    def encode_corpus(self, corpus_docs: list[dict[str, Any]]) -> np.ndarray:
        """Формирует текстовые представления мемов и кодирует их в матрицу эмбеддингов."""
        texts: list[str] = []
        for doc in corpus_docs:
            user_caption = doc.get("user_caption") or ""
            rich_desc = doc.get("rich_description") or ""
            transcript = doc.get("transcript_or_text") or ""
            tags = " ".join(doc.get("tags") or [])

            full_text = f"{user_caption} {tags} {transcript} {rich_desc}".strip()
            prefixed = f"{self.passage_prefix}{full_text}"
            texts.append(prefixed)

        self._corpus_embeddings = self._encode_texts(texts)
        return self._corpus_embeddings

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Кодирует список поисковых запросов."""
        prefixed_queries = [f"{self.query_prefix}{q}" for q in queries]
        return self._encode_texts(prefixed_queries)

    def predict_scores(self, query_repr: np.ndarray, corpus_repr: np.ndarray) -> np.ndarray:
        """Скалярное произведение между нормализованными векторами = Косинусное сходство."""
        scores: np.ndarray = np.dot(query_repr, corpus_repr.T)
        return scores

    def get_model_size_mb(self) -> float:
        """Возвращает размер сгенерированного ONNX INT8 файла."""
        if self._onnx_path and self._onnx_path.exists():
            return self._onnx_path.stat().st_size / (1024 * 1024)
        return 0.0
