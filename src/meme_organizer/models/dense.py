"""Dense Text Bi-Encoder Retriever (Clean FP32 PyTorch for Research & MLflow)."""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from meme_organizer.config import AppConfig
from meme_organizer.core.contracts import BaseRetriever

logger = logging.getLogger(__name__)


class DenseTextRetriever(BaseRetriever):
    """Dense Bi-Encoder retriever running in clean FP32 PyTorch."""

    def __init__(
        self,
        model_name: str,
        hf_model_id: str,
        max_length: int = 64,
        batch_size: int = 32,
        query_prefix: str = "",
        passage_prefix: str = "",
        cache_dir: Path = Path(".cache/models/hf"),
        mlflow_run_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        self.hf_model_id = hf_model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.cache_dir = cache_dir
        self.mlflow_run_name = mlflow_run_name

        self._tokenizer: Any = None
        self._model: Any = None
        self._corpus_embeddings: np.ndarray | None = None
        self._hf_save_dir: Path | None = None
        self._loaded_from_mlflow: bool = False

    def load_or_build(self) -> None:
        """Load tokenizer and FP32 weights (from MLflow hf_model artifact if requested, else HF)."""
        if self.mlflow_run_name and self._try_load_from_mlflow():
            return

        logger.info("Загрузка FP32 модели %s (%s)...", self.model_name, self.hf_model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
        self._model = AutoModel.from_pretrained(self.hf_model_id)

    @staticmethod
    def _is_mlflow_reachable(tracking_uri: str) -> bool:
        """Fast TCP check to avoid 2-minute urllib3 retry hangs when MLflow is offline."""
        parsed = urlparse(tracking_uri)
        if parsed.scheme not in {"http", "https"}:
            return True
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    def _try_load_from_mlflow(self) -> bool:
        """Try loading fine-tuned FP32 HuggingFace model directory from a finished MLflow run."""
        import mlflow

        try:
            cfg = AppConfig()
            if not self._is_mlflow_reachable(cfg.mlflow_tracking_uri):
                logger.info("MLflow сервер (%s) недоступен, пропускаем загрузку.", cfg.mlflow_tracking_uri)
                return False
            client = mlflow.tracking.MlflowClient(cfg.mlflow_tracking_uri)
            exp = client.get_experiment_by_name(cfg.experiment_name)
            if not exp:
                return False

            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"tags.mlflow.runName = '{self.mlflow_run_name}' and attributes.status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs:
                return False

            run_id = runs[0].info.run_id
            artifacts = client.list_artifacts(run_id)
            if not any(a.path == "hf_model" for a in artifacts):
                return False

            dest_dir = self.cache_dir / "mlflow_downloaded" / run_id[:8]
            dest_dir.mkdir(parents=True, exist_ok=True)
            local_dir = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="hf_model",
                    dst_path=str(dest_dir),
                    tracking_uri=cfg.mlflow_tracking_uri,
                )
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
            self._model = AutoModel.from_pretrained(str(local_dir))
            self._loaded_from_mlflow = True
            logger.info("Загружена чистая FP32 модель из MLflow (run=%s): %s", run_id[:8], local_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Не удалось загрузить FP32 модель из MLflow (%s), fallback на базовый HF: %s",
                self.mlflow_run_name,
                exc,
            )
            return False
        else:
            return True

    def _mean_pooling(self, model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
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

    def get_artifact_dirs(self) -> dict[str, Path]:
        """Return saved FP32 HuggingFace model directory if available."""
        if self._hf_save_dir and self._hf_save_dir.exists():
            return {"hf_model": self._hf_save_dir}
        return {}

    def get_model_size_mb(self) -> float:
        """Memory footprint of FP32 PyTorch model weights in MB."""
        if self._model is not None:
            total_bytes = sum(p.numel() * p.element_size() for p in self._model.parameters())
            return float(total_bytes) / (1024 * 1024)
        return 0.0
