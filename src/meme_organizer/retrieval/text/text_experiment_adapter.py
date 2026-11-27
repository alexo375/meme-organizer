"""Text encoder/retriever experiment adapter for BaseExperimentPipeline."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from meme_organizer.domain.base_experiment_adapter import BaseExperimentAdapter
from meme_organizer.evaluation.pipeline.retrieval_evaluator import RetrievalEvaluator

if TYPE_CHECKING:
    from pathlib import Path

    from meme_organizer.domain.base_retriever import BaseRetriever
    from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker
    from meme_organizer.evaluation.pipeline.evaluation_dataset_preparer import EvaluationSplitData
    from meme_organizer.evaluation.pipeline.retrieval_evaluator import EvaluationResult


class TextExperimentAdapter(BaseExperimentAdapter):
    """Runs text encoder/retriever evaluation on a prepared split and logs standard metrics."""

    def __init__(
        self,
        run_name: str,
        retriever: BaseRetriever,
        split_data: EvaluationSplitData,
        train_mode: bool = False,
        onnx_path: Path | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize text experiment adapter with retriever, split data, and optional ONNX path."""
        super().__init__(run_name=run_name, tags=tags or {"domain": "text"})
        self.retriever = retriever
        self.split_data = split_data
        self.train_mode = train_mode
        self.onnx_path = onnx_path

        self._eval_result: EvaluationResult | None = None
        self._train_sec: float = 0.0

    def get_params(self) -> dict[str, Any]:
        """Return model name, split metadata, and training flag."""
        return {
            "model_name": getattr(self.retriever, "model_name", self.retriever.__class__.__name__),
            "train_mode": self.train_mode,
            "corpus_size": len(self.split_data.corpus_docs),
            "num_queries": len(self.split_data.queries),
        }

    def prepare(self) -> None:
        """Load or initialize the underlying text model."""
        self.retriever.load_or_build()

    def execute(self, tracker: ExperimentTracker | None = None) -> None:
        """Optionally fit and run RetrievalEvaluator."""
        if self.train_mode and self.split_data.train_pairs:
            t_train = time.perf_counter()
            self.retriever.fit(
                self.split_data.train_pairs,
                self.split_data.corpus_docs,
                tracker=tracker,
                val_pairs=self.split_data.val_pairs,
            )
            self._train_sec = time.perf_counter() - t_train

        self._eval_result = RetrievalEvaluator().evaluate(self.retriever, self.split_data)

    def compute_quality_metrics(self) -> dict[str, float]:
        """Return MRR@5, Recall@1, Recall@5, and NDCG@5."""
        if self._eval_result is None:
            return {}
        return self._eval_result.metrics.to_dict()

    def compute_performance_metrics(self) -> dict[str, float]:
        """Return latency, throughput, training time, and model size."""
        if self._eval_result is None:
            return {}
        p = self._eval_result.profile
        return {
            "cpu_latency_ms": p.cpu_latency_ms,
            "cpu_latency_p95_ms": p.cpu_latency_p95_ms,
            "eval_throughput_qps": round(
                len(self.split_data.queries) / max(self._eval_result.eval_inference_time_sec, 1e-6), 2
            ),
            "training_time_sec": round(self._train_sec, 3),
            "eval_inference_time_sec": round(self._eval_result.eval_inference_time_sec, 3),
            "model_size_mb": p.model_size_mb,
        }

    def get_error_table(self) -> list[dict[str, Any]]:
        """Return Top-15 worst queries."""
        if self._eval_result is None:
            return []
        return self._eval_result.worst_queries

    def get_model_artifacts(self) -> dict[str, Path]:
        """Return raw ONNX artifact path if provided."""
        return {"onnx_raw": self.onnx_path} if self.onnx_path is not None else {}
