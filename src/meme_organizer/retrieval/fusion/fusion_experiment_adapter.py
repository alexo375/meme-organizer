"""Multi-channel fusion experiment adapter for BaseExperimentPipeline."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import mlflow
import numpy as np
import optuna
import torch
from torch import nn

from meme_organizer.domain.base_experiment_adapter import BaseExperimentAdapter
from meme_organizer.domain.ranking_metrics import RankingMetrics
from meme_organizer.evaluation.metrics.error_analyzer import ErrorAnalyzer
from meme_organizer.retrieval.fusion.gated_attention_fusion import GatedAttentionFusion
from meme_organizer.retrieval.fusion.multi_channel_scorer import MultiChannelScorer

if TYPE_CHECKING:
    from meme_organizer.evaluation.metrics.experiment_tracker import ExperimentTracker

FusionMethod = Literal["rrf", "optuna", "gated_attention"]


class FusionExperimentAdapter(BaseExperimentAdapter):
    """Runs 4-channel fusion (BM25, Meta, OCR, ASR) with full latency, error, and artifact logging."""

    def __init__(
        self,
        run_name: str,
        method: FusionMethod,
        q_train: np.ndarray,
        s_train: np.ndarray,
        gt_train: list[int],
        q_val: np.ndarray,
        s_val: np.ndarray,
        gt_val: list[int],
        q_test: np.ndarray,
        s_test: np.ndarray,
        gt_test: list[int],
        channel_mask: np.ndarray,
        test_queries: list[str],
        corpus_docs: list[dict[str, Any]],
        n_trials: int = 60,
        epochs: int = 60,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Initialize fusion adapter with precomputed 4-channel matrices and split ground truth."""
        super().__init__(run_name=run_name, tags=tags or {"domain": "fusion", "method": method})
        self.method = method
        self.q_train, self.s_train, self.gt_train = q_train, s_train, gt_train
        self.q_val, self.s_val, self.gt_val = q_val, s_val, gt_val
        self.q_test, self.s_test, self.gt_test = q_test, s_test, gt_test
        self.channel_mask = channel_mask
        self.test_queries = test_queries
        self.corpus_docs = corpus_docs
        self.n_trials = n_trials
        self.epochs = epochs

        self._metrics: RankingMetrics | None = None
        self._error_rows: list[dict[str, Any]] = []
        self._latencies_ms: list[float] = []
        self._eval_sec: float = 0.0
        self._model_size_mb: float = 0.0
        self._best_weights: np.ndarray | None = None
        self._artifact_dir: tempfile.TemporaryDirectory[str] | None = None
        self._artifact_path: Path | None = None

    def get_params(self) -> dict[str, Any]:
        """Return fusion method and channel configuration."""
        params: dict[str, Any] = {
            "method": self.method,
            "channels": "bm25,meta,ocr,asr",
            "num_queries": len(self.gt_test),
            "corpus_size": len(self.corpus_docs),
        }
        if self.method == "optuna":
            params["n_trials"] = self.n_trials
        elif self.method == "gated_attention":
            params["epochs"] = self.epochs
        return params

    def prepare(self) -> None:
        """Reset per-run state."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._latencies_ms.clear()

    def execute(self, tracker: ExperimentTracker | None = None) -> None:
        """Обучает параметры объединения, оценивает тестовые запросы и замеряет задержку."""
        if self.method == "rrf":
            scores = self._run_rrf()
        elif self.method == "optuna":
            scores = self._run_optuna(tracker)
        else:
            scores = self._run_gated_attention()

        self._metrics = RankingMetrics.calculate(scores, self.gt_test)
        self._error_rows = ErrorAnalyzer().find_worst_queries(
            scores, self.gt_test, self.test_queries, [d["silver_filename"] for d in self.corpus_docs]
        )

    def _run_rrf(self) -> np.ndarray:
        t0 = time.perf_counter()
        scores = MultiChannelScorer.rrf_fuse([self.s_test[:, :, i] for i in range(self.s_test.shape[2])])
        self._eval_sec = time.perf_counter() - t0
        for qi in range(len(self.gt_test)):
            tq = time.perf_counter()
            _ = MultiChannelScorer.rrf_fuse([self.s_test[qi : qi + 1, :, i] for i in range(self.s_test.shape[2])])
            self._latencies_ms.append((time.perf_counter() - tq) * 1000.0)
        self._model_size_mb = 0.0
        return scores

    def _run_optuna(self, tracker: ExperimentTracker | None) -> np.ndarray:
        best_val = 0.0

        def obj(trial: optuna.Trial) -> float:
            nonlocal best_val
            w = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(4)])
            w = w / max(float(w.sum()), 1e-9)
            val_mrr = RankingMetrics.calculate((self.s_val * w).sum(axis=-1), self.gt_val).mrr_at_5
            best_val = max(best_val, val_mrr)
            if tracker is not None and mlflow.active_run():
                mlflow.log_metrics({"trial_val_mrr": val_mrr, "best_val_mrr": best_val}, step=trial.number)
            return val_mrr

        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=self.n_trials)
        w_best = np.array([study.best_params[f"w{i}"] for i in range(4)])
        self._best_weights = w_best / w_best.sum()
        if mlflow.active_run():
            mlflow.log_params(
                {
                    f"best_w_{c}": round(float(self._best_weights[i]), 4)
                    for i, c in enumerate(["bm25", "meta", "ocr", "asr"])
                }
            )

        t0 = time.perf_counter()
        scores = (self.s_test * self._best_weights).sum(axis=-1)
        self._eval_sec = time.perf_counter() - t0
        for qi in range(len(self.gt_test)):
            tq = time.perf_counter()
            _ = (self.s_test[qi : qi + 1] * self._best_weights).sum(axis=-1)
            self._latencies_ms.append((time.perf_counter() - tq) * 1000.0)
        self._model_size_mb = 0.0
        return np.asarray(scores, dtype=np.float32)

    def _run_gated_attention(self) -> np.ndarray:
        gate = GatedAttentionFusion(query_dim=self.q_train.shape[1], num_channels=4, hidden_dim=64)
        opt = torch.optim.AdamW(gate.parameters(), lr=5e-3, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        q_t = torch.from_numpy(self.q_train).float()
        s_t = torch.from_numpy(self.s_train).float()
        m_t = torch.from_numpy(self.channel_mask).bool()
        y_t = torch.tensor(self.gt_train, dtype=torch.long)
        q_vt = torch.from_numpy(self.q_val).float()
        s_vt = torch.from_numpy(self.s_val).float()

        for ep in range(self.epochs):
            gate.train()
            opt.zero_grad()
            loss = loss_fn(gate(q_t, s_t, m_t) * 10.0, y_t)
            loss.backward()
            opt.step()
            if (ep + 1) % 5 == 0 and mlflow.active_run():
                gate.eval()
                with torch.no_grad():
                    val_mrr = RankingMetrics.calculate(gate(q_vt, s_vt, m_t).numpy(), self.gt_val).mrr_at_5
                mlflow.log_metrics({"train_loss": float(loss.item()), "val_mrr_at_5": val_mrr}, step=ep + 1)

        gate.eval()
        q_te_t = torch.from_numpy(self.q_test).float()
        s_te_t = torch.from_numpy(self.s_test).float()
        t0 = time.perf_counter()
        with torch.no_grad():
            scores = gate(q_te_t, s_te_t, m_t).numpy()
        self._eval_sec = time.perf_counter() - t0

        with torch.no_grad():
            for qi in range(len(self.gt_test)):
                tq = time.perf_counter()
                _ = gate(q_te_t[qi : qi + 1], s_te_t[qi : qi + 1], m_t)
                self._latencies_ms.append((time.perf_counter() - tq) * 1000.0)

        self._artifact_dir = tempfile.TemporaryDirectory()
        self._artifact_path = Path(self._artifact_dir.name) / "gated_attention_fusion.pt"
        torch.save(gate.state_dict(), self._artifact_path)
        self._model_size_mb = round(self._artifact_path.stat().st_size / (1024 * 1024), 4)
        return np.asarray(scores, dtype=np.float32)

    def compute_quality_metrics(self) -> dict[str, float]:
        """Return MRR@5, Recall@1, Recall@5, and NDCG@5."""
        return self._metrics.to_dict() if self._metrics is not None else {}

    def compute_performance_metrics(self) -> dict[str, float]:
        """Return fusion latency, throughput, and gate model size."""
        return {
            "cpu_latency_ms": round(float(np.percentile(self._latencies_ms, 50)), 4) if self._latencies_ms else 0.0,
            "cpu_latency_p95_ms": round(float(np.percentile(self._latencies_ms, 95)), 4) if self._latencies_ms else 0.0,
            "eval_inference_time_sec": round(self._eval_sec, 4),
            "eval_throughput_qps": round(len(self.gt_test) / max(self._eval_sec, 1e-6), 2),
            "model_size_mb": self._model_size_mb,
        }

    def get_error_table(self) -> list[dict[str, Any]]:
        """Return Top-15 worst queries."""
        return self._error_rows

    def get_model_artifacts(self) -> dict[str, Path]:
        """Return trained GatedAttentionFusion weights if available."""
        return {"gate_weights": self._artifact_path} if self._artifact_path is not None else {}
