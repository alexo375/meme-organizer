"""Адаптер дообучения Dense Bi-Encoder модели с Multiple Negatives Ranking Loss (MNRL) и LoRA."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from meme_organizer.evaluation.metrics import RetrievalEvaluator
from meme_organizer.retrieval.adapters.dense_text import DenseTextRetrieverAdapter
from meme_organizer.retrieval.onnx.exporter import OnnxExporter

if TYPE_CHECKING:
    from meme_organizer.retrieval.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


class PairDataset(Dataset[dict[str, str]]):
    """Датасет пар (запрос, текст мема)."""

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, str]:
        q, d = self.pairs[idx]
        return {"query": q, "document": d}


class FineTunedDenseAdapter(DenseTextRetrieverAdapter):
    """Адаптер для Contrastive Domain Fine-Tuning на базе LoRA и Multiple Negatives Ranking Loss."""

    def __init__(
        self,
        model_name: str = "rubert-tiny2-finetuned",
        base_model_id: str = "cointegrated/rubert-tiny2",
        epochs: int = 4,
        lr: float = 2e-4,
        batch_size: int = 32,
        max_length: int = 64,
        lora_r: int = 8,
        lora_alpha: int = 16,
        query_prefix: str = "",
        passage_prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            hf_model_id=base_model_id,
            max_length=max_length,
            batch_size=batch_size,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            **kwargs,
        )
        self.epochs = epochs
        self.lr = lr
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

    def _prepare_train_tuples(
        self,
        train_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        corpus_text_map = {}
        for doc in corpus_docs:
            u_caption = doc.get("user_caption") or ""
            rich_desc = doc.get("rich_description") or ""
            transcript = doc.get("transcript_or_text") or ""
            tags = " ".join(doc.get("tags") or [])
            full_text = f"{u_caption} {tags} {transcript} {rich_desc}".strip()
            corpus_text_map[doc["file_hash"]] = f"{self.passage_prefix}{full_text}"

        tuples: list[tuple[str, str]] = []
        for p in train_pairs:
            doc_text = corpus_text_map.get(p["file_hash"], "")
            q_clean = p["query"].strip()
            if doc_text and q_clean:
                tuples.append((f"{self.query_prefix}{q_clean}", doc_text))
        return tuples

    def _train_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader[dict[str, str]],
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
    ) -> float:
        model.train()
        losses: list[float] = []
        for batch in loader:
            q_enc = self._tokenizer(
                batch["query"],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            d_enc = self._tokenizer(
                batch["document"],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            optimizer.zero_grad()
            q_out = model(**q_enc)
            d_out = model(**d_enc)
            q_emb = torch.nn.functional.normalize(self._mean_pooling(q_out, q_enc["attention_mask"]), p=2, dim=1)
            d_emb = torch.nn.functional.normalize(self._mean_pooling(d_out, d_enc["attention_mask"]), p=2, dim=1)
            sim_matrix = torch.matmul(q_emb, d_emb.T) / 0.05
            labels = torch.arange(sim_matrix.size(0))
            loss = loss_fn(sim_matrix, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def _validate_epoch(
        self,
        corpus_docs: list[dict[str, Any]],
        val_queries: list[str],
        val_gt_indices: list[int],
        corpus_hashes: list[str],
    ) -> dict[str, float]:
        c_emb = self.encode_corpus(corpus_docs)
        q_emb_val = self.encode_queries(val_queries)
        scores = self.predict_scores(q_emb_val, c_emb)
        v_m, _ = RetrievalEvaluator.compute_metrics(
            scores_matrix=scores,
            ground_truth_indices=val_gt_indices,
            queries=val_queries,
            corpus_hashes=corpus_hashes,
            top_k=5,
        )
        return {
            "val_mrr_at_5": v_m.mrr_at_5,
            "val_recall_at_1": v_m.recall_at_1,
            "val_recall_at_5": v_m.recall_at_5,
        }

    def fit(
        self,
        train_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
        val_pairs: list[dict[str, str]] | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Дообучает модель с Multiple Negatives Ranking Loss (In-Batch Negatives) и логирует эпохи."""
        train_tuples = self._prepare_train_tuples(train_pairs, corpus_docs)
        logger.info("Подготовлено %d пар для обучения модели %s...", len(train_tuples), self.model_name)

        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        peft_model = get_peft_model(self._model, peft_config=lora_cfg)
        loader = DataLoader(PairDataset(train_tuples), batch_size=self.batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=self.lr, weight_decay=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        corpus_hash_to_idx = {doc["file_hash"]: i for i, doc in enumerate(corpus_docs)}
        corpus_hashes = [doc["file_hash"] for doc in corpus_docs]
        val_queries = [p["query"] for p in val_pairs] if val_pairs else []
        val_gt = [corpus_hash_to_idx[p["file_hash"]] for p in val_pairs] if val_pairs else []

        for epoch in range(1, self.epochs + 1):
            avg_loss = self._train_epoch(peft_model, loader, optimizer, loss_fn)
            peft_model.eval()
            self._model = peft_model.merge_and_unload()
            metrics: dict[str, float] = {"train_loss": avg_loss}

            if val_pairs and val_queries:
                v_metrics = self._validate_epoch(corpus_docs, val_queries, val_gt, corpus_hashes)
                metrics.update(v_metrics)
                logger.info(
                    "Эпоха %d/%d [%s]: loss=%.4f | val_MRR@5=%.4f",
                    epoch,
                    self.epochs,
                    self.model_name,
                    avg_loss,
                    v_metrics["val_mrr_at_5"],
                )

            if tracker:
                tracker.log_epoch_metrics(epoch=epoch, metrics=metrics)

            if epoch < self.epochs:
                peft_model = get_peft_model(self._model, peft_config=lora_cfg)

        try:
            self._onnx_path = OnnxExporter.export_text_encoder(
                model=self._model,
                tokenizer=self._tokenizer,
                output_dir=self.onnx_dir,
                base_name=self.model_name.replace("/", "_"),
                max_length=self.max_length,
            )
            logger.info("Дообученная ONNX модель сохранена: %s", self._onnx_path)
        except Exception:
            logger.exception("Ошибка экспорта дообученной модели в ONNX")
