"""Contrastive Domain Fine-Tuning for Dense Bi-Encoders with LoRA and MNRL."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from meme_organizer.metrics.ranking import calculate_ranking_metrics
from meme_organizer.models.dense import DenseTextRetriever
from meme_organizer.models.onnx_exporter import OnnxExporter

if TYPE_CHECKING:
    from meme_organizer.core.tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class PairDataset(Dataset[dict[str, str]]):
    """Dataset of (query, meme_document) pairs."""

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, str]:
        q, d = self.pairs[idx]
        return {"query": q, "document": d}


class FineTunedDenseRetriever(DenseTextRetriever):
    """Fine-tunes a Dense Bi-Encoder using LoRA and in-batch negatives."""

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

    def fit(
        self,
        train_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
        val_pairs: list[dict[str, str]] | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Domain adaptation with LoRA and Multiple Negatives Ranking Loss."""
        corpus_dict: dict[str, str] = {}
        for d in corpus_docs:
            text = (
                f"{d.get('user_caption', '')} "
                f"{d.get('rich_description', '')} "
                f"{d.get('transcript_or_text', '')} "
                f"{' '.join(d.get('tags', []))}"
            ).strip()
            corpus_dict[d["file_hash"]] = text

        training_data: list[tuple[str, str]] = []
        for pair in train_pairs:
            f_hash = pair["file_hash"]
            if f_hash in corpus_dict:
                q_text = f"{self.query_prefix}{pair['query'].strip()}"
                d_text = f"{self.passage_prefix}{corpus_dict[f_hash]}"
                training_data.append((q_text, d_text))

        logger.info("Подготовлено %d пар для обучения модели %s...", len(training_data), self.model_name)

        target_modules = ["query", "key", "value", "dense"]
        if "e5" in self.model_name or "bert" in self.hf_model_id:
            target_modules = ["query", "value"]

        peft_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        peft_model = get_peft_model(self._model, peft_config)
        peft_model.train()

        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=self.lr)
        dataloader = DataLoader(PairDataset(training_data), batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                optimizer.zero_grad()
                q_enc = self._tokenizer(
                    batch["query"], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                )
                d_enc = self._tokenizer(
                    batch["document"], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                )

                q_out = peft_model(**q_enc)
                d_out = peft_model(**d_enc)

                q_embs = torch.nn.functional.normalize(
                    self._mean_pooling(q_out, q_enc["attention_mask"]), p=2, dim=1
                )
                d_embs = torch.nn.functional.normalize(
                    self._mean_pooling(d_out, d_enc["attention_mask"]), p=2, dim=1
                )

                scores = torch.matmul(q_embs, d_embs.T) * 20.0
                labels = torch.arange(len(batch["query"]), device=scores.device)
                loss = torch.nn.functional.cross_entropy(scores, labels)

                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            val_metrics_log: dict[str, float] = {"train_loss": avg_loss}
            if val_pairs:
                self._model = peft_model.merge_and_unload()
                val_mrr = self._eval_validation(val_pairs, corpus_docs)
                val_metrics_log.update(val_mrr)
                peft_model = get_peft_model(self._model, peft_config)

            if tracker:
                tracker.log_epoch_metrics(epoch=epoch, metrics=val_metrics_log)

            logger.info(
                "Эпоха %d/%d [%s]: loss=%.4f | val_MRR@5=%.4f",
                epoch,
                self.epochs,
                self.model_name,
                avg_loss,
                val_metrics_log.get("val_mrr_at_5", 0.0),
            )

        self._model = peft_model.merge_and_unload()

        # Экспорт дообученной модели в ONNX INT8
        safe_name = self.model_name.replace("/", "_")
        self._onnx_path = OnnxExporter.export_text_encoder(
            model=self._model,
            tokenizer=self._tokenizer,
            output_dir=self.onnx_dir,
            base_name=safe_name,
            max_length=self.max_length,
        )
        logger.info("Дообученная ONNX модель сохранена: %s", self._onnx_path)

    def _eval_validation(self, val_pairs: list[dict[str, str]], corpus_docs: list[dict[str, Any]]) -> dict[str, float]:
        corpus_embs = self.encode_corpus(corpus_docs)
        corpus_hash_to_idx = {d["file_hash"]: idx for idx, d in enumerate(corpus_docs)}

        val_queries = [p["query"] for p in val_pairs]
        target_indices = [corpus_hash_to_idx[p["file_hash"]] for p in val_pairs]

        q_embs = self.encode_queries(val_queries)
        scores = self.predict_scores(q_embs, corpus_embs)
        val_metrics = calculate_ranking_metrics(scores, target_indices, top_k=5)

        return {
            "val_recall_at_1": val_metrics.recall_at_1,
            "val_recall_at_5": val_metrics.recall_at_5,
            "val_mrr_at_5": val_metrics.mrr_at_5,
        }
