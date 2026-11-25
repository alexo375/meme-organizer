"""Contrastive Domain Fine-Tuning for Dense Bi-Encoders with LoRA, MNRL, and Early Stopping."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from meme_organizer.metrics.ranking import calculate_ranking_metrics
from meme_organizer.models.dense import DenseTextRetriever

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
    """Fine-tunes a Dense Bi-Encoder using LoRA, in-batch negatives, and Early Stopping."""

    def __init__(
        self,
        model_name: str = "rubert-tiny2-finetuned",
        base_model_id: str = "cointegrated/rubert-tiny2",
        epochs: int = 50,
        patience: int = 10,
        lr: float = 2e-4,
        batch_size: int = 32,
        max_length: int = 64,
        lora_r: int = 8,
        lora_alpha: int = 16,
        query_prefix: str = "",
        passage_prefix: str = "",
        seed: int = 42,
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
        self.patience = patience
        self.lr = lr
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.seed = seed

    def _build_pair_tuples(self, pairs: list[dict[str, str]], corpus_dict: dict[str, str]) -> list[tuple[str, str]]:
        """Format (query, document) pairs with model prefixes."""
        formatted: list[tuple[str, str]] = []
        for pair in pairs:
            f_hash = pair["file_hash"]
            if f_hash in corpus_dict:
                q_text = f"{self.query_prefix}{pair['query'].strip()}"
                d_text = f"{self.passage_prefix}{corpus_dict[f_hash]}"
                formatted.append((q_text, d_text))
        return formatted

    def _compute_mnrl_loss(self, model: torch.nn.Module, queries: list[str], documents: list[str]) -> torch.Tensor:
        """Compute Multiple Negatives Ranking Loss for a batch of queries and documents."""
        q_enc = self._tokenizer(queries, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        d_enc = self._tokenizer(
            documents, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        q_out = model(**q_enc)
        d_out = model(**d_enc)

        q_embs = torch.nn.functional.normalize(self._mean_pooling(q_out, q_enc["attention_mask"]), p=2, dim=1)
        d_embs = torch.nn.functional.normalize(self._mean_pooling(d_out, d_enc["attention_mask"]), p=2, dim=1)

        scores = torch.matmul(q_embs, d_embs.T) * 20.0
        labels = torch.arange(len(queries), device=scores.device)
        return torch.nn.functional.cross_entropy(scores, labels)

    def fit(  # noqa: PLR0915
        self,
        train_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
        val_pairs: list[dict[str, str]] | None = None,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        """Domain adaptation with LoRA, MNRL, and Early Stopping on validation MRR@5."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            self._model = AutoModel.from_pretrained(self.hf_model_id)

        corpus_dict: dict[str, str] = {}
        for d in corpus_docs:
            text = (
                f"{d.get('user_caption', '')} "
                f"{d.get('rich_description', '')} "
                f"{d.get('transcript_or_text', '')} "
                f"{' '.join(d.get('tags', []))}"
            ).strip()
            corpus_dict[d["file_hash"]] = text

        training_data = self._build_pair_tuples(train_pairs, corpus_dict)
        logger.info(
            "Подготовлено %d пар для обучения модели %s (epochs=%d, patience=%d)...",
            len(training_data),
            self.model_name,
            self.epochs,
            self.patience,
        )

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

        best_val_mrr = -1.0
        epochs_no_improve = 0
        best_lora_state: dict[str, torch.Tensor] | None = None

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                optimizer.zero_grad()
                loss = self._compute_mnrl_loss(peft_model, batch["query"], batch["document"])
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)
            val_metrics_log: dict[str, float] = {"train_loss": avg_train_loss}

            if val_pairs:
                peft_model.eval()
                self._model = peft_model
                with torch.no_grad():
                    val_metrics = self._eval_validation(val_pairs, corpus_docs, corpus_dict, peft_model)
                val_metrics_log.update(val_metrics)
                peft_model.train()

                current_mrr = val_metrics["val_mrr_at_5"]
                if current_mrr > best_val_mrr + 1e-4:
                    best_val_mrr = current_mrr
                    epochs_no_improve = 0
                    best_lora_state = {
                        k: v.detach().cpu().clone() for k, v in peft_model.state_dict().items() if "lora_" in k
                    }
                else:
                    epochs_no_improve += 1

            if tracker:
                tracker.log_epoch_metrics(epoch=epoch, metrics=val_metrics_log)

            logger.info(
                "Эпоха %d/%d [%s]: train_loss=%.4f | val_loss=%.4f | val_MRR@5=%.4f (best=%.4f, no_improve=%d/%d)",
                epoch,
                self.epochs,
                self.model_name,
                avg_train_loss,
                val_metrics_log.get("val_loss", 0.0),
                val_metrics_log.get("val_mrr_at_5", 0.0),
                max(best_val_mrr, 0.0),
                epochs_no_improve,
                self.patience,
            )

            if val_pairs and epochs_no_improve >= self.patience:
                logger.info(
                    "Early Stopping на эпохе %d [%s]: val_MRR@5 не улучшался %d эпох (best=%.4f)",
                    epoch,
                    self.model_name,
                    self.patience,
                    best_val_mrr,
                )
                break

        if best_lora_state is not None:
            peft_model.load_state_dict(best_lora_state, strict=False)
            logger.info("Восстановлены лучшие веса LoRA адаптера (val_MRR@5=%.4f)", best_val_mrr)

        self._model = peft_model.merge_and_unload()

        # Сохранение чистых смердженных FP32 весов и токенизатора для MLflow Artifacts
        safe_name = self.model_name.replace("/", "_")
        save_dir = self.cache_dir / safe_name
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(save_dir))
        self._tokenizer.save_pretrained(str(save_dir))
        self._hf_save_dir = save_dir
        logger.info("Чистая дообученная FP32 модель сохранена: %s", self._hf_save_dir)

    def _eval_validation(
        self,
        val_pairs: list[dict[str, str]],
        corpus_docs: list[dict[str, Any]],
        corpus_dict: dict[str, str],
        peft_model: torch.nn.Module,
    ) -> dict[str, float]:
        """Compute validation MNRL loss and full-corpus ranking metrics."""
        # 1. Validation MNRL loss
        val_tuples = self._build_pair_tuples(val_pairs, corpus_dict)
        val_loader = DataLoader(PairDataset(val_tuples), batch_size=self.batch_size, shuffle=False)
        total_val_loss = 0.0
        val_batches = 0
        for batch in val_loader:
            loss = self._compute_mnrl_loss(peft_model, batch["query"], batch["document"])
            total_val_loss += float(loss.item())
            val_batches += 1
        avg_val_loss = total_val_loss / max(val_batches, 1)

        # 2. Full-corpus ranking metrics
        corpus_embs = self.encode_corpus(corpus_docs)
        corpus_hash_to_idx = {d["file_hash"]: idx for idx, d in enumerate(corpus_docs)}

        val_queries = [p["query"] for p in val_pairs]
        target_indices = [corpus_hash_to_idx[p["file_hash"]] for p in val_pairs]

        q_embs = self.encode_queries(val_queries)
        scores = self.predict_scores(q_embs, corpus_embs)
        val_metrics = calculate_ranking_metrics(scores, target_indices, top_k=5)

        return {
            "val_loss": avg_val_loss,
            "val_recall_at_1": val_metrics.recall_at_1,
            "val_recall_at_5": val_metrics.recall_at_5,
            "val_mrr_at_5": val_metrics.mrr_at_5,
        }
