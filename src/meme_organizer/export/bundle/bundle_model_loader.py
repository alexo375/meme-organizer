"""Загрузчик или дообучатель базовой модели и токенизатора для мобильного бандла."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from transformers import AutoModel, AutoTokenizer

from meme_organizer.data.dataset_loader import DatasetLoader
from meme_organizer.retrieval.text.dense_text_retriever import DenseTextRetriever
from meme_organizer.retrieval.text.fine_tuned_dense_retriever import FineTunedDenseRetriever

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from meme_organizer.config import AppConfig
    from meme_organizer.export.bundle.bundle_profiles import ModelBundleProfile

logger = logging.getLogger(__name__)


class BundleModelLoader:
    """Загружает дообученную или базовую модель и токенизатор согласно профилю."""

    @staticmethod
    def load_or_train(
        profile: ModelBundleProfile,
        config: AppConfig,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Возвращает пару (модель, токенизатор) из MLflow, дообучения или HuggingFace.

        Вход: профиль бандла и конфигурация приложения.
        Выход: кортеж (PreTrainedModel, PreTrainedTokenizerBase).
        """
        if "finetuned" in profile.model_name:
            dense_loader = DenseTextRetriever(
                model_name=profile.model_name,
                hf_model_id=profile.base_hf_id,
                mlflow_run_name=f"{profile.model_name}_test",
            )
            dense_loader.load_or_build()
            if dense_loader._loaded_from_mlflow:
                logger.info("Используются чистые FP32 веса из MLflow для сборки бандла.")
                return dense_loader._model, dense_loader._tokenizer

            logger.info("Дообучение адаптера для модели %s (MLflow недоступен)...", profile.model_name)
            retriever = FineTunedDenseRetriever(
                model_name=profile.model_name,
                base_model_id=profile.base_hf_id,
                epochs=profile.epochs,
                patience=profile.patience,
                lr=profile.lr,
                batch_size=profile.batch_size,
                max_length=profile.max_length,
                lora_r=profile.lora_r,
                lora_alpha=profile.lora_alpha,
                query_prefix=profile.query_prefix,
                passage_prefix=profile.passage_prefix,
                seed=profile.seed,
            )
            train_pairs = DatasetLoader.load_query_pairs(config.splits_path, "train")
            val_pairs = DatasetLoader.load_query_pairs(config.splits_path, "val")
            corpus = DatasetLoader.load_corpus(config.gold_dataset_path)

            retriever.fit(train_pairs=train_pairs, corpus_docs=corpus, val_pairs=val_pairs)
            return retriever._model, retriever._tokenizer

        tokenizer = AutoTokenizer.from_pretrained(profile.base_hf_id)
        model = AutoModel.from_pretrained(profile.base_hf_id)
        return model, tokenizer
