from collections.abc import Callable

from meme_organizer.retrieval.adapters.dense_text import DenseTextRetrieverAdapter
from meme_organizer.retrieval.adapters.finetuned_dense import FineTunedDenseAdapter
from meme_organizer.retrieval.adapters.hybrid import HybridRetrieverAdapter
from meme_organizer.retrieval.base import BaseRetrieverAdapter
from meme_organizer.retrieval.baselines.bm25 import BM25RetrieverAdapter

ADAPTER_REGISTRY: dict[str, Callable[[], BaseRetrieverAdapter]] = {
    "bm25": BM25RetrieverAdapter,
    "rubert-tiny2": lambda: DenseTextRetrieverAdapter(
        model_name="rubert-tiny2",
        hf_model_id="cointegrated/rubert-tiny2",
    ),
    "multilingual-e5-small": lambda: DenseTextRetrieverAdapter(
        model_name="multilingual-e5-small",
        hf_model_id="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "rubert-tiny2-finetuned": lambda: FineTunedDenseAdapter(
        model_name="rubert-tiny2-finetuned",
        base_model_id="cointegrated/rubert-tiny2",
    ),
    "e5-small-finetuned": lambda: FineTunedDenseAdapter(
        model_name="multilingual-e5-small-finetuned",
        base_model_id="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "multilingual-e5-small-finetuned": lambda: FineTunedDenseAdapter(
        model_name="multilingual-e5-small-finetuned",
        base_model_id="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "hybrid-rrf": lambda: HybridRetrieverAdapter(mode="rrf"),
    "hybrid-optuna": lambda: HybridRetrieverAdapter(mode="optuna"),
    "hybrid-rerank-ce": lambda: HybridRetrieverAdapter(mode="rerank_ce", rerank_top_k=20),
}


def get_adapter(model_name: str) -> BaseRetrieverAdapter:
    """Factory creating adapter instances by name."""
    if model_name not in ADAPTER_REGISTRY:
        raise ValueError(f"Неизвестное имя модели: {model_name}")
    return ADAPTER_REGISTRY[model_name]()
