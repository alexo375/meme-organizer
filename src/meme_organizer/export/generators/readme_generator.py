"""Генератор документации в формате Markdown для интеграции мобильного клиента."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meme_organizer.domain.preset_manifest import PresetManifest

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent.parent / "resources" / "mobile_readme.md"


class ReadmeGenerator:
    """Генерирует технический README.md для интеграции на Android и Kotlin."""

    @classmethod
    def generate(cls, manifest: PresetManifest) -> str:
        """Формирует руководство по интеграции на Kotlin по заданному manifest и возвращает строку Markdown.

        Вход: объект PresetManifest.
        Выход: отрендеренная строка Markdown.
        """
        q_prefix = manifest.prefixes.get("query", "query: ")
        p_prefix = manifest.prefixes.get("passage", "passage: ")
        w_dense = manifest.hybrid_scoring.dense_weight
        w_bm25 = manifest.hybrid_scoring.bm25_weight
        sp = manifest.tokenizer.special_token_ids

        prov_section = ""
        if manifest.provenance:
            m = manifest.provenance.evaluated_onnx_test_metrics
            prov_section = (
                "\n## 6. Метаданные сборки и метрики\n"
                f"- Git Commit: `{manifest.provenance.git_commit}`\n"
                f"- ONNX SHA-256: `{manifest.provenance.onnx_sha256}`\n"
                f"- Паритет PyTorch FP32 / ONNX INT8: `mean_cosine = {manifest.provenance.pytorch_onnx_mean_cosine_similarity}`, "
                f"`max_abs_diff = {manifest.provenance.pytorch_onnx_max_abs_diff}`\n"
                f"- Метрики ONNX INT8 на тестовом сплите (`{manifest.provenance.splits_path}`):\n"
                f"  - Dense Only: `MRR@5 = {m.get('onnx_dense_mrr_at_5', 'n/a')}`, "
                f"`Recall@1 = {m.get('onnx_dense_recall_at_1', 'n/a')}`, "
                f"`Recall@5 = {m.get('onnx_dense_recall_at_5', 'n/a')}`\n"
                f"  - Hybrid (BM25 + Dense): `MRR@5 = {m.get('onnx_hybrid_mrr_at_5', 'n/a')}`, "
                f"`Recall@1 = {m.get('onnx_hybrid_recall_at_1', 'n/a')}`, "
                f"`Recall@5 = {m.get('onnx_hybrid_recall_at_5', 'n/a')}`\n"
            )

        template = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
        return template.substitute(
            model_name=manifest.model_name,
            max_sequence_length=manifest.max_sequence_length,
            embedding_dimension=manifest.embedding_dimension,
            schema_version=manifest.schema_version,
            q_prefix=q_prefix,
            p_prefix=p_prefix,
            sp_bos=sp.get("bos", 0),
            sp_pad=sp.get("pad", 1),
            sp_eos=sp.get("eos", 2),
            sp_unk=sp.get("unk", 3),
            w_dense=w_dense,
            w_bm25=w_bm25,
            prov_section=prov_section,
        )
