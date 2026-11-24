"""Markdown documentation generator for mobile client integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meme_organizer.mobile.manifest import PresetManifest


class ReadmeGenerator:
    """Generates complete and unambiguous README.md for Android / Kotlin developer."""

    @classmethod
    def generate(cls, manifest: PresetManifest) -> str:
        """Render complete integration guide with Kotlin snippets."""
        return f"""# Meme Organizer Mobile Text Engine — Integration Guide

Пакет предназначен для локального (on-device) семантического и гибридного поиска мемов в Android-приложении.

## 1. Содержимое пакета
- **`text_encoder_int8.onnx`** — квантованная INT8 модель ({manifest.model_name}).
  - **ВНИМАНИЕ:** Внутрь ONNX-графа **уже вшиты Mean Pooling и L2-нормализация**!
  - **Вход:** `input_ids` [1, {manifest.max_sequence_length}], `attention_mask` [1, {manifest.max_sequence_length}]
  - **Выход:** `sentence_embedding` [1, {manifest.embedding_dimension}] с нормой $\\|v\\|_2 = 1.0$.
  - Android-клиенту **НЕ НУЖНО** производить матричные умножения, пулинги или нормализацию векторов — модель возвращает готовый единичный вектор эмбеддинга.
- **`tokenizer/`** — файлы токенизатора:
  - `sentencepiece.bpe.model` (для C++ / JNI SentencePiece)
  - `tokenizer.json`, `special_tokens_map.json` (для HuggingFace / OnnxRuntime-extensions)
- **`preset_manifest.json`** — декларативный контракт метаданных (размерность, формулы, префиксы).
- **`smoke_test_pack.json`** — эталонный набор для проверки паритета (входные токены, ожидаемые векторы и косинусные скоры для JUnit тестов).

---

## 2. Подключение в Android (`build.gradle.kts`)

```kotlin
dependencies {{
    // ONNX Runtime Mobile
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")

    // (Опционально) Если используется встроенная токенизация ONNX:
    // implementation("com.microsoft.onnxruntime:onnxruntime-extensions-android:0.10.0")
}}
```

Поместите `text_encoder_int8.onnx` в `app/src/main/assets/models/` или скачивайте динамически в `context.filesDir`.

---

## 3. Архитектура поиска на девайсе (Гибрид BM25 + Dense)

Модель оптимизирована для двухэтапного или комбинированного скоринга:

$$\\text{{Score}} = {manifest.hybrid_scoring.alpha_bm25} \\times \\text{{Score}}_{{\\text{{BM25}}}} + {manifest.hybrid_scoring.beta_dense} \\times \\text{{CosineSimilarity}}$$

1. **BM25 (Лексический поиск):** Реализуется через встроенный в Android **SQLite FTS5** (по таблице описаний и транскриптов мемов).
2. **Dense (Семантический поиск):**
   - Текст запроса дополняется префиксом `{manifest.prefixes.get("query", "query: ")}`.
   - Текст описания мема дополняется префиксом `{manifest.prefixes.get("passage", "passage: ")}`.
   - Вычисляется скалярное произведение (для L2-нормализованных векторов `dotProduct(q, doc)` строго равно `cosineSimilarity`).

---

## 4. Пример реализации на Kotlin

```kotlin
package com.memeorganizer.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer

class TextEmbeddingEngine(private val session: OrtSession, private val env: OrtEnvironment) {{

    companion object {{
        const val MAX_SEQ_LEN = {manifest.max_sequence_length}
        const val EMBEDDING_DIM = {manifest.embedding_dimension}
        const val QUERY_PREFIX = "{manifest.prefixes.get("query", "query: ")}"
        const val PASSAGE_PREFIX = "{manifest.prefixes.get("passage", "passage: ")}"
    }}

    /**
     * Получение L2-нормализованного вектора эмбеддинга (размер 384).
     * @param inputIds массив токенов, выровненный до MAX_SEQ_LEN
     * @param attentionMask маска внимания (1 для реальных токенов, 0 для паддингов)
     */
    fun encode(inputIds: LongArray, attentionMask: LongArray): FloatArray {{
        require(inputIds.size == MAX_SEQ_LEN) {{ "inputIds must have size $MAX_SEQ_LEN" }}
        require(attentionMask.size == MAX_SEQ_LEN) {{ "attentionMask must have size $MAX_SEQ_LEN" }}

        val shape = longArrayOf(1, MAX_SEQ_LEN.toLong())
        val tensorInputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape)
        val tensorAttentionMask = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape)

        val inputs = mapOf(
            "input_ids" to tensorInputIds,
            "attention_mask" to tensorAttentionMask
        )

        session.run(inputs).use {{ result ->
            val embeddingTensor = result[0].value as Array<FloatArray>
            // Модель уже выполнила Mean-Pooling и L2-нормализацию!
            return embeddingTensor[0]
        }}
    }}

    /**
     * Косинусное сходство между двумя нормализованными векторами (скалярное произведение).
     */
    fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {{
        var dot = 0f
        for (i in 0 until EMBEDDING_DIM) {{
            dot += v1[i] * v2[i]
        }}
        return dot
    }}

    /**
     * Итоговый гибридный скор мема.
     */
    fun computeHybridScore(bm25NormalizedScore: Float, cosineSimilarity: Float): Float {{
        return {manifest.hybrid_scoring.alpha_bm25}f * bm25NormalizedScore + {manifest.hybrid_scoring.beta_dense}f * cosineSimilarity
    }}
}}
```

---

## 5. Проверка паритета (Smoke Test Verification)

В файле `smoke_test_pack.json` находятся эталонные сэмплы из Python-пайплайна.

Для проверки паритета на Android напишите простой JUnit-тест:
1. Загрузите сэмпл `query_1` из `smoke_test_pack.json`.
2. Подайте `input_ids` и `attention_mask` в `TextEmbeddingEngine.encode()`.
3. Сравните первые 10 значений (`embedding_head`) и норму вектора:
   ```kotlin
   val embedding = engine.encode(sample.inputIds, sample.attentionMask)

   // Проверка L2-нормы (должна быть 1.0)
   var norm = 0f
   for (v in embedding) norm += v * v
   assertEquals(1.0f, Math.sqrt(norm.toDouble()).toFloat(), 1e-4f)

   // Сравнение значений с эталоном
   for (i in 0 until 10) {{
       assertEquals(sample.embeddingHead[i], embedding[i], 1e-3f)
   }}
   ```
"""
