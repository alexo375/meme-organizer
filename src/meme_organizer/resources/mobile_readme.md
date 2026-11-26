# Meme Organizer Text Engine (Mobile ONNX)

Пакет для локального текстового поиска в Android-приложении.

## 1. Состав пакета
- `text_encoder_int8.onnx` — модель `$model_name` в формате ONNX (Dynamic INT8).
  - Mean Pooling и L2-нормализация встроены в ONNX-граф.
  - Входы: `input_ids` `[batch, $max_sequence_length]` (`int64`), `attention_mask` `[batch, $max_sequence_length]` (`int64`).
  - Выход: `sentence_embedding` `[batch, $embedding_dimension]` (`float32`), L2-норма = `1.0`.
- `tokenizer/` — файлы токенизатора (`tokenizer.json`, `sentencepiece.bpe.model`, `tokenizer_config.json`, `special_tokens_map.json`).
- `preset_manifest.json` — параметры модели, токенизации, формулы скоринга и метрики сборки (`schema_version: $schema_version`).
- `smoke_test_pack.json` — эталонные входы и выходы для тестов паритета.

---

## 2. Токенизация
Модель использует токенизатор XLM-RoBERTa:
- Рекомендуется загружать `tokenizer/tokenizer.json` (через HuggingFace Tokenizers JNI или `onnxruntime-extensions`). При использовании `sentencepiece.bpe.model` напрямую необходимо учитывать сдвиг ID словаря XLM-R/Fairseq, иначе ID токенов не совпадут.
- Префикс для поискового запроса: `"$q_prefix"`
- Префикс для описания мема: `"$p_prefix"`
- Параметры: `padding = "max_length"`, `truncation = true`, `max_length = $max_sequence_length`.
- Спецтокены: `<s> = $sp_bos`, `<pad> = $sp_pad`, `</s> = $sp_eos`, `<unk> = $sp_unk`.

Эталонные массивы `input_ids` и `attention_mask` для проверки токенизатора лежат в `smoke_test_pack.json`.

---

## 3. Формула гибридного скоринга
Перед сложением оба канала (Dense cosine similarity и BM25) нормализуются через Min-Max по всем кандидатам для каждого запроса в диапазон `[0.0, 1.0]`:

`score = $w_dense * min_max_norm(cosine) + $w_bm25 * min_max_norm(bm25)`

где `min_max_norm(x_i) = (x_i - min(x)) / max(max(x) - min(x), 1e-9)`.

---

## 4. Пример интеграции на Kotlin

### 4.1. Зависимости (`build.gradle.kts`)
```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")
}
```

### 4.2. Инференс и гибридный скоринг
```kotlin
package com.memeorganizer.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.LongBuffer
import kotlin.math.max

class TextEmbeddingEngine(private val session: OrtSession, private val env: OrtEnvironment) {

    companion object {
        const val MAX_SEQ_LEN = $max_sequence_length
        const val EMBEDDING_DIM = $embedding_dimension
        const val QUERY_PREFIX = "$q_prefix"
        const val PASSAGE_PREFIX = "$p_prefix"
        const val WEIGHT_DENSE = ${w_dense}f
        const val WEIGHT_BM25 = ${w_bm25}f
    }

    fun encode(inputIds: LongArray, attentionMask: LongArray): FloatArray {
        require(inputIds.size == MAX_SEQ_LEN) { "inputIds must have size $$MAX_SEQ_LEN" }
        require(attentionMask.size == MAX_SEQ_LEN) { "attentionMask must have size $$MAX_SEQ_LEN" }

        val shape = longArrayOf(1, MAX_SEQ_LEN.toLong())
        val tIds = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape)
        val tMask = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape)

        val inputs = mapOf("input_ids" to tIds, "attention_mask" to tMask)
        session.run(inputs).use { result ->
            val tensor = result[0].value as Array<FloatArray>
            return tensor[0]
        }
    }

    fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        var dot = 0f
        for (i in 0 until EMBEDDING_DIM) dot += v1[i] * v2[i]
        return dot
    }

    fun minMaxNormalize(scores: FloatArray): FloatArray {
        if (scores.isEmpty()) return scores
        val minVal = scores.min()
        val maxVal = scores.max()
        val diff = max(maxVal - minVal, 1e-9f)
        return FloatArray(scores.size) { i -> (scores[i] - minVal) / diff }
    }

    fun combineHybridScores(denseScores: FloatArray, bm25Scores: FloatArray): FloatArray {
        val dNorm = minMaxNormalize(denseScores)
        val bNorm = minMaxNormalize(bm25Scores)
        return FloatArray(denseScores.size) { i ->
            WEIGHT_DENSE * dNorm[i] + WEIGHT_BM25 * bNorm[i]
        }
    }
}
```

### 4.3. Реализация BM25Okapi на Kotlin
Совпадает с Python `rank_bm25.BM25Okapi` и не зависит от системного SQLite FTS5:
```kotlin
package com.memeorganizer.ml

import kotlin.math.ln

class SimpleBm25Engine(
    documents: List<String>,
    private val k1: Double = 1.5,
    private val b: Double = 0.75,
    private val epsilon: Double = 0.25
) {
    private val tokenRegex = Regex("[^\\p{L}\\p{N}_]+")
    private val docTokens: List<List<String>> = documents.map { tokenize(it) }
    private val docLengths: IntArray = docTokens.map { it.size }.toIntArray()
    private val avgdl: Double = docLengths.average().takeIf { !it.isNaN() && it > 0 } ?: 1.0
    private val idf: Map<String, Double>

    init {
        val n = documents.size
        val df = mutableMapOf<String, Int>()
        for (tokens in docTokens) {
            for (term in tokens.toSet()) df[term] = (df[term] ?: 0) + 1
        }
        val rawIdf = mutableMapOf<String, Double>()
        val negTerms = mutableListOf<String>()
        var idfSum = 0.0
        for ((term, freq) in df) {
            val score = ln((n - freq + 0.5) / (freq + 0.5))
            rawIdf[term] = score
            idfSum += score
            if (score < 0) negTerms.add(term)
        }
        val avgIdf = if (df.isNotEmpty()) idfSum / df.size else 0.0
        val epsIdf = epsilon * avgIdf
        for (term in negTerms) rawIdf[term] = epsIdf
        idf = rawIdf
    }

    fun tokenize(text: String): List<String> =
        text.lowercase().split(tokenRegex).filter { it.length > 1 }

    fun scoreAll(query: String): FloatArray {
        val qTokens = tokenize(query)
        return FloatArray(docTokens.size) { docIdx ->
            val tokens = docTokens[docIdx]
            val dl = docLengths[docIdx]
            var score = 0.0
            for (q in qTokens) {
                val termIdf = idf[q] ?: continue
                val tf = tokens.count { it == q }.toDouble()
                score += termIdf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
            }
            score.toFloat()
        }
    }
}
```

---

## 5. Проверка паритета (`smoke_test_pack.json`)
- `samples`: `input_ids`, `attention_mask`, `bm25_tokens`, `embedding_head`, `full_embedding`.
- `similarity_tests`: ожидаемые значения косинусного сходства.
- `hybrid_ranking_tests`: эталонный расчёт `dense_raw_scores`, `dense_norm_scores`, `bm25_raw_scores`, `bm25_norm_scores`, `hybrid_scores` и `expected_ranking` по 4 документам.
$prov_section
