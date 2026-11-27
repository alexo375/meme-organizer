# Meme Organizer Mobile ML Engine (ONNX)

Пакет для локального поиска мемов и видеореакций в Android-приложении. Поддерживает два профиля поставки:
- `v0.2.0-mobile` (`multilingual-e5-small-finetuned`): текстовый поиск (Dense + BM25).
- `v0.3.x-mobile` (`multimodal-v3`): мультимодальный поиск по 4 каналам (BM25 + Meta + OCR + ASR) с `gated_attention_fusion.onnx`.

## 1. Состав пакета

### Базовые файлы (`v0.2.0-mobile` и `v0.3.0-mobile`)
- `text_encoder_int8.onnx` — модель `$model_name` в формате ONNX (Dynamic INT8).
  - Mean Pooling и L2-нормализация встроены в ONNX-граф.
  - Входы: `input_ids` `[batch, $max_sequence_length]` (`int64`), `attention_mask` `[batch, $max_sequence_length]` (`int64`).
  - Выход: `sentence_embedding` `[batch, $embedding_dimension]` (`float32`), L2-норма = `1.0`.
- `tokenizer/` — файлы токенизатора (`tokenizer.json`, `tokenizer_config.json`).
- `preset_manifest.json` — параметры модели, токенизации, формулы скоринга, список `components` с SHA-256 и метрики сборки (`schema_version: $schema_version`).
- `smoke_test_pack.json` — эталонные входы и выходы для тестов паритета (текст + fusion).

### Дополнительные файлы (`v0.3.x-mobile`, `multimodal-v3`)
> Примечание: это OCR+ASR мультимодальность (без visual encoder и без визуального `tag_bank`).
- `ocr/` — `det.onnx` (детектор PP-OCR из `rapidocr_onnxruntime`), `inference.onnx` (распознавание `cyrillic_PP-OCRv5_mobile_rec_onnx`) + `dict.txt`.
- `asr/` — `GigaAM-CTC` (`model.int8.onnx` + `tokens.txt` для `sherpa-onnx` NeMo CTC, 16 kHz mono float32 PCM, чанкинг длинного аудио по 20–30 с на стороне клиента).
- `gated_attention_fusion.onnx` — Query-Conditioned Attention Gate (единый ONNX-файл, ~0.1 МБ).

---

## 2. Токенизация
Модель использует токенизатор XLM-RoBERTa:
- Рекомендуется загружать `tokenizer/tokenizer.json` (через HuggingFace Tokenizers JNI или `onnxruntime-extensions`). При использовании `sentencepiece.bpe.model` напрямую необходимо учитывать сдвиг ID словаря XLM-R/Fairseq, иначе ID токенов не совпадут.
- Префикс для поискового запроса: `"$q_prefix"`
- Префикс для текста документа (описание, OCR, ASR): `"$p_prefix"`
- Параметры: `padding = "max_length"`, `truncation = true`, `max_length = $max_sequence_length`.
- Спецтокены: `<s> = $sp_bos`, `<pad> = $sp_pad`, `</s> = $sp_eos`, `<unk> = $sp_unk`.

Эталонные массивы `input_ids` и `attention_mask` для проверки токенизатора лежат в `smoke_test_pack.json`.

---

## 3. Скоринг и индексация

Перед объединением все каналы нормализуются через Min-Max по всем кандидатам для каждого запроса в диапазон `[0.0, 1.0]`:

`min_max_norm(x_i) = (x_i - min(x)) / max(max(x) - min(x), 1e-9)`

### Текстовый профиль (`v0.2.0-mobile`)
`score = $w_dense * min_max_norm(cosine) + $w_bm25 * min_max_norm(bm25)`

### Мультимодальный профиль (`v0.3.x-mobile`, `multimodal-v3`)
**Индексация при импорте (в фоне через `WorkManager`):**
- Сохранить `ocr_text` (картинки: `raw`; видео/GIF: `uniform_3` кадров с дедупликацией строк) и `asr_text` (моно 16 kHz PCM из видео) в SQLite + `FTS5`.
- Закодировать через `text_encoder_int8.onnx` до 3 отдельных векторов `float[384]` на мем:
  - `vec_meta` = `e5(user_caption + tags)`
  - `vec_ocr` = `e5(ocr_text)` *(если не пустой)*
  - `vec_asr` = `e5(asr_text)` *(если не пустой)*

**Поиск и контракт `gated_attention_fusion.onnx`:**
1. Посчитать 4 канала скоров: `s0 = BM25/FTS5`, `s1 = cosine(q, vec_meta)`, `s2 = cosine(q, vec_ocr)`, `s3 = cosine(q, vec_asr)`.
2. Нормализовать каждый канал по текущему запросу в `[0, 1]` (`min-max`).
3. Сформировать `channel_mask` `[n_docs, 4]` (`bool`): столбцы 0 и 1 всегда `true`, столбец 2 = `ocr_text.isNotBlank()`, столбец 3 = `asr_text.isNotBlank()`.
4. Прогнать `gated_attention_fusion.onnx`:
   - Входы: `query_emb` `[n_queries, 384]` (`float32`), `channel_scores` `[n_queries, n_docs, 4]` (`float32`), `channel_mask` `[n_docs, 4]` (`bool`).
   - Выход: `fused_scores` `[n_queries, n_docs]` (`float32`).

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
    private val avgDocLen: Double = if (docLengths.isNotEmpty()) docLengths.average() else 0.0
    private val idf: Map<String, Double> = computeIdf(documents.size)

    fun tokenize(text: String): List<String> =
        tokenRegex.split(text.lowercase()).filter { it.length >= 2 }

    private fun computeIdf(corpusSize: Int): Map<String, Double> {
        val docFreq = HashMap<String, Int>()
        for (tokens in docTokens) {
            for (t in tokens.toSet()) docFreq[t] = (docFreq[t] ?: 0) + 1
        }
        val rawIdf = HashMap<String, Double>()
        val negativeWords = ArrayList<String>()
        var idfSum = 0.0
        for ((word, freq) in docFreq) {
            val score = ln((corpusSize - freq + 0.5) / (freq + 0.5))
            rawIdf[word] = score
            idfSum += score
            if (score < 0) negativeWords.add(word)
        }
        val averageIdf = if (rawIdf.isNotEmpty()) idfSum / rawIdf.size else 0.0
        val epsValue = epsilon * averageIdf
        for (word in negativeWords) rawIdf[word] = epsValue
        return rawIdf
    }

    fun scoreAll(query: String): FloatArray {
        val qTokens = tokenize(query)
        val scores = FloatArray(docTokens.size)
        for (i in docTokens.indices) {
            val dTokens = docTokens[i]
            val dLen = docLengths[i]
            val tf = HashMap<String, Int>()
            for (t in dTokens) tf[t] = (tf[t] ?: 0) + 1
            var sum = 0.0
            for (qt in qTokens) {
                val freq = tf[qt] ?: continue
                val wordIdf = idf[qt] ?: continue
                val num = freq * (k1 + 1.0)
                val den = freq + k1 * (1.0 - b + b * dLen / avgDocLen)
                sum += wordIdf * (num / den)
            }
            scores[i] = sum.toFloat()
        }
        return scores
    }
}
```

---

## 5. Проверка интеграции (`smoke_test_pack.json`)
- `tokenization_tests`: эталонные `input_ids` и `attention_mask` для нескольких запросов.
- `embedding_tests`: ожидаемый срез первых 8 компонент `embedding_head` и полный `full_embedding` (384D).
- `similarity_tests`: ожидаемые значения косинусного сходства.
- `hybrid_ranking_tests`: эталонный расчёт `dense_raw_scores`, `dense_norm_scores`, `bm25_raw_scores`, `bm25_norm_scores`, `hybrid_scores` и `expected_ranking` по 4 документам.
$prov_section
