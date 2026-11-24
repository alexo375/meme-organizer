# Meme Organizer — Архитектурный манифест и план экспериментов

**Версия:** 1.0  
**Дата фиксации:** 2026-11-24  
**Статус исходных данных:** Gold Dataset готов (`data/gold/gold_dataset.jsonl`, 318 чистых верифицированных записей).  
**Целевой рантайм:** On-Device Android (ONNX Runtime Mobile, RAM < 150 MB, инференс < 50 мс на CPU).  
**Железо разработки:** Linux, CPU Intel Core i7-12700H (20 потоков, без дискретной NVIDIA GPU).

---

## 1. Ретроспектива и анализ граблей (Почему прошлый проект `bot-detector-maas` причинял боль)

1. **Enterprise Over-engineering и раздутый трекинг:**
   - В `bot-detector-maas` трекер (`ExperimentTracker`, `builder.py`, `lifecycle.py`, 10 файлов логгеров) содержал десятки жестких проверок (`require_graphics`, `require_feature_tree`, `require_submission`, `require_dataset`).
   - Если в конце 20-минутного тюнинга разработчик забывал вызвать какую-то мелкую функцию — всё падало с исключением, ран признавался невалидным, приходилось чистить базу и перезапускать заново.
   - *Решение для Meme Organizer:* **Инверсия контроля (Inversion of Control)** через паттерн **Template Method**. Эксперимент оборачивается в единый раннер `RetrievalPipeline.run()`, который **сам внутри себя** вызывает нужные шаги и автоматически логирует строго необходимый минимум.

2. **Data Hoarding (Логирование всего подряд):**
   - В MLflow складировались тонны тяжелых артефактов, деревья признаков, дампы датасетов. База пухла, UI тормозил, а реальной пользы для принятия инженерных решений это не несло.
   - *Решение:* Логировать **только 4 вещи**:
     1. Параметры модели/конфига (без лишних метаданных).
     2. Метрики ранжирования ($MRR@5$, $Recall@1$, $Recall@5$, $NDCG@5$).
     3. On-Device профиль (Размер ONNX в МБ, CPU Latency в мс).
     4. Таблица ошибок (Top-15 Worst Queries) в CSV для качественного анализа ошибок (Error Analysis).

3. **Тяжелое обучение на CPU (почему RuBERT учился 20 минут):**
   - В бот-детекторе гонялся Full Fine-Tuning всех весов BERT с длинным контекстом (`max_length=128+`) на тысячах сэмплов с 5 фолдами.
   - *Решение:*
     - Сначала **Zero-Shot Pretrained бейзлайны** (без обучения вообще).
     - Поисковые фразы мемов короткие (`max_length=32`).
     - Если потребуется дообучение — строго **замороженный backbone + LoRA / адаптер**, обучающийся за 2–3 минуты.

---

## 2. Архитектура кодовой базы (Clean Architecture & OOP)

Кодовая база строится на двух канонических паттернах: **Template Method** и **Adapter**.

### 2.1. Контракт адаптера модели (`BaseRetrieverAdapter`)
Любая модель (BM25, RuBERT, E5, SigLIP, Cross-Encoder) реализует единый интерфейс:
- `load_or_build()` — загрузка весов или инициализация индекса.
- `fit(train_pairs)` — опциональное дообучение (для LoRA / Sparse / Custom).
- `encode_corpus(corpus_docs)` — получение матриц эмбеддингов или инвертированного индекса для базы мемов.
- `encode_queries(queries)` — получение векторов или токенов для запросов.
- `predict_scores(query_repr, corpus_repr)` — расчет матрицы сходства (cosine / dot-product / BM25 scores).
- `export_onnx(output_path)` — экспорт в мобильный формат ONNX с INT8 квантованием.

### 2.2. Шаблонный пайплайн (`RetrievalPipeline`)
```
  Pipeline.run() (ЕДИНЫЙ ЖЕСТКИЙ ЦИКЛ)
  │
  ├── 1. adapter.load_or_build()
  ├── 2. [if train_mode] adapter.fit(train_dataset)
  ├── 3. corpus_embeddings = adapter.encode_corpus(corpus)
  ├── 4. query_embeddings = adapter.encode_queries(test_queries)
  ├── 5. scores = adapter.predict_scores(query_embeddings, corpus_embeddings)
  ├── 6. metrics, error_table = evaluator.compute(scores, ground_truth)
  ├── 7. profile = profiler.benchmark(adapter, sample_query)
  └── 8. tracker.auto_log(config, metrics, profile, error_table)
```
Разработчик пишет **только логику адаптера** (20–40 строк кода). Сломать трекинг или забыть посчитать метрику архитектурно невозможно.

---

## 3. Политика работы с данными и предотвращение утечек (Leakage Prevention)

1. **Разделение сущностей (Corpus vs Queries):**
   - **Corpus (318 мемов):** Полная база документов, по которой ведется поиск. Не сплитуется, так как поиск на девайсе всегда идет по полной локальной базе.
   - **Queries & Captions:** Сплитуются на `Train`, `Val` и `Test`.
2. **Scene-Group Awareness (GroupShuffleSplit):**
   - Связанные мемы (например, серия нарезок Властелина Колец: Арагорн, Леголас, Гимли; сцены с Тони Старком; серийные комиксы с пауком) группируются по `scene_group`.
   - Вся группа целиком попадает либо в Train, либо в Test. Это исключает утечку, когда модель «узнает» персонажа или стилистику по тестовому дублю.
3. **Фиксация сплита (`data/splits/splits_v1.json`):**
   - Сплит вычисляется один раз с фиксированным `seed=42`, хэшируется и коммитится.
   - Все модели на всех этапах тестируются на **абсолютно одинаковом Test Set**.
4. **OOF (Out-Of-Fold) для мета-моделей и Reranker:**
   - Если строится второй уровень (Cross-Encoder или Blending весов), негативные примеры и ранги первого уровня извлекаются строго через K-Fold кросс-валидацию, без заглядывания в таргет.

---

## 4. Панорама подходов по доменам (Roadmap)

### Домен 1: Текстовый семантический поиск (Text Retrieval)
*Фокус: поиск по тексту, смыслу, цитатам и сарказму.*
1. **Lexical Baseline:** BM25 (на базе чистой Python/SQLite реализации, аналогичной SQLite FTS5 на Android).
2. **Dense Zero-Shot Bi-Encoders:**
   - `cointegrated/rubert-tiny2` (29M параметров, ~15 МБ ONNX).
   - `intfloat/multilingual-e5-small` (117M параметров, ~40 МБ ONNX).
   - `BAAI/bge-m3` / `bge-small-en-v1.5` (сравнение плотности векторов).
3. **Contrastive Domain Fine-Tuning:**
   - Multiple Negatives Ranking Loss (MNRL) с Hard Negatives (майнинг топ-ошибок BM25).
   - Дообучение только LoRA-адаптера (ранг $r=8$, $\alpha=16$) для `rubert-tiny2` (время обучения < 3 минут на CPU).
4. **Knowledge Distillation (Учитель $\rightarrow$ Ученик):**
   - Извлечение эмбеддингов/скоров через сильную Teacher-модель (через API Gemini или локальную Qwen2.5-7B в 4-bit) и дистилляция в `rubert-tiny2`.

### Домен 2: Визуальный поиск (Visual / Multimodal Retrieval)
*Фокус: поиск по визуальным образам («кот в шапке», «мужик показывает палец»).*
1. **Keyframe Extraction Strategy для видео:**
   - Равномерный сэмплинг (1–3 кадра на ролик) против сэмплинга по гистограмме движения.
2. **Модели зрительного поиска:**
   - `Apple/MobileCLIP-S0` / `MobileCLIP-B` (архитектура, оптимизированная под мобильные NPU).
   - `google/siglip-base-patch16-256` (SOTA по качеству мультимодального выравнивания).
   - `ai-forever/ruclip-vit-base-patch32` (русскоязычный CLIP).

### Домен 3: Речь и цитаты (Audio/ASR Domain)
*Фокус: реакшн-видео с голосом и панчлайнами.*
1. Оценка необходимости On-Device инференса ASR против статического индекса.
2. Тестирование `sherpa-onnx` / `Whisper-tiny` на устойчивость к фоновой музыке и интернет-сленгу.

### Домен 4: Late Fusion & Re-ranking
*Фокус: объединение текстовых и визуальных сигналов в финальный топ-5.*
1. **Score Normalization & Fusion:**
   $$Score = \alpha \cdot Score_{text} + \beta \cdot Score_{visual} + \gamma \cdot BM25$$
   Оптимизация весов через Optuna на Validation-сплите.
2. **Cross-Encoder Reranker:**
   - Оценка выигрыша по качеству ($NDCG@5$) от прогона топ-20 кандидатов через легкий Cross-Encoder против цены по задержке (latency budget).

---

## 5. План действий на ближайший час (Старт новой ветки `master`)

1. **Создание чистой ветки `master`:**
   - Отделение от служебного кода доразметки.
   - Фиксация датасета `data/gold/gold_dataset.jsonl` как неизменяемого артефакта.
2. **Генерация честного сплита данных (`data/splits/splits_v1.json`):**
   - Группировка связанных мемов (`scene_group`).
   - Сохранение списков train/val/test query pairs.
3. **Разработка ядра трекинга и оценки (`src/meme_organizer/`):**
   - `core/contracts.py`: абстрактный класс `BaseRetrieverAdapter`.
   - `evaluation/metrics.py`: честный, векторизованный расчет $Recall@K$, $MRR$, $NDCG@K$.
   - `tracking/runner.py`: шаблонный `RetrievalPipeline` с авто-логированием в локальный MLflow (`sqlite:///mlflow.db`).
4. **Первый сравнительный запуск (Baseline 1: BM25 vs Baseline 2: RuBERT-tiny2):**
   - Реализация двух легковесных адаптеров.
   - Запуск через единый CLI: `python -m meme_organizer.experiments.run`.
   - Получение первой точки в лидерборде MLflow.
