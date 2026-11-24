# Meme Organizer — ML Search & Retrieval Pipeline

Репозиторий машинного обучения для мобильного приложения **Meme Organizer** (поиск мемов и реакций на устройстве Android).

Целевой рантайм: Android (ONNX Runtime Mobile, INT8 квантование, RAM < 150 MB, латентность < 50 мс на CPU).

Подробный архитектурный план и правила разработки описаны в [.agents/ROADMAP.md](.agents/ROADMAP.md) и [.agents/GEMINI.md](.agents/GEMINI.md).

---

## 🚀 Быстрый старт инфраструктуры (Docker + MLflow)

Для трекинга экспериментов используется **MLflow** с бэкендом **PostgreSQL** в Docker:

```bash
# Запуск PostgreSQL и MLflow Tracking Server
docker compose up -d

# Проверка статуса сервисов
docker compose ps
```
Веб-интерфейс MLflow доступен по адресу: **http://localhost:5000**.

Остановить сервисы:
```bash
docker compose down
```

---

## 🛠 Локальное окружение (uv)

Требуется Python 3.13 и `uv`:

```bash
uv sync
```

### Генерация честного сплита данных (Leak-free)
```bash
uv run python -m meme_organizer.data_splits.splitter
```
Создает фиксированный сплит `data/splits/splits_v1.json` без утечек связанных мемов (`scene_group`).

### Запуск экспериментов
```bash
# Запуск baseline модели (BM25)
uv run python -m meme_organizer.experiments.run --model bm25
```
Все метрики ($MRR@5$, $Recall@1$, $Recall@5$, $NDCG@5$), профили задержки и Top-15 Worst Queries автоматически логируются в MLflow.

---

## 🧪 Тестирование и качество кода

```bash
uv run poe test       # Запуск pytest
uv run poe lint       # Проверка ruff
uv run poe typecheck  # Проверка mypy
```
