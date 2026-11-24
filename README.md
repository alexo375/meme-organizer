# Meme Organizer — ML & Retrieval Pipeline

On-device мультимодальный поиск и индексация мемов (фото, GIF, видео со звуком) для Android.

## Быстрый старт

1. Скопируй `.env.example` в `.env` и укажи `GEMINI_API_KEY`.
2. Установи зависимости: `uv sync --all-groups`.
3. Запусти smoke-тест разметки на 5 файлах: `uv run just label-smoke`.
4. Сгенерируй HTML-ревьюер: `uv run just review-silver`.
5. Запусти полную разметку (с авто-продолжением): `uv run just label-bronze`.
