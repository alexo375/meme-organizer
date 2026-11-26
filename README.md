# Meme Organizer ML Pipeline

Пайплайн обучения, оценки и экспорта моделей поиска мемов и видеореакций для Android-приложения Meme Organizer.

## Быстрый старт

Требуется Python 3.13, `uv` и Docker.

Поднять PostgreSQL и сервер MLflow (веб-интерфейс на `http://localhost:5000`):

```bash
docker compose up -d
```

Установить зависимости:

```bash
uv sync
```

Собрать мобильный бандл или опубликовать релиз можно через скрипты в `scripts/`:

```bash
uv run python scripts/build_mobile_bundle.py --help
uv run python scripts/publish_release.py --help
```

## Проверки

Запуск линтера, проверки типов и тестов:

```bash
just check-fast
```
