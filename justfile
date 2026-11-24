set shell := ["bash", "-euo", "pipefail", "-c"]

# Показать список доступных сценариев
default:
    @just --list

# === Комплексные сценарии валидации ===

# Полный прогон всех проверок проекта: статика, безопасность, тесты с покрытием
[group('check')]
check: check-static check-security
    uv run poe test-cov

# Быстрый цикл валидации перед коммитом (вся статика + юнит-тесты)
[group('check')]
check-fast: check-static
    uv run poe test-unit

# Автоматическое исправление стиля кода и форматирования
[group('check')]
fix:
    uv run poe lint-fix
    uv run poe format

# Все статические проверки: линтинг, форматирование, типы, мертвый код, длины файлов, ООП-стиль
[group('check')]
check-static:
    uv run poe lint
    uv run poe format-check
    uv run poe typecheck
    uv run poe dead-code
    uv run poe check-lengths
    uv run poe oop-check

# Проверки безопасности: Bandit и pip-audit
[group('check')]
check-security:
    uv run poe security-audit
    uv run poe audit

# === Пайплайн разметки датасета (Bronze -> Silver) ===

# Безопасная конвертация замаскированных HEIF/WEBP в data/bronze -> честный .jpg
[group('data')]
normalize-bronze *args:
    uv run meme-label --log-level INFO normalize {{args}}

# Smoke-прогон авторазметки на 5 файлах
[group('data')]
label-smoke *args:
    uv run meme-label --log-level INFO label --max-files 5 {{args}}

# Полный прогон авторазметки data/bronze -> data/silver (с авто-продолжением)
[group('data')]
label-bronze *args:
    uv run meme-label --log-level WARNING label {{args}}

# Повторная попытка только для файлов из failed.jsonl
[group('data')]
label-retry-failed *args:
    uv run meme-label --log-level WARNING label --retry-failed {{args}}

# Генерация локального HTML-ревьюера для data/silver
[group('data')]
review-silver *args:
    uv run meme-label review {{args}}

# Интерактивная веб-среда доразметки и слияния дубликатов (FastAPI UI)
[group('data')]
curate *args:
    uv run meme-label curate {{args}}
