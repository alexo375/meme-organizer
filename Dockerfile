FROM python:3.13-slim

WORKDIR /app

# Установка системных зависимостей (ffmpeg, curl, build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libheif-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Копируем метаданные проекта и ставим зависимости
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Копируем исходный код
COPY src/ src/
COPY README.md ./
RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["uv", "run", "meme-label", "curate", "--host", "0.0.0.0", "--port", "8080"]
