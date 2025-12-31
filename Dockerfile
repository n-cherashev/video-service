# Video Service Dockerfile
# Multi-stage build для оптимизации размера образа
#
# Build:
#   docker build -t video-service .
#
# Run:
#   docker run -p 8000:8000 video-service

# =============================================================================
# Stage 1: Builder - установка зависимостей
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Системные зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы зависимостей И исходный код
# (нужно для pip install -e .)
COPY pyproject.toml ./
COPY config/ ./config/
COPY core/ ./core/
COPY features/ ./features/
COPY models/ ./models/
COPY utils/ ./utils/
COPY io/ ./io/
COPY api/ ./api/
COPY ui/ ./ui/
COPY llm/ ./llm/
COPY ml/ ./ml/
COPY main.py ./

# Устанавливаем pip и зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

# =============================================================================
# Stage 2: Runtime - финальный образ
# =============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Системные зависимости для runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем установленные пакеты из builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем код приложения
COPY --from=builder /app /app

# Создаём директории
RUN mkdir -p /app/temp /app/results /app/artifacts /app/models

# Создаём непривилегированного пользователя
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Переменные окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV VIDEO_SERVICE_TEMP_DIR=/app/temp
ENV VIDEO_SERVICE_OUTPUT_DIR=/app/results
ENV USE_CELERY=true

# Healthcheck через Python (curl может быть недоступен в некоторых образах)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Порт
EXPOSE 8000

# По умолчанию запускаем FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
