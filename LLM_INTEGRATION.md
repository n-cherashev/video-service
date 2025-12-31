# LLM Integration для Video Service

## Обзор

Интегрирован Ollama LLM для многоязычного анализа юмора и тематической сегментации видео. Убраны языковые словари, добавлена поддержка structured output через JSON Schema.

## Архитектура

### LLM модуль (`llm/`)
- `base.py` - интерфейс LLMClient
- `ollama_client.py` - HTTP клиент для Ollama API
- `null_client.py` - fallback клиент (возвращает безопасные значения)
- `settings.py` - конфигурация Ollama
- `schemas.py` - JSON Schema для валидации ответов
- `prompts.py` - system/user промпты

### Обновленные хэндлеры
- `HumorDetectionHandler` - LLM-оценка юмора (0-1 score)
- `TopicSegmentationHandler` - LLM-генерация тем (slug format)

## Конфигурация

### Environment Variables
```bash
# LLM Settings
VIDEO_SERVICE_LLM_ENABLED=true
VIDEO_SERVICE_LLM_BASE_URL=http://localhost:11434
VIDEO_SERVICE_LLM_MODEL=llama3.2
VIDEO_SERVICE_LLM_TIMEOUT_SECONDS=30.0
VIDEO_SERVICE_LLM_RETRIES=2
VIDEO_SERVICE_LLM_TEMPERATURE=0.0
VIDEO_SERVICE_LLM_MAX_INPUT_CHARS=4000
```

### Dependency Injection
```python
# В main.py
llm_settings = OllamaSettings(...)
llm_client = OllamaLLMClient(llm_settings) if enabled else NullLLMClient()

handlers = [
    # ...
    HumorDetectionHandler(llm_client=llm_client),
    TopicSegmentationHandler(llm_client=llm_client),
    # ...
]
```

## API Ollama

### Request Format
```json
{
  "model": "llama3.2",
  "stream": false,
  "format": { /* JSON Schema */ },
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "options": {"temperature": 0}
}
```

### Response Format
```json
{
  "message": {
    "content": "{\"score\": 0.7, \"label\": \"clear\", ...}"
  }
}
```

## Промпты

### Humor Detection
- **Input**: текстовый сегмент + язык
- **Output**: `{score: 0-1, label: "none"|"light"|"likely"|"clear"|"strong", reason_codes: []}`
- **Особенности**: учитывает ASR шум, не зависит от языка

### Topic Segmentation  
- **Input**: список текстовых блоков + язык
- **Output**: `{topics: [{slug, title, confidence}, ...]}`
- **Особенности**: slug в snake_case, поддержка кириллицы

## Fallback Strategy

### При ошибках LLM:
- **Humor**: score = 0.0
- **Topics**: topic = "general"
- **Timeout/Network**: автоматические ретраи с backoff
- **Invalid JSON**: regex extraction + валидация

### При LLM_ENABLED=false:
- Используется `NullLLMClient`
- Пайплайн работает без LLM функций

## Тестирование

```bash
# Unit тесты
python -m pytest tests/test_llm_handlers.py -v

# Интеграционный тест
python -c "from main import build_handlers; ..."
```

### Покрытие тестами:
- ✅ Обработка пустых сегментов
- ✅ Fallback при ошибках LLM  
- ✅ Нормализация языков (Russian→ru, English→en)
- ✅ Нормализация slug (Hello World→hello_world)
- ✅ Коррекция длины массивов topics

## Производительность

### Оптимизации:
- **Кеширование**: SHA256 хеш от (тип, язык, текст)
- **Батчинг**: topics обрабатываются одним запросом
- **Лимиты**: max_input_chars, max_segments, max_blocks
- **Таймауты**: настраиваемые timeout + retries

### Мониторинг:
- Логирование времени запросов
- Логирование ошибок (без полного текста)
- Счетчики cache hit/miss

## Требования

### Обязательные:
- `httpx` для HTTP клиента
- Ollama server на `localhost:11434` (или настроенный URL)

### Опциональные:
- Модель `llama3.2` или совместимая
- GPU для ускорения (опционально)

## Примеры использования

### Запуск с LLM:
```bash
export VIDEO_SERVICE_LLM_ENABLED=true
python main.py video.mp4
```

### Запуск без LLM:
```bash  
export VIDEO_SERVICE_LLM_ENABLED=false
python main.py video.mp4
```

### Кастомная модель:
```bash
export VIDEO_SERVICE_LLM_MODEL=mistral
python main.py video.mp4
```