# Обновление BaseHandler и логирования

## Изменения в BaseHandler

### Удалено:
```python
@property
def name(self) -> str:
    return self.__class__.__name__
```

### Причина:
- Упрощение базового класса
- Каждый хэндлер теперь сам отвечает за свой вывод

## Изменения в логировании

### Было (в main.py):
```python
for i, handler in enumerate(handlers, 1):
    print(f"[{i:2d}/{len(handlers)}] {handler.name}")
    context = handler.handle(context)
```

### Стало (в core/pipeline.py):
```python
def run_pipeline(context: dict[str, Any], handlers: Iterable[BaseHandler]) -> dict[str, Any]:
    for h in handlers:
        context = h.handle(context)
    return context
```

### В каждом хэндлере:
```python
def handle(self, context: dict[str, Any]) -> dict[str, Any]:
    print("[N] HandlerName")  # N - номер хэндлера
    
    # логика хэндлера...
    
    print("✓ Результат работы")
    return context
```

## Нумерация хэндлеров

1. ReadFileHandler
2. VideoMetaHandler  
3. FFmpegExtractHandler
4. MotionAnalysisFrameDiffHandler
5. AudioFeaturesHandler
6. AudioEventsHandler
7. SceneDetectionHandler
8. SpeechToTextHandler
9. SentimentAnalysisHandler
10. HumorDetectionHandler
11. TopicSegmentationHandler
12. FusionTimelineHandler
13. HighlightDetectionHandler
14. ChapterBuilderHandler
15. FinalizeAnalysisHandler

## Преимущества:

1. **Инкапсуляция**: Каждый хэндлер сам отвечает за свой вывод
2. **Простота**: Убрана лишняя абстракция в BaseHandler
3. **Читаемость**: Четкая нумерация и описание каждого этапа
4. **Гибкость**: Хэндлеры могут выводить дополнительную информацию по необходимости