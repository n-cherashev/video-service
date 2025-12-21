# ТЗ: Анализ движения в видео и реализация трёх методов детекции

## 1. Контекст проекта и назначение

### 1.1. Контекст

Проект представляет собой **консольный пайплайн обработки видео**, построенный на архитектуре **handlers**:

- Каждый этап обработки видео реализован отдельным классом-обработчиком (handler)
- Handlers принимают и модифицируют общий `context: dict[str, Any]`
- Handlers вызываются последовательно, формируя конвейер обработки

Уже реализованные этапы (по текущему репозиторию):

1. **ReadFileHandler**
   - Валидирует входной путь к файлу
   - Нормализует и сохраняет путь к видео в контексте
   - Пишет:
     - `context["video_path"]: str`
     - `context["video_size_bytes"]: int`

2. **FFmpegExtractHandler**
   - Извлекает аудио из входного видео через ffmpeg
   - Сохраняет путь к аудиофайлу
   - Пишет:
     - `context["audio_path"]: str`

Дальше в пайплайне нужен этап **анализа движения в видео** — построение временного ряда активности движения по ролику.

### 1.2. Цель нового функционала

Цель — реализовать **анализ движения в видеопотоке (motion analysis)**, который:

1. Для каждого интервала времени (обычно 1 секунда) оценивает **интенсивность движения**.
2. Возвращает **motion heatmap** — список точек вида:
   ```python
   {
       "time": float,   # время в секундах (целое или дробное)
       "value": float,  # нормализованная интенсивность движения, 0.0–1.0
   }
   ```
3. Считает агрегированную статистику по ролику:
   - Средняя интенсивность движения
   - Максимальная интенсивность
   - Количество секунд, для которых есть данные
4. Обогащает `context` этими данными, чтобы следующие handlers могли:
   - Выбирать динамичные фрагменты
   - Определять «тихие» моменты
   - Использовать информацию для нарезки, суммаризации, captioning и т.д.

### 1.3. Место в пайплайне

Порядок запуска в `main.py` (логический):

1. `ReadFileHandler` — проверка и подготовка входного файла
2. `FFmpegExtractHandler` — извлечение звука
3. **Motion Analysis Handler** (новый этап) — анализ движения
   - Основной используемый метод: **Frame Differencing**
   - Дополнительно реализуются ещё два метода (Background Subtraction, Optical Flow) для экспериментов и сравнения

---

## 2. Анализ задачи: что значит «анализ движения»

### 2.1. Что считаем движением

Под «движением» в контексте этого проекта понимается **изменение визуального содержимого между соседними кадрами**:

- Если картинка между кадрами почти не меняется → низкая активность
- Если много пикселей меняют яркость / положение объектов → высокая активность

Мы не пытаемся:
- Детектировать конкретные объекты (людей, машины)
- Строить треки объектов во времени
- Делать семантическую сегментацию

Задача более простая и агрегированная: **одна числовая оценка «насколько кадр динамичный»** и агрегирование по секундам.

### 2.2. Выходные артефакты анализа

В результате работы handler’а в `context` должны появиться:

1. **Метаданные видео**:
   - `fps: float` — частота кадров
   - `duration_seconds: float | None` — длительность видео в секундах (если удалось определить)

2. **Motion heatmap**:
   ```python
   context["motion_heatmap"]: list[dict[str, float]] = [
       {"time": 0.0, "value": 0.12},
       {"time": 1.0, "value": 0.34},
       ...
   ]
   ```
   - `time` — секунда таймлайна (обычно целое число в секундах)
   - `value` — нормализованная интенсивность движения, 0.0–1.0

3. **Сводная статистика**:
   ```python
   context["motion_summary"]: dict[str, float | int] = {
       "mean": 0.23,       # среднее значение value по роликам
       "max": 1.0,         # максимальное значение value
       "seconds": 121,     # количество секунд с рассчитанным значением
   }
   ```

4. **Служебная информация**:
   - `context["detection_method"]: str` — идентификатор использованного метода (`"frame_diff"`, `"background_subtraction_mog2"`, `"optical_flow_farneback"` и т.п.)
   - `context["processing_time_seconds"]: float` — время работы handler’а в секундах

### 2.3. Требования к точности и скорости

1. **Точность**:
   - Мы не стремимся к идеальной pixel-perfect оценке, но хотим устойчивую метрику:
     - Высокие пики на участках с большим движением
     - Низкие значения на статичных участках
   - Относительная шкала важнее абсолютных значений: нам важна форма кривой по времени

2. **Скорость**:
   - Пайплайн должен быть работоспособен на **длинных видео** (5–30 минут)
   - На типичном железе (CPU без GPU) время обработки должно быть разумным
   - Поэтому:
     - Основной метод в пайплайне — самый лёгкий (**Frame Differencing**)
     - Более тяжёлые методы (Background Subtraction, Optical Flow) реализуются отдельно для анализа/сравнения

3. **Надёжность**:
   - Обработка не должна падать на видео с плохо определённым FPS или кадрами
   - Все handler’ы обязаны корректно освобождать ресурсы (VideoCapture)

---

## 3. Обзор методов детекции движения

### 3.1. Frame Differencing (разница кадров)

**Идея**: измеряем разницу яркости между кадрами t и t-1. Если много пикселей изменились существенно → есть движение.

**Формула**:

```text
I_t   = текущий кадр в grayscale
I_t-1 = предыдущий кадр в grayscale

D_t   = |I_t - I_t-1|             # абсолютная разница
M_t   = D_t > threshold           # бинарная маска движения
motion_value_t = (# белых пикселей) / (общее число пикселей)
```

**Плюсы**:
- Очень быстрый метод (O(N пикселей/кадр))
- Простая и понятная реализация
- Не требует сложной инициализации

**Минусы**:
- Чувствителен к шуму и небольшим изменениям освещения
- Слабо работает при медленном плавном движении, если threshold слишком высок

**Вывод**: идеален как **основной метод** для пайплайна (баланс скорости и качества).

---

### 3.2. Background Subtraction (вычитание фона)

**Идея**: модель фона строится как статистическое описание «типичного» кадра. Всё, что сильно отличается от фона, считается движением (foreground).

В OpenCV доступны два популярных алгоритма:

1. `cv2.createBackgroundSubtractorMOG2()`
   - Модель фона основана на **смеси гауссианов (Gaussian Mixture Model)** для каждого пикселя.
   - Учитывает постепенные изменения в сцене (изменение освещения, шевеление листвы и т.п.).

2. `cv2.createBackgroundSubtractorKNN()`
   - Модель фона основана на **k-ближайших соседях** в пространстве состояний пикселя.
   - Хорошо справляется с резкими изменениями, обычно быстрее, менее точен для сложного шума.

**Плюсы**:
- Лучшая фильтрация шума, чем простая разница кадров
- Умеет отделять фон от переднего плана
- Подходит для стационарной камеры (камеры наблюдения)

**Минусы**:
- Требует **warm-up** периода (несколько десятков/сотен кадров) для обучения модели
- Медленнее, чем Frame Differencing
- Потребляет больше памяти

**Вывод**: хороший **экспериментальный/альтернативный** метод для использования в специфичных сценариях, но не как основной в пайплайне.

---

### 3.3. Optical Flow (оптический поток)

**Идея**: для каждого пикселя оценивается вектор движения (dx, dy) между кадрами. Мы получаем **поле скоростей** — куда и насколько сдвинулся каждый пиксель.

В OpenCV классический алгоритм — **Farneback Dense Optical Flow** (`cv2.calcOpticalFlowFarneback`) [docs OpenCV].

Результат — массив `flow` размером (H, W, 2):
- `flow[..., 0]` — смещение по X (горизонталь)
- `flow[..., 1]` — смещение по Y (вертикаль)

Интенсивность движения можно оценивать как **магнитуду вектора скорости**:

```text
magnitude = sqrt(dx^2 + dy^2)
```

**Плюсы**:
- Самый информативный метод
- Позволяет оценить не только факт движения, но и направление/скорость
- Особенно полезен для сложного движения (спорт, трафик)

**Минусы**:
- Вычислительно тяжёлый (существенно медленнее двух других методов)
- Чувствителен к параметрам пирамиды, размера окна, итерациям

**Вывод**: идеально как **исследовательский метод** и для высокоточной аналитики, но слишком тяжёлый, чтобы быть основным этапом пайплайна.

---

## 4. Требования к архитектуре реализации

### 4.1. Общие требования к handler’ам

1. **Интерфейс**:
   - Все handlers должны наследоваться от общего `BaseHandler`
   - Все должны реализовывать метод:

     ```python
     def handle(self, context: dict[str, Any]) -> dict[str, Any]:
         ...
     ```

2. **Входные данные**:
   - Обязательные:
     - `context["video_path"]: str`
   - Опциональные (handler может использовать для логов):
     - `context["video_size_bytes"]: int | None`

3. **Выходные данные** (для всех трёх методов единый контракт):
   ```python
   context["fps"]: float
   context["duration_seconds"]: float | None
   context["motion_heatmap"]: list[dict[str, float]]
   context["motion_summary"]: dict[str, float | int]
   context["detection_method"]: str
   context["processing_time_seconds"]: float
   ```

4. **Обработка ошибок**:
   - Если `video_path` отсутствует — `ValueError("'video_path' not provided in context")`
   - Если видео невозможно открыть (`VideoCapture.isOpened() == False`) — `RuntimeError("Cannot open video: ...")`
   - Если недостаточно кадров для анализа (например, только 1 кадр) — `RuntimeError("Not enough frames to analyze motion")`

5. **Ресурсы**:
   - Использовать `cv2.VideoCapture(video_path)`
   - Всегда вызывать `cap.release()` в `finally`-блоке

6. **Производительность**:
   - Применять downscale кадров и шаг по кадрам там, где разумно (особенно в Frame Differencing и Background Subtraction)
   - В Optical Flow — осторожно с параметрами (levels, win_size, num_iters) из-за высокой стоимости

---

## 5. Библиотеки и зависимости

### 5.1. Внешние библиотеки

Общие для всех трёх реализаций:

- **OpenCV**:
  - Пакет: `opencv-python`
  - Основные используемые компоненты:
    - `cv2.VideoCapture` — чтение видео
    - `cv2.cvtColor` — конвертация в grayscale
    - `cv2.resize` — изменение размера
    - `cv2.GaussianBlur` — размытие
    - `cv2.absdiff` — разность кадров
    - `cv2.threshold` — пороговая фильтрация
    - `cv2.dilate` / `cv2.morphologyEx` — морфологические операции
    - `cv2.createBackgroundSubtractorMOG2`, `cv2.createBackgroundSubtractorKNN` — модели фона
    - `cv2.calcOpticalFlowFarneback` — оптический поток

- **NumPy**:
  - Для операций над матрицами и статистики:
    - `np.isfinite`, `np.mean`, `np.max`, операции над массивами

### 5.2. Обновления `pyproject.toml`

```toml
[project]
dependencies = [
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    # ... другие зависимости проекта
]
```

---

## 6. Дизайн трёх handler’ов

Реализовать **три отдельных handler’а**, каждый в своём файле:

1. `handlers/motion_analysis_frame_diff_handler.py`
2. `handlers/motion_analysis_background_sub_handler.py`
3. `handlers/motion_analysis_optical_flow_handler.py`

### Общая схема работы каждого handler’а

1. Валидация `context` и получение `video_path`
2. Инициализация `cv2.VideoCapture`
3. Получение `fps`, `frame_count`, подсчёт `duration_seconds`
4. Главный цикл по кадрам
5. Расчёт пометочного значения `motion_value` для каждого обрабатываемого кадра
6. Агрегация по секундам (сумма + счётчик → среднее)
7. Нормализация в диапазон `[0.0, 1.0]`
8. Заполнение `motion_heatmap`, `motion_summary` и служебных полей
9. Логирование
10. Освобождение ресурсов

---

## 7. Handler 1: FrameDifferencing — MotionAnalysisFrameDiffHandler

### 7.1. Файл и сигнатура

- Файл: `handlers/motion_analysis_frame_diff_handler.py`
- Класс: `MotionAnalysisFrameDiffHandler(BaseHandler)`

### 7.2. Конструктор и параметры

```python
class MotionAnalysisFrameDiffHandler(BaseHandler):
    def __init__(
        self,
        resize_width: int = 320,
        frame_step: int = 2,
        diff_threshold: int = 25,
        blur_kernel: tuple[int, int] = (5, 5),
        dilate_iterations: int = 1,
    ) -> None:
        ...
```

**Назначение параметров**:

- `resize_width: int = 320`
  - Если > 0 — кадры уменьшаются до этой ширины с сохранением пропорций
  - Уменьшает количество пикселей → ускоряет расчёты
  - Если 0 — использовать исходное разрешение

- `frame_step: int = 2`
  - Обрабатывать каждый N-й кадр (frame_index % frame_step == 0)
  - Экономия времени: `frame_step=2` → ~2x ускорение
  - При `frame_step=1` обрабатываются все кадры

- `diff_threshold: int = 25`
  - Порог бинаризации для разницы кадров (0–255)
  - Значения ниже threshold считаются шумом (0), выше — движением (255)

- `blur_kernel: tuple[int, int] = (5, 5)`
  - Размер ядра для `cv2.GaussianBlur`
  - Должен быть нечётным
  - Чем больше, тем сильнее подавление шума (но больше размывание деталей)

- `dilate_iterations: int = 1`
  - Количество итераций `cv2.dilate` для бинарной маски
  - Склеивает близко расположенные области движения и гасит мелкий шум

### 7.3. Логика `handle()` (по шагам)

1. **Валидация контекста**
   - Проверить `"video_path" in context` и что это непустая строка
   - Иначе → `ValueError("'video_path' not provided in context")`

2. **Открытие видео**
   - `cap = cv2.VideoCapture(video_path)`
   - Если `not cap.isOpened()` → `RuntimeError("Cannot open video: {video_path}")`

3. **Получение метаданных**
   - `fps = cap.get(cv2.CAP_PROP_FPS)`
     - Если `fps <= 0` или `not np.isfinite(fps)` → использовать `fps = 25.0`
   - `frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)`
     - Если `frame_count > 0` и `np.isfinite(frame_count)` → `duration_seconds = frame_count / fps`
     - Иначе `duration_seconds = None`

4. **Инициализация аккумуляторов**
   - `sum_by_sec: dict[int, float]` — сумма motion-значений по секундам
   - `cnt_by_sec: dict[int, int]` — количество измерений на секунду
   - `prev_gray: np.ndarray | None = None`
   - `frame_index: int = -1`
   - `processed_frames: int = 0`
   - Замер времени: `start_time = time.time()`

5. **Цикл чтения кадров**
   - `while True:`
     - `ok, frame = cap.read()`
     - Если `not ok` → выход из цикла
     - `frame_index += 1`
     - Fps-step фильтр:
       - Если `frame_index % frame_step != 0` → `continue`
     - Конвертация в grayscale:
       - `gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`
     - Downscale (если `resize_width > 0`):
       - Вычислить новые размеры с сохранением пропорций
       - `gray = cv2.resize(gray, (new_w, new_h), cv2.INTER_AREA)`
     - Gaussian blur (если `blur_kernel > (1,1)`):
       - `gray = cv2.GaussianBlur(gray, blur_kernel, 0)`

     - Если `prev_gray is not None` → вычислить разницу:
       - `diff = cv2.absdiff(prev_gray, gray)`
       - Бинаризация:
         - `_, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)`
       - Морфология (если `dilate_iterations > 0`):
         - `thresh = cv2.dilate(thresh, None, iterations=dilate_iterations)`
       - Подсчёт движущихся пикселей:
         - `changed = cv2.countNonZero(thresh)`
         - `total = thresh.shape[0] * thresh.shape[1]`
         - `motion_value = changed / total if total > 0 else 0.0`
       - Привязка ко времени:
         - `t = frame_index / fps`
         - `second = int(t)`
       - Агрегация:
         - `sum_by_sec[second] += motion_value`
         - `cnt_by_sec[second] += 1`

     - Обновление `prev_gray = gray`
     - `processed_frames += 1`

6. **Проверка количества кадров**
   - Если `processed_frames < 2` → `RuntimeError("Not enough frames to analyze motion")`

7. **Построение heatmap**
   - `secs = sorted(sum_by_sec.keys())`
   - Для каждой секунды считать среднее:
     - `avg = sum_by_sec[sec] / cnt_by_sec[sec]`
   - Собрать массив `avg_values`
   - Найти максимум `max_avg`
   - Нормализовать: `value = avg / max_avg`, если `max_avg > 0`, иначе `0.0`
   - Сформировать `motion_heatmap = [{"time": float(sec), "value": float(norm)} ...]`

8. **Статистика**
   - `values = [p["value"] for p in motion_heatmap]`
   - `mean_val = float(np.mean(values)) if values else 0.0`
   - `max_val = float(np.max(values)) if values else 0.0`

9. **Заполнение контекста**
   - `context["fps"] = fps`
   - `context["duration_seconds"] = duration_seconds`
   - `context["motion_heatmap"] = motion_heatmap`
   - `context["motion_summary"] = {"mean": mean_val, "max": max_val, "seconds": len(motion_heatmap)}`
   - `context["detection_method"] = "frame_diff"`
   - `context["processing_time_seconds"] = time.time() - start_time`

10. **Логирование**
    - Вывести строку вида:
      ```text
      ✓ Motion analysis (Frame Diff): seconds=XXX mean=0.123 max=1.000 time=7.4s
      ```

11. **Освобождение ресурсов**
    - В блоке `finally`: `cap.release()`

---

## 8. Handler 2: Background Subtraction — MotionAnalysisBackgroundSubHandler

### 8.1. Файл и сигнатура

- Файл: `handlers/motion_analysis_background_sub_handler.py`
- Класс: `MotionAnalysisBackgroundSubHandler(BaseHandler)`

### 8.2. Конструктор и параметры

```python
class MotionAnalysisBackgroundSubHandler(BaseHandler):
    def __init__(
        self,
        method: str = "mog2",      # "mog2" или "knn"
        resize_width: int = 320,
        frame_step: int = 1,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = -1.0,  # -1 = авто
        background_ratio: float = 0.7, # для MOG2
    ) -> None:
        ...
```

### 8.3. Особенности параметров

- `method`:
  - `"mog2"`: `cv2.createBackgroundSubtractorMOG2`
  - `"knn"`: `cv2.createBackgroundSubtractorKNN`

- `history`: сколько кадров использовать для построения модели фона
- `var_threshold`: чувствительность к отличиям от фона
- `detect_shadows`: включать ли детекцию теней
- `learning_rate`:
  - `-1.0` — автоадаптация (рекомендуется)
  - >0 — вручную задавать скорость обучения
- `background_ratio` (только для MOG2): доля смеси гауссианов, считающихся фоном

### 8.4. Логика `handle()` (по шагам)

Общие шаги (валидация, открытие видео, чтение fps/frame_count, инициализация аккумуляторов) аналогичны Frame Diff handler’у.

Отличия:

1. **Инициализация background subtractor’а**

```python
if method == "mog2":
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )
    # background_ratio можно установить через bg_sub.setBackgroundRatio(background_ratio)

elif method == "knn":
    bg_sub = cv2.createBackgroundSubtractorKNN(
        history=history,
        detectShadows=detect_shadows,
    )
```

2. **Warm-up период (обучение фона)**

- В начале рекомендуется прогнать первые N кадров (например, 120–200) только для обучения модели, без накопления метрики.
- Логика:

```python
warmup_frames = min(200, int(history / 2))

for i in range(warmup_frames):
    ok, frame = cap.read()
    if not ok:
        break
    if resize_width > 0:
        # downscale
    fg_mask = bg_sub.apply(frame, learningRate=learning_rate)
```

- После этого сбросить `frame_index` и начать основной цикл анализа.

3. **Главный цикл анализа**

В цикле чтения кадров:

- Применить downscale (как в Frame Diff)
- Применить background subtractor:

```python
fg_mask = bg_sub.apply(frame, learningRate=learning_rate)
```

- Если включены тени (MOG2):
  - В бинарной маске `fg_mask` тени обычно имеют значение 127, foreground — 255
  - Можно исключить тени из метрики:

```python
if detect_shadows:
    fg_mask[fg_mask == 127] = 0
```

- Применить морфологические операции (опционально):

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
```

- Подсчитать долю foreground-пикселей:

```python
foreground_pixels = cv2.countNonZero(fg_mask)

total_pixels = fg_mask.shape[0] * fg_mask.shape[1]

motion_value = foreground_pixels / total_pixels if total_pixels > 0 else 0.0
```

- Привязать к секунде (как в Frame Diff) и агрегировать.

4. **Построение heatmap и статистики** — то же, что в FrameDiff handler’е

5. **Заполнение контекста**

- В поле `detection_method` указать:
  - `"background_subtraction_mog2"` или `"background_subtraction_knn"`

---

## 9. Handler 3: Optical Flow — MotionAnalysisOpticalFlowHandler

### 9.1. Файл и сигнатура

- Файл: `handlers/motion_analysis_optical_flow_handler.py`
- Класс: `MotionAnalysisOpticalFlowHandler(BaseHandler)`

### 9.2. Конструктор и параметры

```python
class MotionAnalysisOpticalFlowHandler(BaseHandler):
    def __init__(
        self,
        resize_width: int = 320,
        frame_step: int = 1,
        num_levels: int = 5,
        pyr_scale: float = 0.5,
        win_size: int = 15,
        num_iters: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        use_gaussian: bool = False,
        motion_threshold: float = 0.1,
    ) -> None:
        ...
```

### 9.3. Объяснение параметров

- `num_levels` — количество уровней пирамиды (чем больше, тем лучше ловится глобальное движение, но медленнее)
- `pyr_scale` — масштаб между уровнями пирамиды (0.5 → каждый уровень в 2 раза меньше)
- `win_size` — размер окна для усреднения полинома (должен быть нечётным)
- `num_iters` — количество итераций уточнения на каждом уровне
- `poly_n`, `poly_sigma` — параметры полиномиальной аппроксимации
- `motion_threshold` — порог магнитуды нормированного вектора, выше которого пиксель считается «движущимся»

### 9.4. Логика `handle()` (по шагам)

1. Общая инициализация (аналогична предыдущим handler’ам)
2. В цикле по кадрам:
   - Конвертировать кадр в grayscale
   - Downscale при необходимости
   - Если `prev_gray is not None`, вызвать:

```python
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, gray,
    flow=None,
    pyr_scale=pyr_scale,
    levels=num_levels,
    winsize=win_size,
    iterations=num_iters,
    poly_n=poly_n,
    poly_sigma=poly_sigma,
    flags=0,
)
```

- Вычислить магнитуду и угол:

```python
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
```

- Нормализовать магнитуду:

```python
max_mag = float(np.max(mag)) if mag.size else 0.0
if max_mag <= 0:
    motion_value = 0.0
else:
    mag_norm = mag / max_mag
    moving_mask = mag_norm > motion_threshold
    motion_value = float(np.sum(moving_mask)) / float(mag_norm.size)
```

- Привязать к секунде и агрегировать (как в предыдущих методах)

3. Построение heatmap и статистики — по общей схеме
4. В `detection_method` записать `"optical_flow_farneback"`

---

## 10. Интеграция в общий пайплайн

### 10.1. Основной метод для пайплайна

В основном рабочем пайплайне использовать **только Frame Differencing**:

```python
from handlers.motion_analysis_frame_diff_handler import MotionAnalysisFrameDiffHandler

handlers = [
    ReadFileHandler(),
    FFmpegExtractHandler(),
    MotionAnalysisFrameDiffHandler(),
    # ... остальные handlers
]
```

### 10.2. Альтернативы для экспериментов

Опционально добавить переключатель (например, по аргументу командной строки) для выбора метода:

```python
if motion_method == "frame_diff":
    motion_handler = MotionAnalysisFrameDiffHandler()
elif motion_method == "background_sub":
    motion_handler = MotionAnalysisBackgroundSubHandler(method="mog2")
elif motion_method == "optical_flow":
    motion_handler = MotionAnalysisOpticalFlowHandler()
```

---

## 11. Тестирование и валидация

### 11.1. Набор тестовых видео

Рекомендуется проверить на:

1. **Статичное видео** (без движения)
   - Ожидается: низкий `mean`, `max` ≈ 0.1 или ниже
2. **Видео с постоянным движением**
   - Ожидается: `mean` высокий (0.5–0.8)
3. **Видео со вспышками движения**
   - Ожидается: пики `max` ≈ 1.0 в моменты экшена
4. **Видео с плавным движением камеры** (панорамы)
   - Оптический поток должен уверенно показывать движение

### 11.2. Критерии успешности

- Handler не падает на реальных видеофайлах разных форматов
- Обработанное видео имеет разумные значения `motion_heatmap`
- Время обработки для Frame Differencing — в рамках нескольких секунд на видео длиной 5 минут (при downscale + frame_step)
- Для Background Subtraction и Optical Flow — приемлемое время с учётом сложности алгоритмов

---

## 12. Итоги

В рамках этого ТЗ необходимо:

1. Выполнить полноценный **анализ задачи** и реализовать **3 метода детекции движения**:
   - Frame Differencing — основной метод пайплайна
   - Background Subtraction (MOG2/KNN) — альтернативный метод
   - Optical Flow (Farneback) — исследовательский метод

2. Реализовать для каждого метода отдельный handler с:
   - Единым интерфейсом `handle(context)`
   - Общим форматом выходных данных
   - Корректной обработкой ошибок и освобождением ресурсов

3. Обновить зависимости проекта (`opencv-python`, `numpy`).

4. Интегрировать основной handler (Frame Differencing) в общий пайплайн сразу после `FFmpegExtractHandler`.
