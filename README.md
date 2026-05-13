# 🎙️ Meeting Protocol — FluidAudio on Apple M4 GPU

Локальный инструмент для автоматической **транскрибации и диаризации** аудио/видео совещаний с разметкой по спикерам.  
Работает **полностью офлайн** на процессоре Apple M4 — все вычисления на GPU/ANE через CoreML.

---

## ✨ Возможности

| Функция | Детали |
|---------|--------|
| 🎤 Транскрибация | Parakeet TDT 0.6B v3 (NVIDIA → CoreML), русский + 25+ языков |
| 👥 Диаризация | FluidAudio offline pipeline, до 10 спикеров |
| ⚡ Скорость | ~150× real-time на M4 (57 мин → 23 сек) |
| 🔒 Приватность | Полностью локально, интернет только при первой загрузке моделей |
| 📥 Экспорт | TXT-протокол, SRT-субтитры, JSON |
| 🖥️ Интерфейс | Streamlit web-UI + CLI-скрипт |

### Почему так быстро?

FluidAudio использует **CoreML** — фреймворк Apple для запуска нейросетей. На M4 CoreML автоматически направляет вычисления на **ANE (Apple Neural Engine)** и **GPU**, обходя CPU. Результат: диаризация 57-минутного файла за **11.8 секунды** (RTFx 293×), транскрибация 393 сегментов параллельно за **21 секунду**.

---

## 📋 Требования

| Компонент | Версия |
|-----------|--------|
| macOS | 14.0+ (Sonoma) |
| Процессор | Apple Silicon (M1/M2/M3/M4) |
| Xcode Command Line Tools | 15+ |
| Swift | 6.0+ |
| Python | 3.9+ |
| ffmpeg | любая актуальная |
| Свободного места | ~3 ГБ (модели загружаются автоматически) |

---

## 🚀 Установка

### Актуальная ветка

Пока изменения не слиты в `main`, используйте ветку `codex/fix-python39-fluid-app`.
В ней исправлен запуск `app_fluid.py` на Python 3.9 и сохранён FluidAudio-пайплайн
для транскрибации и диаризации через CoreML на Apple Silicon.

### 1. Установите системные зависимости

```bash
# Xcode Command Line Tools (если не установлены)
xcode-select --install

# ffmpeg
brew install ffmpeg

# Python (если нужен)
brew install python@3.11
```

### 2. Клонируйте этот репозиторий

```bash
git clone --branch codex/fix-python39-fluid-app --single-branch https://github.com/damadan/meeting-protocol.git
cd meeting-protocol
```

### 3. Клонируйте и соберите FluidAudio

```bash
# Клонируем рядом с проектом — на уровень выше
cd ..
git clone --depth=1 https://github.com/FluidInference/FluidAudio.git
cd FluidAudio

# Сборка (~90 секунд на M4)
swift build -c release --product fluidaudiocli

# Проверяем
.build/release/fluidaudiocli transcribe --help
```

> **Важно:** скрипты ожидают бинарник по пути `../FluidAudio/.build/release/fluidaudiocli` относительно папки проекта. Если вы разместили FluidAudio в другом месте — обновите переменную `FLUID_CLI` в начале файлов `fluid_pipeline.py` и `app_fluid.py`.

### 4. Создайте Python-окружение

```bash
cd meeting-protocol

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install streamlit
```

> Никаких тяжёлых ML-зависимостей в Python — весь инференс выполняет Swift-бинарник через CoreML.

### 5. Первый запуск — загрузка моделей

При первом запуске FluidAudio автоматически скачает модели с HuggingFace (~2-3 ГБ):

- **Parakeet TDT 0.6B v3** — ASR-модель (транскрибация)
- **Дарайзер** — сегментация по спикерам (WeSpeaker embeddings + VBx кластеризация)

Модели кэшируются в `~/.cache/huggingface/` и при повторных запусках загружаются мгновенно.

---

## 💻 Использование

### Веб-интерфейс (Streamlit)

```bash
source .venv/bin/activate
streamlit run app_fluid.py
# → откроется http://localhost:8501
```

1. Загрузите MP4, MKV, WAV, MP3 или другой формат (до 10 ГБ)
2. Нажмите **▶ Запустить обработку**
3. Дождитесь результата (для 1 часа аудио — ~30-60 секунд)
4. При желании введите имена спикеров
5. Нажмите **⬇️ Скачать TXT** или выберите другой формат внизу

### CLI — командная строка

```bash
source .venv/bin/activate

# Базовый запуск — вывод в терминал
python3 fluid_pipeline.py meeting.mp4

# Сохранить все форматы
python3 fluid_pipeline.py meeting.mp4 \
    --output protocol.txt \
    --srt protocol.srt \
    --json-out protocol.json

# Если диаризация уже сделана — пропустить этот шаг
python3 fluid_pipeline.py meeting.mp4 \
    --diar-json diarization.json \
    --output protocol.txt

# Все опции
python3 fluid_pipeline.py --help
```

**Параметры CLI:**

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--workers N` | 8 | Потоков параллельной транскрипции |
| `--merge-gap SEC` | 1.0 | Склейка реплик одного спикера (сек) |
| `--output FILE` | — | Сохранить TXT-протокол |
| `--srt FILE` | — | Сохранить SRT-субтитры |
| `--json-out FILE` | — | Сохранить JSON |
| `--diar-json FILE` | — | Использовать готовый JSON диаризации |
| `--no-merge` | — | Не склеивать соседние реплики |

#### Пример вывода

```
============================================================
🎙️  FluidAudio Pipeline — M4 GPU/ANE (CoreML)
============================================================

📊 Железо:
  Chip: Apple M4
  Total Number of Cores: 10 (4 performance and 6 efficiency)

[1/3] 🔊 Извлечение аудио из meeting.mp4...
    ✓ 57.6 мин аудио → 1.5с

[2/3] 👥 Диаризация (CoreML на M4 GPU/ANE)...
    ✓ 6 спикеров, 393 сегментов — 11.8с (RTFx=293×)
    Спикеры: S1, S2, S3, S4, S5, S6

[3/3] 📝 Транскрибация 393 сегментов (8 потоков, CoreML на M4 GPU/ANE)...
  [████████████████████████████████████████] 393/393
    ✓ 392 сегментов распознано — 21.2с

⏱️  Итого: 22.9с на 57.6 мин аудио (RTFx≈151×)

📊 Статистика спикеров:
  S1: 80 реплик, 00:17:07 суммарно, ~2267 слов
  S2: 41 реплик, 00:19:03 суммарно, ~2404 слов
  S3: 18 реплик, 00:06:15 суммарно, ~890 слов
  S4: 22 реплик, 00:07:29 суммарно, ~1101 слов
  S5: 16 реплик, 00:02:39 суммарно, ~458 слов
  S6:  2 реплики, 00:00:14 суммарно, ~39 слов

============================================================
📋 ПРОТОКОЛ
============================================================
[00:00:03 — 00:00:22] S1:
Сегодня хотела бы проговорить про рассылки. Расскажите, пожалуйста,
как у вас выглядит процесс подготовки?

[00:02:42 — 00:03:41] S2:
Могу сказать от западного округа. У нас рассылки через Unisender.
Это платный сервис, оплата зависит от количества подписчиков...
```

---

## 🏗️ Архитектура

```
meeting-protocol/          ← этот репозиторий
├── app_fluid.py           # Streamlit веб-интерфейс
├── fluid_pipeline.py      # CLI-пайплайн
├── .streamlit/
│   └── config.toml        # maxUploadSize = 10240 МБ
└── README.md

../FluidAudio/             ← собирается отдельно
└── .build/release/
    └── fluidaudiocli      # Swift-бинарник (CoreML)
```

### Как работает пайплайн

```
MP4/WAV/MP3
  │
  ▼  ffmpeg (-ar 16000 -ac 1)
WAV (16 кГц, моно)
  │
  ├──► fluidaudiocli process --mode offline
  │         CoreML → Apple M4 ANE/GPU
  │         WeSpeaker embeddings → VBx кластеризация
  │         ↓
  │         diarization.json
  │         [{ startTimeSeconds, endTimeSeconds, speakerId }, ...]
  │
  └──► Параллельно для каждого сегмента (8 потоков):
            ffmpeg -ss START -t DUR  →  chunk.wav
            fluidaudiocli transcribe chunk.wav
            CoreML → Apple M4 ANE/GPU
            Parakeet TDT 0.6B v3 (русский, EN, 25+ языков)
            ↓
            текст сегмента
  │
  ▼  merge_consecutive(gap=1.0s)
Протокол: TXT / SRT / JSON
```

---

## 📊 Производительность на Apple M4

Реальный тест: совещание, 57 минут, 6 спикеров, русский язык.

| Этап | Время | RTFx |
|------|-------|------|
| Извлечение аудио (ffmpeg) | 1.5 сек | 2300× |
| Диаризация (CoreML/ANE) | 11.8 сек | **293×** |
| Транскрибация 393 сег. × 8 потоков | 21.2 сек | ~800× сегмент |
| **Итого** | **22.9 сек** | **≈151×** |

Для сравнения: тот же файл на CPU-only пайплайне (DiariZen + GigaAM ONNX) обрабатывается за 15-25 минут.

---

## 🔧 Настройка

**Лимит загружаемых файлов** — файл `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 10240   # МБ (сейчас 10 ГБ)
```

**Количество параллельных потоков** — в `app_fluid.py` и `fluid_pipeline.py`:
```python
ASR_WORKERS = 8   # можно поднять до 10 на M4 (10 ядер)
```

**Путь к FluidAudio CLI** — если бинарник лежит не рядом:
```python
FLUID_CLI = Path("/absolute/path/to/FluidAudio/.build/release/fluidaudiocli")
```

---

## 🛠️ Устранение проблем

**`fluidaudiocli` не найден:**
```bash
ls ../FluidAudio/.build/release/fluidaudiocli
# Если файла нет — пересоберите:
cd ../FluidAudio && swift build -c release --product fluidaudiocli
```

**Модели долго скачиваются при первом запуске:**  
Нормально — Parakeet TDT v3 весит ~1.2 ГБ, дарайзер ~800 МБ.  
После первого запуска всё кэшируется в `~/.cache/huggingface/`.

**Диаризация: 0 спикеров:**  
Проверьте качество аудио (минимум 8 кГц, моно). Попробуйте более длинный фрагмент — короткие клипы (< 30 сек) могут давать 0 спикеров.

**Ошибка `swift build` / `module not found`:**
```bash
xcode-select --install      # Установить CLT
xcode-select -p             # Проверить путь
swift --version             # Должно быть 6.0+
```

**Ошибка `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`:**  
Обновите репозиторий до актуальной версии. Эта ошибка возникала в Python 3.9 из-за
ранней оценки type hints в `app_fluid.py`.

**Streamlit не запускается:**
```bash
source .venv/bin/activate
pip install streamlit
streamlit run app_fluid.py
```

---

## 📦 Зависимости

**Python:**
- `streamlit` — веб-интерфейс

**Системные:**
- `ffmpeg` — конвертация аудио/видео
- `swift` / Xcode CLT — сборка FluidAudio

**Сторонние проекты:**
- [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio) — MIT
- [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) — CC BY 4.0
- [BUTSpeechFIT/DiariZen](https://github.com/BUTSpeechFIT/DiariZen) — CC BY-NC 4.0 (используется в `app.py`)

---

## 📄 Лицензия

MIT

---

## 🙏 Благодарности

- [FluidInference](https://github.com/FluidInference) — за отличный CoreML-пайплайн для аудио на Apple Silicon
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — оригинальная модель Parakeet TDT
- [BUTSpeechFIT](https://github.com/BUTSpeechFIT) — DiariZen (Python-дарайзер, `app.py`)
