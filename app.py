"""
🎙️ Протокол совещания — DiariZen + GigaAM-v3 (ONNX)
Автоматическая расшифровка аудио/видео с определением спикеров.
Полностью локальный пайплайн на CPU.

ASR: onnx-asr + GigaAM-v3 E2E CTC (ONNX Runtime, 2–4× быстрее PyTorch)
Диаризация: DiariZen (PyTorch)
"""

import streamlit as st
import torch
import torchaudio
import numpy as np
import subprocess
import tempfile
import os
import gc
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── CPU-оптимизации (до всего остального) ──────────────────────────
try:
    _ncpu = os.cpu_count() or 4
    torch.set_num_threads(_ncpu)          # все 10 ядер M4 для PyTorch
    torch.set_num_interop_threads(min(6, _ncpu))  # параллельные операции
except RuntimeError:
    pass  # уже установлено при повторном запуске Streamlit

# Патч torch.load: PyTorch 2.6+ требует weights_only=True по умолчанию,
# но старые чекпоинты DiariZen/pyannote несовместимы с этим режимом.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# RAM-диск для темпов (Linux: /dev/shm, иначе системный tmpdir)
RAMDISK = "/dev/shm/speech_tmp"
if os.path.isdir("/dev/shm"):
    os.makedirs(RAMDISK, exist_ok=True)
    tempfile.tempdir = RAMDISK

# ─── Конфигурация ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Протокол совещания",
    page_icon="🎙️",
    layout="wide",
)

SAMPLE_RATE = 16000
MAX_CHUNK_SEC = 24.0
MIN_SEGMENT_SEC = 0.3
DEFAULT_MERGE_GAP = 1.5
PREVIEW_LINES = 2
ASR_WORKERS = 4  # потоки параллельной транскрипции (M4 = 10 ядер)

MODELS_DIR = Path(__file__).parent / "models"
DIARIZEN_HUB = MODELS_DIR / "diarizen-wavlm-large-s80-md"
WESPEAKER_MODEL = MODELS_DIR / "wespeaker-voxceleb-resnet34-LM" / "speaker-embedding.onnx"
ONNX_ASR_DIR = MODELS_DIR / "gigaam-v3-onnx"
ONNX_ASR_MODEL = "gigaam-v3-e2e-ctc"

VIDEO_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "webm"}
AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "wma", "aac"}
ALL_EXTENSIONS = sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)


# ─── Утилиты ────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def is_video(filename: str) -> bool:
    return Path(filename).suffix.lstrip(".").lower() in VIDEO_EXTENSIONS


# ─── Загрузка моделей (кэшируется) ──────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_diarization_model():
    from diarizen.pipelines.inference import DiariZenPipeline

    # Увеличенный batch_size для M4 (10 ядер, ≥16 ГБ RAM)
    config_override = {
        "inference": {"args": {
            "seg_duration": 16,
            "segmentation_step": 0.1,
            "batch_size": 64,
            "apply_median_filtering": True,
        }},
        "clustering": {"args": {
            "method": "VBxClustering",
            "min_speakers": 1,
            "max_speakers": 20,
            "ahc_criterion": "distance",
            "ahc_threshold": 0.6,
            "Fa": 0.07,
            "Fb": 0.8,
            "lda_dim": 128,
            "max_iters": 20,
        }},
    }
    pipeline = DiariZenPipeline(
        diarizen_hub=DIARIZEN_HUB,
        embedding_model=str(WESPEAKER_MODEL),
        config_parse=config_override,
    )
    if hasattr(pipeline, "to"):
        pipeline.to(torch.device("cpu"))
    return pipeline


@st.cache_resource(show_spinner=False)
def load_onnx_asr_model():
    """Загружает GigaAM-v3 E2E CTC через onnx-asr (ONNX Runtime) из локальной папки."""
    import onnx_asr

    model = onnx_asr.load_model(ONNX_ASR_MODEL, path=str(ONNX_ASR_DIR))
    return model


# ─── Извлечение аудио из видео ──────────────────────────────────────
def extract_audio_from_video(video_path: str) -> str:
    output_path = video_path.rsplit(".", 1)[0] + "_extracted.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg ошибка:\n{result.stderr[-500:]}")
    return output_path


# ─── Предобработка аудио ────────────────────────────────────────────
def preprocess_audio(uploaded_file) -> tuple[str, torch.Tensor, int]:
    suffix = Path(uploaded_file.name).suffix
    uploaded_file.seek(0)  # сброс курсора на случай повторного запуска
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    if is_video(uploaded_file.name):
        wav_path = extract_audio_from_video(input_path)
        try:
            os.unlink(input_path)
        except OSError:
            pass
        wav, sr = torchaudio.load(wav_path)
        # ffmpeg уже конвертирует в моно 16kHz, но проверим на всякий случай
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav_path, wav, sr

    wav, sr = torchaudio.load(input_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    output_path = input_path.rsplit(".", 1)[0] + "_16k.wav"
    torchaudio.save(output_path, wav, SAMPLE_RATE)

    try:
        os.unlink(input_path)
    except OSError:
        pass

    return output_path, wav, SAMPLE_RATE


# ─── Диаризация ─────────────────────────────────────────────────────
def run_diarization(audio_path: str, pipeline) -> list[dict]:
    annotation = pipeline(audio_path)
    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        try:
            label = f"Спикер {int(speaker) + 1}"
        except (ValueError, TypeError):
            label = f"Спикер {speaker}"
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": label,
        })
    return segments


# ─── ASR (ONNX) — транскрипция одного сегмента ──────────────────────
def transcribe_segment_onnx(
    wav: torch.Tensor,
    sr: int,
    start: float,
    end: float,
    asr_model,
) -> str:
    """Вырезает чанк и транскрибирует через ONNX-модель."""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    chunk = wav[0, start_sample:end_sample].numpy()  # numpy array
    duration = end - start

    if duration < MIN_SEGMENT_SEC:
        return ""

    if duration <= MAX_CHUNK_SEC:
        return _transcribe_numpy_chunk(chunk, sr, asr_model)

    # Длинный сегмент — разбиваем
    texts = []
    max_samples = int(MAX_CHUNK_SEC * sr)
    offset = 0
    while offset < len(chunk):
        sub_end = min(offset + max_samples, len(chunk))
        text = _transcribe_numpy_chunk(chunk[offset:sub_end], sr, asr_model)
        if text:
            texts.append(text)
        offset = sub_end
    return " ".join(texts)


def _transcribe_numpy_chunk(samples: np.ndarray, sr: int, asr_model) -> str:
    """Транскрибирует numpy-массив через onnx-asr."""
    try:
        # onnx-asr принимает numpy array или путь к WAV
        # Пробуем передать numpy напрямую (быстрее, без I/O)
        text = asr_model.recognize(samples, sample_rate=sr)
        return text.strip() if isinstance(text, str) else str(text).strip()
    except TypeError:
        # Фолбэк: запись во временный WAV
        return _transcribe_via_file(samples, sr, asr_model)
    except Exception as e:
        print(f"[ASR] Ошибка транскрипции: {e}")
        return ""


def _transcribe_via_file(samples: np.ndarray, sr: int, asr_model) -> str:
    """Фолбэк: пишем WAV на диск и передаём путь."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, samples, sr, subtype="PCM_16")
        tmp_path = tmp.name
    try:
        text = asr_model.recognize(tmp_path)
        return text.strip() if isinstance(text, str) else str(text).strip()
    except Exception:
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─── Параллельная транскрипция всех сегментов ───────────────────────
def transcribe_all_segments(
    diar_segments: list[dict],
    wav: torch.Tensor,
    sr: int,
    asr_model,
    progress,
    p_start: float,
    p_end: float,
) -> list[dict]:
    total = len(diar_segments)
    results = [None] * total
    done = 0

    def _do(idx, seg):
        text = transcribe_segment_onnx(wav, sr, seg["start"], seg["end"], asr_model)
        return idx, seg, text

    with ThreadPoolExecutor(max_workers=ASR_WORKERS) as pool:
        futures = {
            pool.submit(_do, i, seg): i
            for i, seg in enumerate(diar_segments)
        }
        for future in as_completed(futures):
            idx, seg, text = future.result()
            done += 1
            pct = p_start + (p_end - p_start) * (done / max(total, 1))
            progress.progress(min(pct, 1.0), text=f"Распознавание речи… ({done}/{total})")
            if text:
                results[idx] = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"],
                    "text": text,
                }

    return [r for r in results if r]


# ─── Пост-обработка ─────────────────────────────────────────────────
def merge_consecutive(results, merge_gap):
    if not results:
        return []
    merged = [results[0].copy()]
    for item in results[1:]:
        prev = merged[-1]
        if item["speaker"] == prev["speaker"] and (item["start"] - prev["end"]) < merge_gap:
            prev["text"] += " " + item["text"]
            prev["end"] = item["end"]
        else:
            merged.append(item.copy())
    return merged


def apply_speaker_names(results: list[dict], name_map: dict) -> list[dict]:
    renamed = []
    for r in results:
        entry = r.copy()
        entry["speaker"] = name_map.get(r["speaker"], r["speaker"])
        renamed.append(entry)
    return renamed


def get_speaker_previews(results: list[dict], n: int = PREVIEW_LINES) -> dict[str, list[dict]]:
    previews: dict[str, list[dict]] = {}
    for r in results:
        spk = r["speaker"]
        if spk not in previews:
            previews[spk] = []
        if len(previews[spk]) < n:
            previews[spk].append(r)
    return previews


# ─── Форматы экспорта ───────────────────────────────────────────────
def to_protocol_txt(results):
    lines = []
    for r in results:
        lines.append(f"[{fmt_time(r['start'])} — {fmt_time(r['end'])}] {r['speaker']}:")
        lines.append(r["text"])
        lines.append("")
    return "\n".join(lines)


def to_srt(results):
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(str(i))
        blocks.append(f"{fmt_srt_time(r['start'])} --> {fmt_srt_time(r['end'])}")
        blocks.append(f"[{r['speaker']}] {r['text']}")
        blocks.append("")
    return "\n".join(blocks)


def to_json(results):
    return json.dumps(results, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
#                       ИНТЕРФЕЙС STREAMLIT
# ═══════════════════════════════════════════════════════════════════

st.title("🎙️ Протокол совещания")
st.caption(
    "Загрузите аудио или видео — получите текстовый протокол "
    "с разметкой по спикерам. Работает полностью локально на CPU."
)

# ─── Session state ──────────────────────────────────────────────────
if "merged_results" not in st.session_state:
    st.session_state.merged_results = None
if "speaker_names" not in st.session_state:
    st.session_state.speaker_names = {}

# ─── Сайдбар ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Настройки")

    merge_gap = st.slider(
        "Склейка реплик одного спикера (сек)",
        min_value=0.0,
        max_value=5.0,
        value=DEFAULT_MERGE_GAP,
        step=0.5,
        help="Если пауза между репликами одного спикера меньше этого — склеиваем в одну.",
    )

    st.divider()
    st.markdown(
        "**Модели:**\n\n"
        "🔹 **Диаризация** — "
        "[DiariZen](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md)  \n"
        "WavLM Large, pruned 80%, ~63M пар.\n\n"
        "🔹 **ASR** — "
        "[GigaAM-v3 E2E CTC](https://huggingface.co/ai-sage/GigaAM-v3)  \n"
        "ONNX Runtime, ~240M пар., с пунктуацией.\n\n"
        "---\n"
        "⚡ **Оптимизации:**\n"
        "- ONNX Runtime (2–4× быстрее)\n"
        f"- {os.cpu_count()} потоков CPU\n"
        f"- {ASR_WORKERS} потока параллельной ASR\n"
        f"- RAM-диск: {'да' if os.path.isdir('/dev/shm') else 'нет'}\n\n"
        "---\n"
        "**Лицензии:**\n"
        "- DiariZen: CC BY-NC 4.0\n"
        "- GigaAM: MIT"
    )

# ─── Загрузка файла ─────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Загрузите аудио или видеофайл",
    type=ALL_EXTENSIONS,
    help="WAV, MP3, FLAC, OGG, M4A, MP4, MKV, AVI, MOV, WEBM.",
)

if uploaded is not None:
    if is_video(uploaded.name):
        st.video(uploaded)
    else:
        st.audio(uploaded, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")

    if st.button("▶ Запустить обработку", type="primary", use_container_width=True):
        st.session_state.merged_results = None
        st.session_state.speaker_names = {}

        t_start = time.time()
        progress = st.progress(0.0, text="Подготовка…")

        # 1. Предобработка
        if is_video(uploaded.name):
            progress.progress(0.02, text="Извлечение аудио из видео (ffmpeg)…")

        try:
            audio_path, wav, sr = preprocess_audio(uploaded)
        except Exception as e:
            st.error(f"Не удалось обработать файл: {e}")
            st.info("Убедитесь, что установлен **ffmpeg** (`sudo apt install ffmpeg`).")
            st.stop()

        duration_sec = wav.shape[1] / sr
        c1, c2 = st.columns(2)
        c1.metric("Длительность", fmt_time(duration_sec))
        c2.metric("Формат", f"{sr} Гц, моно")

        # 2. Диаризация
        progress.progress(0.05, text="Загрузка модели диаризации…")
        try:
            diar_pipeline = load_diarization_model()
        except Exception as e:
            st.error(f"Ошибка загрузки DiariZen: {e}")
            st.code(
                "git clone https://github.com/BUTSpeechFIT/DiariZen.git\n"
                "cd DiariZen && pip install -e .\n"
                "cd pyannote-audio && pip install -e .",
                language="bash",
            )
            st.stop()

        progress.progress(0.10, text="Диаризация — определение спикеров…")
        diar_segments = run_diarization(audio_path, diar_pipeline)

        n_speakers = len(set(s["speaker"] for s in diar_segments))
        c3, c4 = st.columns(2)
        c3.metric("Спикеров", n_speakers)
        c4.metric("Сегментов", len(diar_segments))

        if not diar_segments:
            st.warning("Спикеры не обнаружены. Проверьте качество аудио.")
            os.unlink(audio_path)
            st.stop()

        # 3. ASR (ONNX)
        progress.progress(0.30, text="Загрузка ASR-модели (ONNX Runtime)…")
        try:
            asr_model = load_onnx_asr_model()
        except Exception as e:
            st.error(f"Ошибка загрузки onnx-asr: {e}")
            st.code("pip install onnx-asr", language="bash")
            st.stop()

        results = transcribe_all_segments(
            diar_segments, wav, sr, asr_model, progress, 0.35, 0.90
        )

        # 4. Пост-обработка
        progress.progress(0.95, text="Формирование протокола…")
        merged = merge_consecutive(results, merge_gap)

        try:
            os.unlink(audio_path)
        except OSError:
            pass
        gc.collect()

        elapsed = time.time() - t_start
        progress.progress(1.0, text=f"Готово за {elapsed:.0f} сек.")

        if not merged:
            st.warning("Не удалось распознать речь. Проверьте аудио.")
            st.stop()

        st.session_state.merged_results = merged
        speakers = sorted(set(r["speaker"] for r in merged))
        st.session_state.speaker_names = {s: "" for s in speakers}


# ═══════════════════════════════════════════════════════════════════
#        РЕЗУЛЬТАТЫ + ПЕРЕИМЕНОВАНИЕ СПИКЕРОВ
# ═══════════════════════════════════════════════════════════════════

if st.session_state.merged_results:
    merged = st.session_state.merged_results

    palette = [
        "#5B9BD5", "#E06666", "#6AA84F", "#F6B26B",
        "#8E7CC3", "#C27BA0", "#76A5AF", "#FFD966",
    ]
    speakers = sorted(set(r["speaker"] for r in merged))
    colors = {s: palette[i % len(palette)] for i, s in enumerate(speakers)}

    # ── Переименование спикеров ─────────────────────────────────────
    st.divider()
    st.subheader("👤 Идентификация спикеров")
    st.caption(
        "Посмотрите на фрагменты речи каждого спикера и введите имя. "
        "Протокол обновится автоматически."
    )

    previews = get_speaker_previews(merged)

    for spk in speakers:
        c = colors[spk]
        with st.container(border=True):
            col_preview, col_name = st.columns([3, 1])

            with col_preview:
                st.markdown(
                    f'<span style="color:{c};font-weight:700;font-size:1.1em;">'
                    f'{spk}</span>',
                    unsafe_allow_html=True,
                )
                for p in previews.get(spk, []):
                    st.markdown(
                        f'<span style="color:#888;font-size:0.8em;">'
                        f'{fmt_time(p["start"])}–{fmt_time(p["end"])}</span> '
                        f'<span style="font-size:0.92em;">«{p["text"][:120]}'
                        f'{"…" if len(p["text"]) > 120 else ""}»</span>',
                        unsafe_allow_html=True,
                    )

            with col_name:
                new_name = st.text_input(
                    "Имя",
                    value=st.session_state.speaker_names.get(spk, ""),
                    key=f"name_{spk}",
                    placeholder="напр. Иван",
                    label_visibility="collapsed",
                )
                st.session_state.speaker_names[spk] = new_name

    name_map = {}
    for spk in speakers:
        custom = st.session_state.speaker_names.get(spk, "").strip()
        name_map[spk] = custom if custom else spk

    final = apply_speaker_names(merged, name_map)

    final_speakers = sorted(set(r["speaker"] for r in final))
    final_colors = {}
    for spk_orig, spk_new in name_map.items():
        final_colors[spk_new] = colors[spk_orig]

    # ── Протокол ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📝 Протокол")

    for r in final:
        c = final_colors.get(r["speaker"], "#999")
        st.markdown(
            f'<div style="margin-bottom:12px;">'
            f'<span style="color:{c};font-weight:700;font-size:1.05em;">'
            f'{r["speaker"]}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:#999;font-size:0.82em;">'
            f'{fmt_time(r["start"])} — {fmt_time(r["end"])}</span>'
            f'<br/>'
            f'<span style="font-size:0.97em;">{r["text"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Экспорт ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Экспорт")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "📄 TXT", to_protocol_txt(final),
            file_name="protocol.txt", mime="text/plain",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "🎬 SRT", to_srt(final),
            file_name="protocol.srt", mime="text/plain",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            "📊 JSON", to_json(final),
            file_name="protocol.json", mime="application/json",
            use_container_width=True,
        )

    # ── Статистика ──────────────────────────────────────────────────
    with st.expander("📈 Статистика"):
        for spk in final_speakers:
            spk_items = [r for r in final if r["speaker"] == spk]
            total_dur = sum(r["end"] - r["start"] for r in spk_items)
            total_words = sum(len(r["text"].split()) for r in spk_items)
            c = final_colors.get(spk, "#999")
            st.markdown(
                f'<span style="color:{c};font-weight:700;">'
                f'{spk}</span>: '
                f'{len(spk_items)} реплик, '
                f'{fmt_time(total_dur)} суммарно, '
                f'~{total_words} слов',
                unsafe_allow_html=True,
            )
