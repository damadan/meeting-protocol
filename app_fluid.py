"""
🎙️ Протокол совещания — FluidAudio (CoreML / M4 GPU/ANE)
Транскрибация + диаризация MP4/аудио полностью локально на M4 GPU.

ASR:         FluidAudio Parakeet TDT (CoreML → ANE/GPU)
Диаризация:  FluidAudio offline pipeline (CoreML → ANE/GPU)
RTFx:        ~150–300× real-time (57 мин за 23 сек)
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st

# ─── Конфигурация ───────────────────────────────────────────────────
FLUID_CLI = Path(__file__).parent.parent / "FluidAudio/.build/release/fluidaudiocli"
FFMPEG = "ffmpeg"
SAMPLE_RATE = 16000
MIN_SEG_SEC = 0.5
DEFAULT_MERGE_GAP = 1.0
ASR_WORKERS = 8
PREVIEW_LINES = 2

VIDEO_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "webm"}
AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "wma", "aac"}
ALL_EXTENSIONS = sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS)

st.set_page_config(
    page_title="Протокол совещания · FluidAudio",
    page_icon="🎙️",
    layout="wide",
)


# ─── Утилиты ────────────────────────────────────────────────────────
def fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt_time(sec: float) -> str:
    ms = int((sec % 1) * 1000)
    return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{int(sec%60):02d},{ms:03d}"


def is_video(name: str) -> bool:
    return Path(name).suffix.lstrip(".").lower() in VIDEO_EXTENSIONS


def check_cli() -> bool:
    return FLUID_CLI.exists()


# ─── Шаг 1: извлечение WAV ──────────────────────────────────────────
def extract_wav(input_path: str, out_wav: str) -> float:
    result = subprocess.run([
        FFMPEG, "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        out_wav,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg:\n{result.stderr[-300:]}")
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries",
         "format=duration", "-of", "csv=p=0", out_wav],
        capture_output=True, text=True,
    )
    return float(probe.stdout.strip() or 0)


# ─── Шаг 2: диаризация ──────────────────────────────────────────────
def run_diarization(wav_path: str, out_json: str) -> dict:
    result = subprocess.run([
        str(FLUID_CLI), "process", wav_path,
        "--mode", "offline",
        "--output", out_json,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"fluidaudiocli process:\n{result.stderr[-300:]}")
    with open(out_json) as f:
        return json.load(f)


# ─── Шаг 3: транскрибация ───────────────────────────────────────────
def _cut_wav(src: str, start: float, end: float, dst: str) -> None:
    subprocess.run([
        FFMPEG, "-y",
        "-ss", str(start), "-t", str(end - start),
        "-i", src, "-acodec", "copy", dst,
    ], capture_output=True, check=True)


def _transcribe(wav: str) -> str:
    r = subprocess.run(
        [str(FLUID_CLI), "transcribe", wav],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def transcribe_segment(wav_path: str, seg: dict, tmpdir: str) -> dict | None:
    start, end = seg["startTimeSeconds"], seg["endTimeSeconds"]
    if end - start < MIN_SEG_SEC:
        return None
    chunk = os.path.join(tmpdir, f"seg_{start:.3f}.wav")
    _cut_wav(wav_path, start, end, chunk)
    text = _transcribe(chunk)
    try:
        os.unlink(chunk)
    except OSError:
        pass
    return {"start": start, "end": end, "speaker": seg["speakerId"], "text": text} if text else None


def transcribe_all(wav_path: str, segments: list, progress, p0: float, p1: float, tmpdir: str) -> list:
    total = len(segments)
    indexed = [None] * total
    done = 0

    with ThreadPoolExecutor(max_workers=ASR_WORKERS) as pool:
        futures = {
            pool.submit(transcribe_segment, wav_path, seg, tmpdir): i
            for i, seg in enumerate(segments)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                indexed[idx] = future.result()
            except Exception:
                pass
            done += 1
            pct = p0 + (p1 - p0) * (done / total)
            progress.progress(min(pct, 1.0), text=f"Транскрибация… {done}/{total} сегментов")

    return [r for r in indexed if r]


# ─── Пост-обработка ─────────────────────────────────────────────────
def merge_consecutive(results: list, gap: float) -> list:
    if not results:
        return []
    merged = [results[0].copy()]
    for item in results[1:]:
        prev = merged[-1]
        if item["speaker"] == prev["speaker"] and (item["start"] - prev["end"]) < gap:
            prev["text"] += " " + item["text"]
            prev["end"] = item["end"]
        else:
            merged.append(item.copy())
    return merged


def apply_names(results: list, name_map: dict) -> list:
    return [{**r, "speaker": name_map.get(r["speaker"], r["speaker"])} for r in results]


def get_previews(results: list, n: int = PREVIEW_LINES) -> dict:
    out: dict = {}
    for r in results:
        sp = r["speaker"]
        if sp not in out:
            out[sp] = []
        if len(out[sp]) < n:
            out[sp].append(r)
    return out


# ─── Экспорт ────────────────────────────────────────────────────────
def to_txt(results: list) -> str:
    lines = []
    for r in results:
        lines += [f"[{fmt_time(r['start'])} — {fmt_time(r['end'])}] {r['speaker']}:", r["text"], ""]
    return "\n".join(lines)


def to_srt(results: list) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks += [str(i), f"{fmt_srt_time(r['start'])} --> {fmt_srt_time(r['end'])}",
                   f"[{r['speaker']}] {r['text']}", ""]
    return "\n".join(blocks)


def to_json_str(results: list) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
#                          ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════════

st.title("🎙️ Протокол совещания")
st.caption("FluidAudio · Parakeet TDT + дарайзер · CoreML → **M4 GPU/ANE** · полностью локально")

# ── Предупреждение если CLI не найден ──────────────────────────────
if not check_cli():
    st.error(
        f"⚠️ `fluidaudiocli` не найден: `{FLUID_CLI}`\n\n"
        "Соберите бинарник:\n"
        "```bash\n"
        "cd ~/Desktop/FluidAudio\n"
        "swift build -c release --product fluidaudiocli\n"
        "```"
    )
    st.stop()

# ── Session state ───────────────────────────────────────────────────
for key, default in [
    ("merged_results", None),
    ("speaker_names", {}),
    ("diar_stats", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Сайдбар ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Настройки")

    merge_gap = st.slider(
        "Склейка реплик (сек)", 0.0, 5.0, DEFAULT_MERGE_GAP, 0.5,
        help="Паузы короче этого значения → склеиваем реплики одного спикера",
    )

    st.divider()
    st.markdown(
        "**🔧 Движок:**\n\n"
        "```\n"
        "FluidAudio CLI\n"
        "Parakeet TDT 0.6B v3\n"
        "CoreML → M4 ANE/GPU\n"
        "```\n\n"
        f"⚡ **{ASR_WORKERS} потоков** параллельной ASR  \n"
        "📊 **RTFx ~150×** (57 мин → ~23 сек)\n\n"
        "---\n"
        "Все модели загружаются **автоматически**  \n"
        "из HuggingFace при первом запуске\n\n"
        "---\n"
        "**Лицензии:** FluidAudio — MIT"
    )

# ── Загрузка файла ──────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Загрузите аудио или видеофайл",
    type=ALL_EXTENSIONS,
    help="WAV, MP3, FLAC, M4A, MP4, MKV, MOV, WebM…",
)

if uploaded is not None:
    if is_video(uploaded.name):
        st.video(uploaded)
    else:
        st.audio(uploaded, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")

    if st.button("▶ Запустить обработку", type="primary", use_container_width=True):
        st.session_state.merged_results = None
        st.session_state.speaker_names = {}
        st.session_state.diar_stats = {}

        t_total = time.time()
        progress = st.progress(0.0, text="Подготовка…")

        with tempfile.TemporaryDirectory(prefix="fluid_st_") as tmpdir:

            # 1. Сохраняем загруженный файл
            uploaded.seek(0)
            suffix = Path(uploaded.name).suffix
            input_path = os.path.join(tmpdir, "input" + suffix)
            with open(input_path, "wb") as f:
                f.write(uploaded.read())

            # 2. Извлекаем WAV
            progress.progress(0.03, text="Извлечение аудио (ffmpeg)…")
            wav_path = os.path.join(tmpdir, "audio.wav")
            try:
                duration = extract_wav(input_path, wav_path)
            except Exception as e:
                st.error(f"Ошибка при конвертации: {e}")
                st.stop()

            c1, c2 = st.columns(2)
            c1.metric("Длительность", fmt_time(duration))
            c2.metric("Формат", f"{SAMPLE_RATE} Гц, моно")

            # 3. Диаризация
            progress.progress(0.05, text="Диаризация (FluidAudio · CoreML M4 GPU)…")
            diar_json = os.path.join(tmpdir, "diarization.json")
            t0 = time.time()
            try:
                diar_data = run_diarization(wav_path, diar_json)
            except Exception as e:
                st.error(f"Ошибка диаризации: {e}")
                st.stop()

            diar_time = time.time() - t0
            rtfx = diar_data.get("realTimeFactor", 0)
            segments = [
                s for s in diar_data["segments"]
                if (s["endTimeSeconds"] - s["startTimeSeconds"]) >= MIN_SEG_SEC
            ]
            n_spk = diar_data.get("speakerCount", 0)

            c3, c4, c5 = st.columns(3)
            c3.metric("Спикеров", n_spk)
            c4.metric("Сегментов", len(segments))
            c5.metric("Диаризация", f"{diar_time:.1f}с  (RTFx {rtfx:.0f}×)")

            if not segments:
                st.warning("Спикеры не обнаружены. Проверьте качество аудио.")
                st.stop()

            # 4. Транскрибация
            progress.progress(0.15, text=f"Транскрибация {len(segments)} сегментов (8 потоков, CoreML M4 GPU)…")
            t0 = time.time()
            results = transcribe_all(wav_path, segments, progress, 0.15, 0.90, tmpdir)
            asr_time = time.time() - t0

            c6, c7 = st.columns(2)
            c6.metric("Распознано", f"{len(results)} реплик")
            c7.metric("ASR-время", f"{asr_time:.1f}с")

            if not results:
                st.warning("Речь не распознана. Проверьте аудио.")
                st.stop()

        # 5. Пост-обработка (вне tmpdir — файлы уже не нужны)
        progress.progress(0.95, text="Формирование протокола…")
        merged = merge_consecutive(results, merge_gap)

        total_time = time.time() - t_total
        progress.progress(1.0, text=f"Готово за {total_time:.1f}с (RTFx ≈ {duration/total_time:.0f}×)")

        st.session_state.merged_results = merged
        st.session_state.speaker_names = {
            s: "" for s in sorted(set(r["speaker"] for r in merged))
        }
        st.session_state.diar_stats = {
            "total_time": total_time,
            "duration": duration,
            "diar_rtfx": rtfx,
        }
        gc.collect()


# ═══════════════════════════════════════════════════════════════════
#               РЕЗУЛЬТАТЫ + ПЕРЕИМЕНОВАНИЕ
# ═══════════════════════════════════════════════════════════════════

if st.session_state.merged_results:
    merged: list = st.session_state.merged_results
    stats = st.session_state.diar_stats

    palette = ["#5B9BD5", "#E06666", "#6AA84F", "#F6B26B",
               "#8E7CC3", "#C27BA0", "#76A5AF", "#FFD966"]
    speakers = sorted(set(r["speaker"] for r in merged))
    colors = {s: palette[i % len(palette)] for i, s in enumerate(speakers)}

    # ── Скорость / GPU-верификация ──────────────────────────────────
    st.divider()
    with st.expander("⚡ Производительность M4 GPU/ANE", expanded=False):
        dur = stats.get("duration", 0)
        tt = stats.get("total_time", 1)
        st.markdown(
            f"| Метрика | Значение |\n|---|---|\n"
            f"| Длительность аудио | {fmt_time(dur)} |\n"
            f"| Итоговое время | {tt:.1f}с |\n"
            f"| RTFx (итого) | **{dur/tt:.0f}×** |\n"
            f"| RTFx (диаризация) | **{stats.get('diar_rtfx',0):.0f}×** |\n"
            f"| Движок | CoreML → **Apple M4 ANE/GPU** |\n"
            f"| ASR модель | Parakeet TDT 0.6B v3 |\n"
            f"| Диаризация | FluidAudio offline pipeline |"
        )

    # ── Идентификация спикеров ──────────────────────────────────────
    st.divider()
    st.subheader("👤 Идентификация спикеров")
    st.caption("Посмотрите фрагменты речи → введите имена спикеров")

    previews = get_previews(merged)

    for spk in speakers:
        c = colors[spk]
        with st.container(border=True):
            col_p, col_n = st.columns([3, 1])
            with col_p:
                st.markdown(
                    f'<span style="color:{c};font-weight:700;font-size:1.1em;">{spk}</span>',
                    unsafe_allow_html=True,
                )
                for p in previews.get(spk, []):
                    preview_text = p["text"][:130] + ("…" if len(p["text"]) > 130 else "")
                    st.markdown(
                        f'<span style="color:#888;font-size:0.8em;">'
                        f'{fmt_time(p["start"])}–{fmt_time(p["end"])}</span>'
                        f' <span style="font-size:0.92em;">«{preview_text}»</span>',
                        unsafe_allow_html=True,
                    )
            with col_n:
                new_name = st.text_input(
                    "Имя", value=st.session_state.speaker_names.get(spk, ""),
                    key=f"name_{spk}", placeholder="напр. Иван",
                    label_visibility="collapsed",
                )
                st.session_state.speaker_names[spk] = new_name

    name_map = {
        sp: (st.session_state.speaker_names.get(sp, "").strip() or sp)
        for sp in speakers
    }
    final = apply_names(merged, name_map)
    final_speakers = sorted(set(r["speaker"] for r in final))
    final_colors = {name_map[sp]: colors[sp] for sp in speakers}

    # ── Протокол ────────────────────────────────────────────────────
    st.divider()
    proto_col, btn_col = st.columns([3, 1])
    with proto_col:
        st.subheader("📝 Протокол")
    with btn_col:
        st.download_button(
            "⬇️ Скачать TXT",
            data=to_txt(final),
            file_name="protocol.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary",
        )

    for r in final:
        c = final_colors.get(r["speaker"], "#999")
        st.markdown(
            f'<div style="margin-bottom:14px;">'
            f'<span style="color:{c};font-weight:700;font-size:1.05em;">{r["speaker"]}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="color:#999;font-size:0.82em;">{fmt_time(r["start"])} — {fmt_time(r["end"])}</span>'
            f'<br/>'
            f'<span style="font-size:0.97em;">{r["text"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Экспорт ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Экспорт")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("📄 TXT", to_txt(final),
                           file_name="protocol.txt", mime="text/plain",
                           use_container_width=True)
    with d2:
        st.download_button("🎬 SRT", to_srt(final),
                           file_name="protocol.srt", mime="text/plain",
                           use_container_width=True)
    with d3:
        st.download_button("📊 JSON", to_json_str(final),
                           file_name="protocol.json", mime="application/json",
                           use_container_width=True)

    # ── Статистика ───────────────────────────────────────────────────
    with st.expander("📈 Статистика спикеров"):
        for spk in final_speakers:
            items = [r for r in final if r["speaker"] == spk]
            tot_dur = sum(r["end"] - r["start"] for r in items)
            tot_words = sum(len(r["text"].split()) for r in items)
            c = final_colors.get(spk, "#999")
            st.markdown(
                f'<span style="color:{c};font-weight:700;">{spk}</span>: '
                f'{len(items)} реплик · {fmt_time(tot_dur)} · ~{tot_words} слов',
                unsafe_allow_html=True,
            )
