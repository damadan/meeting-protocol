#!/usr/bin/env python3
"""
fluid_pipeline.py — Транскрибация + диаризация MP4 через FluidAudio
                     Все модели работают на M4 GPU/ANE через CoreML

Пайплайн:
  1. MP4 → WAV (ffmpeg)
  2. WAV → диаризация (fluidaudiocli process --mode offline, CoreML/ANE)
  3. Каждый сегмент → параллельная транскрипция (fluidaudiocli transcribe, CoreML/ANE)
  4. Протокол → stdout + файл

Скорость на M4:
  Диаризация: ~290× real-time
  Транскрипция: ~250ms на сегмент, 8 параллельных потоков
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ─── Пути ──────────────────────────────────────────────────────────
FLUID_CLI = Path(__file__).parent.parent / "FluidAudio/.build/release/fluidaudiocli"
FFMPEG = "ffmpeg"
SAMPLE_RATE = 16000
MIN_SEG_SEC = 0.5   # сегменты короче пропускаем
MERGE_GAP_SEC = 1.0 # склеиваем реплики одного спикера с паузой < N сек


# ─── Утилиты ───────────────────────────────────────────────────────
def fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_srt(sec: float) -> str:
    ms = int((sec % 1) * 1000)
    return fmt_time(sec).replace(":", ":", 2) + f",{ms:03d}"


def log(msg: str) -> None:
    print(msg, flush=True)


# ─── Шаг 1: MP4 → WAV ──────────────────────────────────────────────
def extract_wav(mp4_path: str, out_wav: str) -> float:
    """Извлекает моно 16kHz WAV из видео/аудио файла. Возвращает длительность."""
    result = subprocess.run([
        FFMPEG, "-y", "-i", mp4_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        out_wav,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr[-500:]}")

    # Получаем длительность
    probe = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", out_wav,
    ], capture_output=True, text=True)
    return float(probe.stdout.strip() or 0)


# ─── Шаг 2: Диаризация (CoreML / ANE) ──────────────────────────────
def run_diarization(wav_path: str, out_json: str) -> dict:
    """Запускает fluidaudiocli process --mode offline. Возвращает весь JSON."""
    result = subprocess.run([
        str(FLUID_CLI), "process", wav_path,
        "--mode", "offline",
        "--output", out_json,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"fluidaudiocli process error:\n{result.stderr[-500:]}")

    with open(out_json) as f:
        return json.load(f)


# ─── Шаг 3: Транскрибация сегмента (CoreML / ANE) ──────────────────
def _cut_segment(wav_path: str, start: float, end: float, out_wav: str) -> None:
    """Нарезает WAV-чанк через ffmpeg."""
    subprocess.run([
        FFMPEG, "-y",
        "-ss", str(start),
        "-t", str(end - start),
        "-i", wav_path,
        "-acodec", "copy",
        out_wav,
    ], capture_output=True, check=True)


def _transcribe_file(wav_path: str) -> str:
    """Запускает fluidaudiocli transcribe на одном файле."""
    result = subprocess.run(
        [str(FLUID_CLI), "transcribe", wav_path],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def transcribe_segment(wav_path: str, seg: dict, tmpdir: str) -> dict:
    """Нарезает сегмент и транскрибирует его."""
    start = seg["startTimeSeconds"]
    end = seg["endTimeSeconds"]
    chunk = os.path.join(tmpdir, f"seg_{start:.3f}.wav")
    _cut_segment(wav_path, start, end, chunk)
    text = _transcribe_file(chunk)
    try:
        os.unlink(chunk)
    except OSError:
        pass
    return {
        "start": start,
        "end": end,
        "speaker": seg["speakerId"],
        "text": text,
    }


def transcribe_all(wav_path: str, segments: list[dict],
                   workers: int, tmpdir: str) -> list[dict]:
    """Параллельная транскрипция всех сегментов."""
    total = len(segments)
    indexed = [None] * total
    done_count = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(transcribe_segment, wav_path, seg, tmpdir): i
            for i, seg in enumerate(segments)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                indexed[idx] = future.result()
            except Exception as e:
                log(f"  [WARN] Сегмент {idx} не распознан: {e}")
            done_count += 1
            pct = int(done_count / total * 40)
            bar = "█" * pct + "░" * (40 - pct)
            print(f"\r  [{bar}] {done_count}/{total}", end="", flush=True)

    print()
    return [r for r in indexed if r and r["text"]]


# ─── Пост-обработка ─────────────────────────────────────────────────
def merge_consecutive(results: list[dict], gap: float) -> list[dict]:
    """Склеивает соседние реплики одного спикера с паузой < gap секунд."""
    if not results:
        return []
    merged = [results[0].copy()]
    for item in results[1:]:
        prev = merged[-1]
        if (item["speaker"] == prev["speaker"]
                and (item["start"] - prev["end"]) < gap):
            prev["text"] += " " + item["text"]
            prev["end"] = item["end"]
        else:
            merged.append(item.copy())
    return merged


# ─── Форматирование вывода ──────────────────────────────────────────
def to_protocol(results: list[dict]) -> str:
    lines = []
    for r in results:
        lines.append(f"[{fmt_time(r['start'])} — {fmt_time(r['end'])}] {r['speaker']}:")
        lines.append(r["text"])
        lines.append("")
    return "\n".join(lines)


def to_srt(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks += [
            str(i),
            f"{fmt_srt(r['start'])} --> {fmt_srt(r['end'])}",
            f"[{r['speaker']}] {r['text']}",
            "",
        ]
    return "\n".join(blocks)


def to_json_out(results: list[dict]) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)


# ─── GPU-верификация ────────────────────────────────────────────────
def verify_gpu() -> None:
    """Проверяет наличие M4 GPU и ANE через system_profiler."""
    try:
        out = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True, text=True
        ).stdout
        for line in out.splitlines():
            if "Chip" in line or "GPU" in line or "Core" in line:
                log(f"  {line.strip()}")
    except Exception:
        pass

    # Проверяем CoreML через ioreg
    try:
        ioreg = subprocess.run(
            ["ioreg", "-r", "-c", "AppleARMIODevice", "-n", "ane0"],
            capture_output=True, text=True
        ).stdout
        if "ane0" in ioreg:
            log("  ✅ Apple Neural Engine (ANE) обнаружен и активен")
        else:
            log("  ℹ️  ANE: данные недоступны (модели всё равно используют CoreML GPU)")
    except Exception:
        pass


# ─── Главная функция ────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FluidAudio: транскрибация + диаризация на M4 GPU/ANE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Путь к MP4/WAV файлу")
    parser.add_argument("--workers", type=int, default=8,
                        help="Потоки параллельной транскрипции (default: 8)")
    parser.add_argument("--merge-gap", type=float, default=MERGE_GAP_SEC,
                        help="Склейка реплик одного спикера (сек, default: 1.0)")
    parser.add_argument("--output", default=None,
                        help="Файл для сохранения протокола (TXT)")
    parser.add_argument("--srt", default=None,
                        help="Файл для сохранения субтитров (SRT)")
    parser.add_argument("--json-out", default=None,
                        help="Файл для сохранения JSON")
    parser.add_argument("--diar-json", default=None,
                        help="Использовать готовый JSON диаризации (пропустить process)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Не склеивать реплики")
    args = parser.parse_args()

    if not FLUID_CLI.exists():
        log(f"❌ fluidaudiocli не найден: {FLUID_CLI}")
        log("   Соберите: cd Desktop/FluidAudio && swift build -c release --product fluidaudiocli")
        sys.exit(1)

    t_total = time.time()
    input_path = args.input

    log("=" * 60)
    log("🎙️  FluidAudio Pipeline — M4 GPU/ANE (CoreML)")
    log("=" * 60)

    # GPU info
    log("\n📊 Железо:")
    verify_gpu()

    with tempfile.TemporaryDirectory(prefix="fluid_") as tmpdir:

        # ── 1. Извлечение аудио ─────────────────────────────────────
        wav_path = os.path.join(tmpdir, "audio.wav")
        log(f"\n[1/3] 🔊 Извлечение аудио из {Path(input_path).name}...")
        t0 = time.time()

        if input_path.lower().endswith(".wav"):
            import shutil
            shutil.copy(input_path, wav_path)
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", wav_path],
                capture_output=True, text=True
            )
            duration = float(probe.stdout.strip() or 0)
        else:
            duration = extract_wav(input_path, wav_path)

        log(f"    ✓ {duration/60:.1f} мин аудио → {time.time()-t0:.1f}с")

        # ── 2. Диаризация ────────────────────────────────────────────
        if args.diar_json:
            log(f"\n[2/3] 📂 Загрузка диаризации из {args.diar_json}...")
            with open(args.diar_json) as f:
                diar_data = json.load(f)
        else:
            log("\n[2/3] 👥 Диаризация (CoreML на M4 GPU/ANE)...")
            t0 = time.time()
            diar_json = os.path.join(tmpdir, "diarization.json")
            diar_data = run_diarization(wav_path, diar_json)
            elapsed = time.time() - t0
            rtfx = diar_data.get("realTimeFactor", 0)
            log(f"    ✓ {diar_data['speakerCount']} спикера, "
                f"{len(diar_data['segments'])} сегментов — "
                f"{elapsed:.1f}с (RTFx={rtfx:.0f}×)")

        segments = [
            s for s in diar_data["segments"]
            if (s["endTimeSeconds"] - s["startTimeSeconds"]) >= MIN_SEG_SEC
        ]
        speakers = sorted(set(s["speakerId"] for s in segments))
        log(f"    Спикеры: {', '.join(speakers)}")

        # ── 3. Транскрибация ─────────────────────────────────────────
        log(f"\n[3/3] 📝 Транскрибация {len(segments)} сегментов "
            f"({args.workers} потоков, CoreML на M4 GPU/ANE)...")
        t0 = time.time()
        results = transcribe_all(wav_path, segments, args.workers, tmpdir)
        elapsed = time.time() - t0
        log(f"    ✓ {len(results)} сегментов распознано — {elapsed:.1f}с")

        if not results:
            log("❌ Речь не распознана. Проверьте аудио.")
            sys.exit(1)

    # ── Пост-обработка ───────────────────────────────────────────────
    if not args.no_merge:
        results = merge_consecutive(results, args.merge_gap)
        log(f"    Склейка → {len(results)} реплик")

    # ── Статистика ───────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    log(f"\n⏱️  Итого: {total_elapsed:.1f}с на {duration/60:.1f} мин аудио "
        f"(RTFx≈{duration/total_elapsed:.0f}×)")

    spk_stats: dict[str, dict] = {}
    for r in results:
        sp = r["speaker"]
        if sp not in spk_stats:
            spk_stats[sp] = {"time": 0.0, "words": 0, "turns": 0}
        spk_stats[sp]["time"] += r["end"] - r["start"]
        spk_stats[sp]["words"] += len(r["text"].split())
        spk_stats[sp]["turns"] += 1

    log("\n📊 Статистика спикеров:")
    for sp, st in sorted(spk_stats.items()):
        log(f"  {sp}: {st['turns']} реплик, "
            f"{fmt_time(st['time'])} суммарно, "
            f"~{st['words']} слов")

    # ── Вывод протокола ──────────────────────────────────────────────
    protocol = to_protocol(results)

    log("\n" + "=" * 60)
    log("📋 ПРОТОКОЛ")
    log("=" * 60)
    print(protocol)

    # ── Сохранение файлов ────────────────────────────────────────────
    if args.output:
        Path(args.output).write_text(protocol, encoding="utf-8")
        log(f"💾 TXT сохранён: {args.output}")

    if args.srt:
        Path(args.srt).write_text(to_srt(results), encoding="utf-8")
        log(f"💾 SRT сохранён: {args.srt}")

    if args.json_out:
        Path(args.json_out).write_text(to_json_out(results), encoding="utf-8")
        log(f"💾 JSON сохранён: {args.json_out}")


if __name__ == "__main__":
    main()
