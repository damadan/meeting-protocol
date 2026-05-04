#!/usr/bin/env bash
set -euo pipefail

echo "══════════════════════════════════════════════"
echo "  Установка Speech Protocol Pipeline (ONNX)"
echo "══════════════════════════════════════════════"

# ─── 1. Проверка ffmpeg ─────────────────────
echo ""
echo "[1/5] Проверка системных зависимостей..."

if ! command -v ffmpeg &> /dev/null; then
    echo "⚠  ffmpeg не найден. Установите:"
    echo "   Ubuntu/Debian: sudo apt install ffmpeg"
    echo "   macOS:         brew install ffmpeg"
    exit 1
fi
echo "  ✓ ffmpeg найден"
echo "  ✓ Python: $(python3 --version)"

# ─── 2. Виртуальное окружение ────────────────
echo ""
echo "[2/5] Создание виртуального окружения..."

if command -v conda &> /dev/null; then
    conda create -y --name speech-protocol python=3.10
    eval "$(conda shell.bash hook)"
    conda activate speech-protocol
    echo "  ✓ Conda-окружение speech-protocol"
else
    python3 -m venv .venv
    source .venv/bin/activate
    echo "  ✓ venv в .venv/"
fi

# ─── 3. PyTorch (CPU) ───────────────────────
echo ""
echo "[3/5] Установка PyTorch (CPU)..."

pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
echo "  ✓ PyTorch CPU"

# ─── 4. DiariZen ────────────────────────────
echo ""
echo "[4/5] Установка DiariZen..."

if [ ! -d "DiariZen" ]; then
    git clone https://github.com/BUTSpeechFIT/DiariZen.git
fi
cd DiariZen
pip install -r requirements.txt
pip install -e .
cd pyannote-audio && pip install -e . && cd ..
cd ..
echo "  ✓ DiariZen"

# ─── 5. ONNX ASR + Streamlit ───────────────
echo ""
echo "[5/5] Установка onnx-asr и Streamlit..."

pip install onnx-asr streamlit soundfile
echo "  ✓ onnx-asr, Streamlit, soundfile"

# ─── Готово ─────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  ✅ Установка завершена!"
echo ""
echo "  Запуск:"
echo "    streamlit run app.py"
echo ""
echo "  При первом запуске ONNX-модель GigaAM"
echo "  скачается с HuggingFace (~500 МБ)."
echo "══════════════════════════════════════════════"
