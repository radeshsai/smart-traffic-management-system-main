#!/usr/bin/env bash
# =============================================================
# setup_env.sh — One-shot environment bootstrap script
# Run once after cloning the repo.
# Usage:  bash setup_env.sh
# =============================================================

set -e  # Exit on first error

PYTHON=python3.11
VENV_DIR=".venv"

echo "============================================="
echo " AI Smart Traffic Management System — Setup"
echo "============================================="

# 1. Check Python version
echo "[1/6] Checking Python version..."
$PYTHON --version || { echo "❌ Python 3.11+ not found. Install it first."; exit 1; }

# 2. Create virtual environment
echo "[2/6] Creating virtual environment in $VENV_DIR ..."
$PYTHON -m venv $VENV_DIR

# 3. Activate venv
echo "[3/6] Activating virtual environment..."
source $VENV_DIR/bin/activate

# 4. Upgrade pip
echo "[4/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 5. Install all dependencies
echo "[5/6] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 6. Download YOLOv8n model weights if missing
echo "[6/6] Checking YOLOv8n model weights..."
MODEL_PATH="models/yolov8n.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "  → Downloading yolov8n.pt ..."
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    cp ~/.config/Ultralytics/yolov8n.pt $MODEL_PATH 2>/dev/null || \
    cp yolov8n.pt $MODEL_PATH 2>/dev/null || \
    echo "  ⚠ Copy manually: move yolov8n.pt → models/yolov8n.pt"
else
    echo "  → yolov8n.pt already present."
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. source $VENV_DIR/bin/activate"
echo "  2. Place your 4 traffic videos in data/input/"
echo "  3. python main.py --help"
echo "  4. streamlit run dashboard/streamlit_app.py"
