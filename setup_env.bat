@echo off
REM =============================================================
REM setup_env.bat — Windows one-shot environment bootstrap
REM Run once after cloning the repo.
REM Usage:  setup_env.bat
REM =============================================================

echo =============================================
echo  AI Smart Traffic Management System - Setup
echo =============================================

REM 1. Create virtual environment
echo [1/5] Creating virtual environment...
python -m venv .venv

REM 2. Activate
echo [2/5] Activating virtual environment...
call .venv\Scripts\activate.bat

REM 3. Upgrade pip
echo [3/5] Upgrading pip...
pip install --upgrade pip setuptools wheel

REM 4. Install dependencies
echo [4/5] Installing dependencies...
pip install -r requirements.txt

REM 5. Download YOLO weights
echo [5/5] Downloading YOLOv8n weights...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo.
echo Setup complete!
echo Next: python main.py --help
pause
