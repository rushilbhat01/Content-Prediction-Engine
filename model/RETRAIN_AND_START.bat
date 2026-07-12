@echo off
echo ============================================
echo  Shorts Predictor — Full Retrain Pipeline
echo ============================================
echo.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo [1/6] Installing hook feature dependencies...
pip install easyocr soundfile -q
echo Done.
echo.

echo [2/6] Rebuilding niche-specific train scripts...
python _fix_arrows.py
if %errorlevel% neq 0 ( echo FAILED & pause & exit /b 1 )
echo Done.
echo.

echo [3/6] Extracting hook features for all videos...
echo       (text overlay, audio hook type, Whisper speech - ~2-3hrs)
echo       Tip: add --no-ocr to skip OCR and go 3x faster
python run_hook_extraction.py --resume
if %errorlevel% neq 0 ( echo WARNING: hook extraction had errors, continuing... )
echo Done.
echo.

echo [4/6] Training FITNESS model...
python model_train_fitness.py
if %errorlevel% neq 0 ( echo FAILED & pause & exit /b 1 )
echo Done.
echo.

echo [5/6] Training FOOD model...
python model_train_food.py
if %errorlevel% neq 0 ( echo FAILED & pause & exit /b 1 )
echo Done.
echo.

echo [6/6] Starting API server...
echo  Web UI: http://localhost:8000
echo  Press Ctrl+C to stop.
echo.
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

pause
