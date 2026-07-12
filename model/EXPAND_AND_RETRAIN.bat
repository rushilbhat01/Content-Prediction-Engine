@echo off
echo ================================================
echo  Shorts Predictor - Full Expand + Retrain
echo ================================================
echo.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo [1/8] Downloading missing videos (all niches)...
echo       (uses Chrome browser cookies - keep Chrome installed and logged into YouTube)
python script.py --download-missing
echo Done.
echo.

echo [2/8] Extracting visual features (new videos only)...
python extract_visual_fast.py
echo Done.
echo.

echo [3/8] Extracting audio features (new videos only)...
python extract_audio.py
echo Done.
echo.

echo [4/8] Extracting embeddings (new videos only - slowest step ~2-4hrs)...
python extract_embeddings.py
echo Done.
echo.

echo [5/8] Extracting hook features (new videos only)...
python run_hook_extraction.py --resume
echo Done.
echo.

echo [6/8] Rebuilding niche train scripts...
python _fix_arrows.py
echo Done.
echo.

echo [7/8] Training FITNESS model...
python model_train_fitness.py
if %errorlevel% neq 0 ( echo FAILED & pause & exit /b 1 )
echo Done.
echo.

echo [8/8] Training FOOD model...
python model_train_food.py
if %errorlevel% neq 0 ( echo FAILED & pause & exit /b 1 )
echo Done.
echo.

echo ================================================
echo  All done! Starting API server...
echo  Web UI: http://localhost:8000
echo ================================================
echo.
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

pause
