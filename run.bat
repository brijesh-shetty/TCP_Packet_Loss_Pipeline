@echo off
echo ================================================================
echo   TCP Packet Loss Prediction — Full Pipeline
echo   Based on: "Real-time Prediction of TCP Packet Loss using ML"
echo   IEEE Ref: 10738808
echo ================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
echo.

REM Run full pipeline
echo [2/3] Parsing raw data...
echo [3/3] Training models with Optuna optimization...
echo.
python src/main.py --all --trials 30
echo.

echo ================================================================
echo   DONE! Check enhanced_results.txt for detailed results.
echo ================================================================
pause
