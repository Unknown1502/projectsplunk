@echo off
REM ========================================
REM Insider Threat Detection - One Command Execution
REM ========================================

echo.
echo ====================================================
echo INSIDER THREAT DETECTION - COMPLETE PROJECT EXECUTION
echo ====================================================
echo.

REM Check if we're in the right directory
if not exist "master_execution.py" (
    echo ERROR: Please run this script from the insider_threat_detection directory
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "..\projectenv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\projectenv\Scripts\activate.bat
)

REM Check if model already exists
if exist "..\r1\checkpoints\model_checkpoint.h5" (
    echo.
    echo ✓ Trained model found! Skipping training phase.
    echo.
    
    REM Run Splunk integration directly
    echo Starting Splunk Integration...
    python master_execution.py
    
) else (
    echo.
    echo ! No trained model found. Starting full pipeline...
    echo.
    
    REM Run training first
    echo Phase 1: Training Model...
    python scripts\train.py
    
    if errorlevel 1 (
        echo.
        echo ERROR: Training failed!
        pause
        exit /b 1
    )
    
    echo.
    echo Phase 2: Splunk Integration...
    python master_execution.py
)

echo.
echo ====================================================
echo EXECUTION COMPLETE!
echo ====================================================
echo.
echo Next steps:
echo 1. Access Splunk at http://localhost:8000
echo 2. Use the custom search command: ^| insiderthreatpredict
echo 3. Import the dashboard from splunk_integration/dashboards/
echo.
pause
