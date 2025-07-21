# ========================================
# Insider Threat Detection - One Command Execution (PowerShell)
# ========================================

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "INSIDER THREAT DETECTION - COMPLETE PROJECT EXECUTION" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "master_execution.py")) {
    Write-Host "ERROR: Please run this script from the insider_threat_detection directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment if it exists
if (Test-Path "..\projectenv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ..\projectenv\Scripts\Activate.ps1
}

# Check if model already exists
if (Test-Path "..\r1\checkpoints\model_checkpoint.h5") {
    Write-Host ""
    Write-Host "✓ Trained model found! Skipping training phase." -ForegroundColor Green
    Write-Host ""
    
    # Run Splunk integration directly
    Write-Host "Starting Splunk Integration..." -ForegroundColor Yellow
    python master_execution.py
    
} else {
    Write-Host ""
    Write-Host "! No trained model found. Starting full pipeline..." -ForegroundColor Yellow
    Write-Host ""
    
    # Run training first
    Write-Host "Phase 1: Training Model..." -ForegroundColor Yellow
    python scripts\train.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Training failed!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-Host ""
    Write-Host "Phase 2: Splunk Integration..." -ForegroundColor Yellow
    python master_execution.py
}

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host "EXECUTION COMPLETE!" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Access Splunk at http://localhost:8000"
Write-Host "2. Use the custom search command: | insiderthreatpredict"
Write-Host "3. Import the dashboard from splunk_integration/dashboards/"
Write-Host ""
Read-Host "Press Enter to exit"
