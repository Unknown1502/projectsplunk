@echo off
echo ========================================
echo MOVING PROJECT TO SPLUNK FOLDER
echo ========================================
echo.

set SPLUNK_APPS="C:\Program Files\Splunk\etc\apps"
set PROJECT_NAME=insider_threat_detection

echo Moving project from:
echo %cd%
echo.
echo To Splunk apps folder:
echo %SPLUNK_APPS%\%PROJECT_NAME%
echo.

echo [1] Creating backup...
xcopy /E /I /Y . "%SPLUNK_APPS%\%PROJECT_NAME%_backup" > nul 2>&1

echo [2] Copying project to Splunk apps...
xcopy /E /I /Y . "%SPLUNK_APPS%\%PROJECT_NAME%" > nul 2>&1

echo [3] Creating proper app structure...
cd /d "%SPLUNK_APPS%\%PROJECT_NAME%"

:: Create required Splunk app directories
mkdir default 2>nul
mkdir local 2>nul
mkdir bin 2>nul
mkdir lookups 2>nul

:: Create app.conf
echo [install] > default\app.conf
echo is_configured = true >> default\app.conf
echo state = enabled >> default\app.conf
echo. >> default\app.conf
echo [ui] >> default\app.conf
echo is_visible = true >> default\app.conf
echo label = Insider Threat Detection >> default\app.conf
echo. >> default\app.conf
echo [launcher] >> default\app.conf
echo author = Security Team >> default\app.conf
echo description = ML-based insider threat detection >> default\app.conf
echo version = 1.0.0 >> default\app.conf

:: Create inputs.conf for HEC
echo [http://insider_threat] > default\inputs.conf
echo disabled = 0 >> default\inputs.conf
echo index = main >> default\inputs.conf
echo indexes = main >> default\inputs.conf
echo sourcetype = insider_threat >> default\inputs.conf
echo token = %HEC_TOKEN% >> default\inputs.conf

:: Update paths in Python files
echo [4] Updating paths in configuration files...
powershell -Command "(Get-Content 'splunk_credentials.json') -replace 'C:\\Program Files\\Splunk\\etc\\apps\\projectsplunk', '%SPLUNK_APPS:\=\\%\\%PROJECT_NAME%' | Set-Content 'splunk_credentials.json'"

echo.
echo ========================================
echo PROJECT MOVED SUCCESSFULLY!
echo ========================================
echo.
echo Project is now at: %SPLUNK_APPS%\%PROJECT_NAME%
echo.
echo Next steps:
echo 1. Restart Splunk:
echo    "%PROGRAMFILES%\Splunk\bin\splunk.exe" restart
echo.
echo 2. Access the app in Splunk Web:
echo    http://localhost:8000/en-US/app/insider_threat_detection
echo.
echo 3. Run scripts from new location:
echo    cd "%SPLUNK_APPS%\%PROJECT_NAME%"
echo    python automated_splunk_integration_fixed.py
echo.
pause
