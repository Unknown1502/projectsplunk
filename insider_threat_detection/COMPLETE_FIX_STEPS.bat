@echo off
echo ========================================
echo COMPLETE SPLUNK INTEGRATION FIX
echo ========================================
echo.

echo STEP 1: Fix HEC Token Issue
echo ----------------------------
echo The HEC token in your config is invalid (403 error)
echo Run: python fix_hec_token.py
echo.
echo This will:
echo - Create a new valid HEC token
echo - Update splunk_credentials.json
echo - Test the connection
echo.
pause

echo.
echo STEP 2: Copy credentials to Splunk app folder
echo ----------------------------------------------
echo Since the project is already moved, copy the updated credentials:
echo.
copy splunk_credentials.json "C:\Program Files\Splunk\etc\apps\insider_threat_detection\" /Y
echo.
echo Credentials copied!
echo.
pause

echo.
echo STEP 3: Navigate to Splunk app folder
echo -------------------------------------
echo cd "C:\Program Files\Splunk\etc\apps\insider_threat_detection"
echo.
pause

echo.
echo STEP 4: Test the integration
echo ----------------------------
echo From the Splunk app folder, run:
echo python automated_splunk_integration_fixed.py
echo.
echo Or test HEC directly:
echo python working_hec_sender.py
echo.
pause

echo.
echo ========================================
echo TROUBLESHOOTING
echo ========================================
echo.
echo If you still get errors:
echo.
echo 1. Make sure Splunk is running:
echo    "C:\Program Files\Splunk\bin\splunk.exe" status
echo.
echo 2. Check HEC is enabled in Splunk Web:
echo    Settings -> Data Inputs -> HTTP Event Collector
echo    Global Settings -> All Tokens -> Enabled
echo.
echo 3. Verify the new token in splunk_credentials.json
echo.
echo 4. Check Windows Firewall isn't blocking port 8088
echo.
pause
