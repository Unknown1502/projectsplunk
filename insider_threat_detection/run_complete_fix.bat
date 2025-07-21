@echo off
echo ========================================
echo COMPLETE SPLUNK INTEGRATION FIX
echo ========================================
echo.

echo [1] Running quick HEC fix...
python quick_hec_fix.py

echo.
echo [2] Testing HEC connection...
python working_hec_sender.py

echo.
echo [3] Testing fixed integration...
python automated_splunk_integration_fixed.py

echo.
echo [4] Running comprehensive fix...
python fix_splunk_integration_complete.py

echo.
echo ========================================
echo FIX COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Check Splunk Web: http://localhost:8000
echo 2. Search: index=main sourcetype=insider_threat
echo 3. Run: python automated_splunk_integration_fixed.py
echo.
pause
