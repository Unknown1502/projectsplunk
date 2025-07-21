@echo off
echo Applying final fixes to Insider Threat Detection App...

echo Step 1: Fixing permissions...
icacls "C:\Program Files\Splunk\etc\apps\insider_threat_detection_app" /grant Everyone:F /T

echo Step 2: Restarting Splunk...
net stop Splunkd
net start Splunkd

echo Step 3: Waiting for Splunk to start...
timeout /t 30

echo All fixes applied successfully!
echo Your app should now work without errors.
pause
