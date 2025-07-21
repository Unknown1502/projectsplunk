@echo off
echo Deploying Insider Threat Detection App to Splunk...
xcopy /E /I /Y "insider_threat_detection_app" "C:\Program Files\Splunk\etc\apps\insider_threat_detection_app"
echo Deployment complete!
pause
