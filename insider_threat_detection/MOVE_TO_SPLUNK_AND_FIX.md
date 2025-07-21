# Moving Project to Splunk and Fixing 400 Error

## Quick Solution

### Step 1: Move Project to Splunk Apps Folder
Run this command to move your project:
```batch
move_to_splunk.bat
```

This will:
- Copy your entire project to `C:\Program Files\Splunk\etc\apps\insider_threat_detection`
- Create proper Splunk app structure
- Update all paths automatically

### Step 2: Fix HEC 400 Error
The 400 error happens because:
1. HEC token mismatch
2. Wrong event format
3. HEC not properly configured

After moving, run from the new location:
```batch
cd "C:\Program Files\Splunk\etc\apps\insider_threat_detection"
python quick_hec_fix.py
```

### Step 3: Restart Splunk
```batch
"C:\Program Files\Splunk\bin\splunk.exe" restart
```

## Why Move to Splunk Folder?

Moving your project inside Splunk's apps folder provides:
1. **Direct Integration**: Splunk can directly access your app
2. **Proper Permissions**: No permission issues
3. **Easy Management**: Manage through Splunk Web UI
4. **Better Performance**: Direct access to models and scripts

## After Moving

1. **Access your app**:
   - Open: http://localhost:8000
   - Navigate to Apps → Insider Threat Detection

2. **Run scripts from new location**:
   ```batch
   cd "C:\Program Files\Splunk\etc\apps\insider_threat_detection"
   python automated_splunk_integration_fixed.py
   ```

3. **Check HEC**:
   - Settings → Data Inputs → HTTP Event Collector
   - Ensure token matches what's in `splunk_credentials.json`

## Fix 400 Error Permanently

1. **Update HEC token** in all scripts to use the one from `splunk_credentials.json`
2. **Use correct event format**:
   ```json
   {
     "event": {your_data},
     "index": "main",
     "sourcetype": "insider_threat"
   }
   ```

3. **Test HEC**:
   ```batch
   python working_hec_sender.py
   ```

## Complete Integration Test

After moving and fixing, run:
```batch
python test_splunk_integration.py
```

This will verify:
- HEC connection works
- Events are being sent
- No 400 errors

## Troubleshooting

If you still get 400 errors:
1. Check Splunk is running
2. Verify HEC is enabled
3. Confirm token in credentials file
4. Check firewall isn't blocking port 8088
