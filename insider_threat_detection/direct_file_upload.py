#!/usr/bin/env python3
"""
Direct File Upload to Splunk - Alternative Method
Since Splunk Extension isn't showing, use these alternative methods
"""

import os
import shutil
from datetime import datetime

def create_splunk_ready_csv():
    """Create CSV file that can be directly uploaded to Splunk"""
    
    print("Creating CSV for direct Splunk upload...")
    
    # Sample threat predictions
    csv_content = """timestamp,user,pc,activity_type,threat_score,risk_category,is_threat
2024-01-01 10:00:00,john.doe,PC-001,HTTP,0.85,high,true
2024-01-01 10:05:00,jane.smith,PC-002,FILE,0.30,low,false
2024-01-01 10:10:00,bob.jones,PC-003,LOGON,0.65,medium,true
2024-01-01 10:15:00,alice.brown,PC-004,EMAIL,0.95,high,true
2024-01-01 10:20:00,charlie.wilson,PC-005,HTTP,0.15,low,false"""
    
    with open("threat_predictions.csv", "w") as f:
        f.write(csv_content)
    
    print("[SUCCESS] Created threat_predictions.csv")
    return "threat_predictions.csv"

def create_splunk_monitor_input():
    """Create a file in Splunk's monitored directory"""
    
    splunk_path = r"C:\Program Files\Splunk\etc\apps\search\lookups"
    
    print(f"\nOption 1: Copy to Splunk's lookup directory")
    print(f"Target: {splunk_path}")
    
    if os.path.exists(splunk_path):
        try:
            csv_file = create_splunk_ready_csv()
            dest = os.path.join(splunk_path, "insider_threats.csv")
            shutil.copy(csv_file, dest)
            print(f"[SUCCESS] Copied to: {dest}")
            print("\nAccess in Splunk:")
            print("| inputlookup insider_threats.csv")
            return True
        except Exception as e:
            print(f"[ERROR] Copy failed: {e}")
            print("Try running as Administrator")
    else:
        print("[ERROR] Splunk lookup directory not found")
    
    return False

def create_upload_instructions():
    """Create instructions for manual upload"""
    
    instructions = """
# MANUAL SPLUNK UPLOAD INSTRUCTIONS

Since the Splunk Extension isn't available, here are 3 ways to get your data into Splunk:

## Method 1: Splunk Web Upload (EASIEST)
1. Open Splunk Web: http://localhost:8000
2. Go to: Settings -> Add Data -> Upload
3. Select "threat_predictions.csv"
4. Follow the wizard:
   - Set Source type: csv
   - Review settings
   - Submit
5. Search: index=main source="threat_predictions.csv"

## Method 2: Copy to Splunk Directory
1. Run this script as Administrator
2. It will copy the CSV to Splunk's lookup directory
3. In Splunk, search: | inputlookup insider_threats.csv

## Method 3: Monitor Directory
1. Create folder: C:\\SplunkData
2. Copy threat_predictions.csv there
3. In Splunk: Settings -> Data Inputs -> Files & Directories
4. Add C:\\SplunkData as monitored directory
5. Splunk will automatically index new files

## Method 4: Use Existing HEC (if fixed)
After fixing HEC configuration:
```python
import requests
url = "http://localhost:8088/services/collector/event"
headers = {"Authorization": "Splunk YOUR_TOKEN"}
data = {"event": {"user": "test", "threat_score": 0.8}}
requests.post(url, json=data, headers=headers)
```

Choose the method that works best for your setup!
"""
    
    with open("SPLUNK_UPLOAD_INSTRUCTIONS.txt", "w", encoding='utf-8') as f:
        f.write(instructions)
    
    print("\n[INFO] Created SPLUNK_UPLOAD_INSTRUCTIONS.txt")
    print("Read this file for detailed upload methods")

def main():
    print("SPLUNK DATA UPLOAD HELPER")
    print("=" * 40)
    
    # Create CSV file
    csv_file = create_splunk_ready_csv()
    
    # Try automatic copy
    print("\nAttempting automatic copy to Splunk...")
    if not create_splunk_monitor_input():
        print("\nUse manual upload instead:")
        print("1. Open http://localhost:8000")
        print("2. Settings -> Add Data -> Upload")
        print(f"3. Select: {os.path.abspath(csv_file)}")
    
    # Create instructions
    create_upload_instructions()
    
    print("\n[COMPLETE] Your options:")
    print("1. Upload threat_predictions.csv via Splunk Web")
    print("2. Read SPLUNK_UPLOAD_INSTRUCTIONS.txt for more methods")
    print("3. The CSV is ready in standard format for any upload method")

if __name__ == "__main__":
    main()
