#!/usr/bin/env python3
"""
Fix HEC Token Issue - Creates new token and updates configuration
"""

import splunklib.client as client
import json
import time
import os

def create_new_hec_token():
    """Create a new HEC token in Splunk"""
    print("FIXING HEC TOKEN ISSUE")
    print("=" * 50)
    
    # Connect to Splunk
    try:
        service = client.connect(
            host="localhost",
            port=8089,
            username="admin",
            password="Sharvil@123",
            scheme="https"
        )
        print("[SUCCESS] Connected to Splunk")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Splunk: {e}")
        print("Make sure Splunk is running and credentials are correct")
        return None
    
    # Enable HEC if not enabled
    settings = service.settings
    hec_settings = settings['http_input']
    
    if hec_settings['disabled'] != '0':
        print("[INFO] Enabling HEC...")
        hec_settings.update(disabled=False)
        time.sleep(2)
    
    # Get HEC tokens
    hec_tokens = service.inputs.http
    
    # Create new token
    token_name = "insider_threat_detection_token"
    
    # Remove old token if exists
    for token in hec_tokens:
        if token.name == token_name:
            print(f"[INFO] Removing old token: {token.name}")
            token.delete()
            time.sleep(1)
    
    # Create new token
    print("[INFO] Creating new HEC token...")
    try:
        new_token = hec_tokens.create(
            name=token_name,
            index="main",
            sourcetype="insider_threat",
            disabled=False
        )
        
        token_value = new_token.token
        print(f"[SUCCESS] New HEC token created: {token_value}")
        
        return token_value
        
    except Exception as e:
        print(f"[ERROR] Failed to create token: {e}")
        return None

def update_credentials_files(new_token):
    """Update all credential files with new token"""
    print("\n[INFO] Updating configuration files...")
    
    # Update splunk_credentials.json in current directory
    creds_file = "splunk_credentials.json"
    if os.path.exists(creds_file):
        with open(creds_file, 'r') as f:
            config = json.load(f)
        
        config['hec_token'] = new_token
        
        with open(creds_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[SUCCESS] Updated {creds_file}")
    
    # Also update in Splunk apps directory if it exists
    splunk_app_creds = r"C:\Program Files\Splunk\etc\apps\insider_threat_detection\splunk_credentials.json"
    if os.path.exists(splunk_app_creds):
        with open(splunk_app_creds, 'r') as f:
            config = json.load(f)
        
        config['hec_token'] = new_token
        
        with open(splunk_app_creds, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[SUCCESS] Updated {splunk_app_creds}")

def test_new_token(token):
    """Test the new HEC token"""
    import requests
    import urllib3
    urllib3.disable_warnings()
    
    print("\n[TEST] Testing new HEC token...")
    
    url = "http://localhost:8088/services/collector/event"
    headers = {"Authorization": f"Splunk {token}"}
    
    test_event = {
        "event": {
            "message": "HEC token test successful",
            "timestamp": time.time(),
            "source": "fix_script"
        },
        "index": "main",
        "sourcetype": "insider_threat"
    }
    
    try:
        response = requests.post(url, json=test_event, headers=headers, verify=False, timeout=5)
        
        if response.status_code == 200:
            print("[SUCCESS] HEC token is working!")
            return True
        else:
            print(f"[ERROR] Test failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

def main():
    # Create new token
    new_token = create_new_hec_token()
    
    if new_token:
        # Update configuration files
        update_credentials_files(new_token)
        
        # Test the new token
        if test_new_token(new_token):
            print("\n" + "=" * 50)
            print("HEC TOKEN FIXED SUCCESSFULLY!")
            print("=" * 50)
            print(f"\nNew HEC Token: {new_token}")
            print("\nYou can now run:")
            print("1. python working_hec_sender.py")
            print("2. python automated_splunk_integration_fixed.py")
        else:
            print("\n[WARNING] Token created but test failed")
            print("Try restarting Splunk and running again")
    else:
        print("\n[ERROR] Failed to create new HEC token")
        print("Please check Splunk is running and try again")

if __name__ == "__main__":
    main()
