#!/usr/bin/env python3
"""
Complete Splunk Integration Fix
Fixes all integration issues including HEC 400 errors
"""

import os
import sys
import json
import shutil
import requests
import subprocess
from pathlib import Path
import time

class SplunkIntegrationFixer:
    def __init__(self):
        self.splunk_home = "C:\\Program Files\\Splunk"
        self.project_path = os.getcwd()
        self.app_name = "insider_threat_detection"
        self.credentials_file = "splunk_credentials.json"
        
        # Load current credentials
        self.load_credentials()
        
    def load_credentials(self):
        """Load Splunk credentials"""
        try:
            with open(self.credentials_file, 'r') as f:
                self.config = json.load(f)
        except:
            print("[ERROR] Could not load credentials file")
            self.config = {}
    
    def fix_hec_configuration(self):
        """Fix HEC configuration and 400 errors"""
        print("\n[1] FIXING HEC CONFIGURATION")
        print("=" * 50)
        
        # Create proper HEC setup script
        hec_fix_script = '''import splunklib.client as client
import json
import time

# Connect to Splunk
service = client.connect(
    host="localhost",
    port=8089,
    username="admin",
    password="Sharvil@123",
    scheme="https"
)

print("[INFO] Connected to Splunk")

# Check if HEC is enabled
settings = service.settings
hec_settings = settings['http_input']

if not hec_settings['disabled'] == '0':
    print("[INFO] Enabling HEC...")
    hec_settings.update(disabled=False)
    time.sleep(2)

# Create or update HEC token
token_name = "insider_threat_token"
hec_tokens = service.inputs.http

# Remove old token if exists
for token in hec_tokens:
    if token.name == token_name:
        print(f"[INFO] Removing old token: {token.name}")
        token.delete()
        time.sleep(1)

# Create new token with proper configuration
print("[INFO] Creating new HEC token...")
new_token = hec_tokens.create(
    name=token_name,
    index="main",
    sourcetype="insider_threat",
    disabled=False
)

# Get the token value
token_value = new_token.token
print(f"[SUCCESS] New HEC token created: {token_value}")

# Update credentials file
creds_file = "splunk_credentials.json"
with open(creds_file, 'r') as f:
    config = json.load(f)

config['hec_token'] = token_value
config['hec_port'] = 8088

with open(creds_file, 'w') as f:
    json.dump(config, f, indent=4)

print("[SUCCESS] Credentials updated")

# Test HEC endpoint
import requests
import urllib3
urllib3.disable_warnings()

test_url = "http://localhost:8088/services/collector/event"
headers = {"Authorization": f"Splunk {token_value}"}
test_event = {
    "event": {
        "message": "HEC test successful",
        "severity": "info"
    },
    "index": "main",
    "sourcetype": "insider_threat"
}

try:
    response = requests.post(test_url, json=test_event, headers=headers, verify=False)
    if response.status_code == 200:
        print("[SUCCESS] HEC test passed!")
    else:
        print(f"[WARNING] HEC test returned: {response.status_code} - {response.text}")
except Exception as e:
    print(f"[ERROR] HEC test failed: {e}")
'''
        
        # Save and run the fix script
        fix_script_path = "temp_hec_fix.py"
        with open(fix_script_path, 'w') as f:
            f.write(hec_fix_script)
        
        print("[INFO] Running HEC fix script...")
        try:
            result = subprocess.run([sys.executable, fix_script_path], capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"[ERROR] {result.stderr}")
        except Exception as e:
            print(f"[ERROR] Failed to run HEC fix: {e}")
        finally:
            if os.path.exists(fix_script_path):
                os.remove(fix_script_path)
    
    def update_integration_scripts(self):
        """Update all integration scripts with correct HEC token"""
        print("\n[2] UPDATING INTEGRATION SCRIPTS")
        print("=" * 50)
        
        # Reload credentials to get new HEC token
        self.load_credentials()
        hec_token = self.config.get('hec_token', '')
        
        # Update automated_splunk_integration.py
        integration_file = "automated_splunk_integration.py"
        if os.path.exists(integration_file):
            print(f"[INFO] Updating {integration_file}...")
            with open(integration_file, 'r') as f:
                content = f.read()
            
            # Replace old token with new one
            import re
            content = re.sub(
                r'self\.hec_token = "[^"]*"',
                f'self.hec_token = "{hec_token}"',
                content
            )
            
            with open(integration_file, 'w') as f:
                f.write(content)
            print("[SUCCESS] Updated integration script")
    
    def deploy_to_splunk_apps(self):
        """Deploy the app to Splunk apps directory"""
        print("\n[3] DEPLOYING TO SPLUNK APPS")
        print("=" * 50)
        
        splunk_app_path = os.path.join(self.splunk_home, "etc", "apps", self.app_name)
        
        # Create app structure
        print(f"[INFO] Creating app at: {splunk_app_path}")
        os.makedirs(splunk_app_path, exist_ok=True)
        
        # Copy essential directories
        dirs_to_copy = ['src', 'config', 'scripts']
        for dir_name in dirs_to_copy:
            src_dir = os.path.join(self.project_path, dir_name)
            if os.path.exists(src_dir):
                dst_dir = os.path.join(splunk_app_path, dir_name)
                print(f"[INFO] Copying {dir_name}...")
                if os.path.exists(dst_dir):
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
        
        # Copy model files
        model_dir = os.path.join(splunk_app_path, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy checkpoints
        checkpoint_src = os.path.join(self.project_path, "r1", "checkpoints")
        if os.path.exists(checkpoint_src):
            print("[INFO] Copying model checkpoints...")
            for file in os.listdir(checkpoint_src):
                src_file = os.path.join(checkpoint_src, file)
                dst_file = os.path.join(model_dir, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
        
        # Create app.conf
        app_conf_dir = os.path.join(splunk_app_path, "default")
        os.makedirs(app_conf_dir, exist_ok=True)
        
        app_conf_content = '''[install]
is_configured = true
state = enabled

[ui]
is_visible = true
label = Insider Threat Detection

[launcher]
author = Security Team
description = Real-time insider threat detection using ML
version = 1.0.0
'''
        
        with open(os.path.join(app_conf_dir, "app.conf"), 'w') as f:
            f.write(app_conf_content)
        
        print("[SUCCESS] App deployed to Splunk")
    
    def create_simple_hec_sender(self):
        """Create a simple HEC sender script"""
        print("\n[4] CREATING SIMPLE HEC SENDER")
        print("=" * 50)
        
        sender_script = '''#!/usr/bin/env python3
"""
Simple Splunk HEC Sender
Send events to Splunk without 400 errors
"""

import requests
import json
import time
from datetime import datetime
import urllib3
urllib3.disable_warnings()

class SimpleSplunkSender:
    def __init__(self):
        # Load credentials
        with open('splunk_credentials.json', 'r') as f:
            self.config = json.load(f)
        
        self.hec_url = f"http://localhost:{self.config['hec_port']}/services/collector/event"
        self.headers = {"Authorization": f"Splunk {self.config['hec_token']}"}
    
    def send_event(self, event_data):
        """Send event to Splunk"""
        # Format event properly
        payload = {
            "event": event_data,
            "index": self.config.get('splunk_index', 'main'),
            "sourcetype": self.config.get('splunk_sourcetype', 'insider_threat'),
            "time": time.time()
        }
        
        try:
            response = requests.post(
                self.hec_url,
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"[SUCCESS] Event sent: {event_data.get('user', 'unknown')}")
                return True
            else:
                print(f"[ERROR] Status {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to send: {e}")
            return False
    
    def test_connection(self):
        """Test HEC connection"""
        test_event = {
            "message": "HEC connection test",
            "timestamp": datetime.now().isoformat(),
            "status": "test"
        }
        
        print("Testing HEC connection...")
        return self.send_event(test_event)

def main():
    sender = SimpleSplunkSender()
    
    # Test connection
    if sender.test_connection():
        print("\\nHEC connection successful!")
        
        # Send sample threat event
        threat_event = {
            "timestamp": datetime.now().isoformat(),
            "user": "test.user",
            "pc": "TEST-PC",
            "activity_type": "HTTP",
            "threat_score": 0.75,
            "risk_category": "high",
            "is_threat": True,
            "details": "Suspicious activity detected"
        }
        
        print("\\nSending sample threat event...")
        sender.send_event(threat_event)
    else:
        print("\\nHEC connection failed. Please check configuration.")

if __name__ == "__main__":
    main()
'''
        
        with open("simple_hec_sender.py", 'w') as f:
            f.write(sender_script)
        
        print("[SUCCESS] Created simple_hec_sender.py")
    
    def restart_splunk(self):
        """Restart Splunk service"""
        print("\n[5] RESTARTING SPLUNK")
        print("=" * 50)
        
        try:
            print("[INFO] Stopping Splunk...")
            subprocess.run([os.path.join(self.splunk_home, "bin", "splunk.exe"), "stop"], 
                         capture_output=True, text=True)
            time.sleep(5)
            
            print("[INFO] Starting Splunk...")
            subprocess.run([os.path.join(self.splunk_home, "bin", "splunk.exe"), "start"], 
                         capture_output=True, text=True)
            time.sleep(10)
            
            print("[SUCCESS] Splunk restarted")
        except Exception as e:
            print(f"[WARNING] Could not restart Splunk automatically: {e}")
            print("Please restart Splunk manually")
    
    def create_test_script(self):
        """Create comprehensive test script"""
        print("\n[6] CREATING TEST SCRIPT")
        print("=" * 50)
        
        test_script = '''#!/usr/bin/env python3
"""Test Splunk Integration"""

import subprocess
import sys

print("TESTING SPLUNK INTEGRATION")
print("=" * 50)

# Test 1: Simple HEC sender
print("\\n[TEST 1] Testing HEC connection...")
subprocess.run([sys.executable, "simple_hec_sender.py"])

# Test 2: Automated integration
print("\\n[TEST 2] Testing automated integration...")
subprocess.run([sys.executable, "automated_splunk_integration.py"])

print("\\n" + "=" * 50)
print("TESTS COMPLETE")
print("Check Splunk Web UI for events:")
print("http://localhost:8000")
print("Search: index=main sourcetype=insider_threat")
'''
        
        with open("test_splunk_integration.py", 'w') as f:
            f.write(test_script)
        
        print("[SUCCESS] Created test_splunk_integration.py")
    
    def run_complete_fix(self):
        """Run all fixes"""
        print("\nSPLUNK INTEGRATION COMPLETE FIX")
        print("=" * 60)
        
        # Run all fixes
        self.fix_hec_configuration()
        self.update_integration_scripts()
        self.deploy_to_splunk_apps()
        self.create_simple_hec_sender()
        self.create_test_script()
        self.restart_splunk()
        
        print("\n" + "=" * 60)
        print("FIX COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run: python test_splunk_integration.py")
        print("2. Check Splunk Web UI: http://localhost:8000")
        print("3. Search for events: index=main sourcetype=insider_threat")
        print("\nIf you still get 400 errors:")
        print("- Check HEC is enabled in Splunk")
        print("- Verify the token in splunk_credentials.json")
        print("- Ensure Splunk is running")

def main():
    fixer = SplunkIntegrationFixer()
    fixer.run_complete_fix()

if __name__ == "__main__":
    main()
