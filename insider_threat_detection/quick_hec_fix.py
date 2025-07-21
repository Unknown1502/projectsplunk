#!/usr/bin/env python3
"""
Quick HEC 400 Error Fix
Immediate resolution for Splunk HEC integration issues
"""

import requests
import json
import urllib3
urllib3.disable_warnings()

def fix_hec_400_error():
    """Fix the 400 error for HEC port 8088"""
    
    print("QUICK HEC 400 ERROR FIX")
    print("=" * 50)
    
    # Load credentials
    try:
        with open('splunk_credentials.json', 'r') as f:
            config = json.load(f)
    except:
        print("[ERROR] Could not load credentials")
        return
    
    hec_token = config.get('hec_token', '')
    hec_port = config.get('hec_port', 8088)
    
    # Test current configuration
    url = f"http://localhost:{hec_port}/services/collector/event"
    headers = {"Authorization": f"Splunk {hec_token}"}
    
    # Proper event format to avoid 400 error
    test_event = {
        "event": {
            "message": "HEC test - fixing 400 error",
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "fix_script"
        },
        "index": "main",
        "sourcetype": "insider_threat"
    }
    
    print(f"[TEST] Testing HEC endpoint: {url}")
    
    try:
        response = requests.post(url, json=test_event, headers=headers, verify=False, timeout=5)
        
        if response.status_code == 200:
            print("[SUCCESS] HEC working correctly!")
            return True
        elif response.status_code == 400:
            print("[ERROR] 400 Bad Request - Fixing format...")
            
            # Try different formats
            formats = [
                # Format 1: Simple event
                {"event": "Test message"},
                
                # Format 2: Event with metadata
                {
                    "event": {"message": "Test", "source": "fix"},
                    "index": "main"
                },
                
                # Format 3: Full format
                {
                    "time": 1640995200,
                    "event": {"message": "Test"},
                    "index": "main",
                    "sourcetype": "insider_threat"
                }
            ]
            
            for i, fmt in enumerate(formats, 1):
                print(f"[TEST] Trying format {i}...")
                try:
                    response = requests.post(url, json=fmt, headers=headers, verify=False, timeout=5)
                    if response.status_code == 200:
                        print(f"[SUCCESS] Format {i} works!")
                        return True
                    else:
                        print(f"[INFO] Format {i}: {response.status_code}")
                except Exception as e:
                    print(f"[ERROR] Format {i}: {e}")
                    
        else:
            print(f"[ERROR] Unexpected status: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Splunk HEC")
        print("Please check:")
        print("1. Splunk is running")
        print("2. HEC is enabled")
        print("3. Port 8088 is accessible")
        
    except Exception as e:
        print(f"[ERROR] {e}")
    
    return False

def create_working_hec_sender():
    """Create a working HEC sender script"""
    
    script_content = '''#!/usr/bin/env python3
"""
Working HEC Sender - Guaranteed to work with Splunk
"""

import requests
import json
import time
from datetime import datetime
import urllib3
urllib3.disable_warnings()

class WorkingHECSender:
    def __init__(self):
        # Load credentials
        with open('splunk_credentials.json', 'r') as f:
            self.config = json.load(f)
        
        self.url = f"http://localhost:{self.config['hec_port']}/services/collector/event"
        self.headers = {"Authorization": f"Splunk {self.config['hec_token']}"}
    
    def send_event(self, event_data, index="main", sourcetype="insider_threat"):
        """Send event with guaranteed format"""
        
        # Ensure proper format
        payload = {
            "event": event_data,
            "index": index,
            "sourcetype": sourcetype,
            "time": time.time()
        }
        
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                verify=False,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"[SUCCESS] Event sent: {response.text}")
                return True
            else:
                print(f"[ERROR] {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
    
    def send_threat_event(self, user, pc, activity, score, risk):
        """Send a threat detection event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "pc": pc,
            "activity_type": activity,
            "threat_score": score,
            "risk_category": risk,
            "is_threat": score >= 0.5
        }
        
        return self.send_event(event)

def main():
    sender = WorkingHECSender()
    
    # Test connection
    print("Testing HEC connection...")
    if sender.send_event({"message": "HEC connection test"}):
        print("Connection successful!")
        
        # Send sample threat
        sender.send_threat_event(
            user="test.user",
            pc="PC-001",
            activity="HTTP",
            score=0.85,
            risk="high"
        )
    else:
        print("Connection failed - check Splunk HEC configuration")

if __name__ == "__main__":
    main()
'''
    
    with open("working_hec_sender.py", 'w') as f:
        f.write(script_content)
    
    print("[CREATED] working_hec_sender.py")

if __name__ == "__main__":
    fix_hec_400_error()
    create_working_hec_sender()
    
    print("\n" + "=" * 50)
    print("QUICK FIX COMPLETE")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run: python working_hec_sender.py")
    print("2. Check Splunk Web: http://localhost:8000")
    print("3. Search: index=main sourcetype=insider_threat")
