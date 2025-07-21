#!/usr/bin/env python3
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
