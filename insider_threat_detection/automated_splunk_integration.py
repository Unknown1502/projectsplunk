#!/usr/bin/env python3
"""
Automated Real-Time Splunk Integration
This script monitors for new events and automatically predicts threats
"""

import time
import json
import requests
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os

class AutomatedThreatPredictor:
    def __init__(self):
        # Splunk HEC configuration
        self.hec_token = "763585e4-4b31-4a4a-b3d9-30ddd1a4a829"
        self.hec_url = "http://localhost:8088/services/collector/event"
        self.headers = {"Authorization": f"Splunk {self.hec_token}"}
        
        # Model paths
        self.model_path = "C:\Program Files\Splunk\etc\apps\projectsplunk\r1\checkpoints\model_checkpoint.h5"
        self.scaler_path = "C:\Program Files\Splunk\etc\apps\projectsplunk\r1\checkpoints\scaler.pkl"
        self.encoders_path = "C:\Program Files\Splunk\etc\apps\projectsplunk\r1\checkpoints\label_encoders.pkl"
        
        # Load model and preprocessors
        self.load_model()
        
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            # Import TensorFlow only when needed
            import tensorflow as tf
            
            # Load model
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print("[INFO] Model loaded successfully")
            else:
                print("[ERROR] Model file not found")
                self.model = None
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("[INFO] Scaler loaded successfully")
            else:
                self.scaler = None
                
            # Load encoders
            if os.path.exists(self.encoders_path):
                with open(self.encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                print("[INFO] Encoders loaded successfully")
            else:
                self.label_encoders = None
                
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None
    
    def predict_threat(self, user, pc, activity_type, details=""):
        """Predict threat score for a single event"""
        if not self.model:
            # Return default if model not loaded
            return {
                "threat_score": 0.5,
                "risk_category": "medium",
                "is_threat": False,
                "prediction_status": "model_not_loaded"
            }
        
        try:
            # Create event dataframe
            event_data = pd.DataFrame([{
                'user': user,
                'pc': pc,
                'activity_type': activity_type,
                'details': details,
                'date': datetime.now()
            }])
            
            # Simple feature engineering (adapt based on your model)
            features = []
            
            # Encode categorical features
            if self.label_encoders:
                for col in ['user', 'pc', 'activity_type']:
                    if col in self.label_encoders:
                        try:
                            event_data[f'{col}_encoded'] = self.label_encoders[col].transform(event_data[col])
                            features.append(event_data[f'{col}_encoded'].values[0])
                        except:
                            features.append(0)  # Unknown category
            
            # Add time-based features
            hour = datetime.now().hour
            is_weekend = datetime.now().weekday() >= 5
            features.extend([hour, int(is_weekend)])
            
            # Convert to numpy array and reshape
            X = np.array([features])
            
            # Scale features
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Make prediction
            if len(X.shape) == 2:
                # Add sequence dimension if needed (for LSTM)
                X = X.reshape(1, 1, -1)
            
            prediction = self.model.predict(X, verbose=0)
            threat_score = float(prediction[0][0])
            
            # Determine risk category
            if threat_score >= 0.8:
                risk_category = "high"
            elif threat_score >= 0.5:
                risk_category = "medium"
            else:
                risk_category = "low"
            
            return {
                "threat_score": round(threat_score, 4),
                "risk_category": risk_category,
                "is_threat": threat_score >= 0.5,
                "prediction_status": "success"
            }
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return {
                "threat_score": 0.5,
                "risk_category": "medium",
                "is_threat": False,
                "prediction_status": f"error: {str(e)}"
            }
    
    def send_to_splunk(self, event_data):
        """Send event with prediction to Splunk"""
        # Format for Splunk
        splunk_event = {
            "time": time.time(),
            "source": "automated_threat_detection",
            "sourcetype": "insider_threat_realtime",
            "index": "main",
            "event": event_data
        }
        
        try:
            # Try sending with index in URL
            url_with_index = f"{self.hec_url}?index=main"
            response = requests.post(
                url_with_index,
                headers=self.headers,
                json={"event": event_data},
                verify=False,
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"[SUCCESS] Sent to Splunk: {event_data['user']} - Risk: {event_data['risk_category']}")
                return True
            else:
                print(f"[ERROR] Splunk HEC: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to send to Splunk: {e}")
            return False
    
    def process_new_event(self, user, pc, activity_type, details=""):
        """Process a new event: predict and send to Splunk"""
        # Make prediction
        prediction = self.predict_threat(user, pc, activity_type, details)
        
        # Create complete event
        event = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "pc": pc,
            "activity_type": activity_type,
            "details": details,
            "threat_score": prediction["threat_score"],
            "risk_category": prediction["risk_category"],
            "is_threat": prediction["is_threat"],
            "prediction_status": prediction["prediction_status"],
            "model_version": "1.0"
        }
        
        # Send to Splunk
        self.send_to_splunk(event)
        
        return event
    
    def monitor_and_predict(self, check_interval=60):
        """Monitor for new events and predict threats"""
        print("\n[START] Automated Threat Monitoring")
        print("=" * 50)
        print("Monitoring for new events...")
        print("Press Ctrl+C to stop")
        
        # Simulate monitoring (in production, this would read from a real source)
        event_sources = [
            {"user": "john.doe", "pc": "PC-001", "activity_type": "HTTP", "details": "Accessing external site"},
            {"user": "jane.smith", "pc": "PC-002", "activity_type": "FILE", "details": "Large file download"},
            {"user": "bob.jones", "pc": "PC-003", "activity_type": "LOGON", "details": "After hours login"},
            {"user": "alice.brown", "pc": "PC-004", "activity_type": "EMAIL", "details": "Bulk email sent"},
        ]
        
        try:
            while True:
                # Simulate new event (in production, read from actual source)
                import random
                event = random.choice(event_sources)
                
                # Process the event
                result = self.process_new_event(
                    user=event["user"],
                    pc=event["pc"],
                    activity_type=event["activity_type"],
                    details=event["details"]
                )
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n[STOP] Monitoring stopped by user")

def create_realtime_script():
    """Create a simple script for real-time monitoring"""
    script_content = '''#!/usr/bin/env python3
"""
Real-time Threat Detection for Splunk
Run this script to continuously monitor and predict threats
"""

from automated_splunk_integration import AutomatedThreatPredictor

def main():
    # Initialize predictor
    predictor = AutomatedThreatPredictor()
    
    # Start monitoring
    predictor.monitor_and_predict(check_interval=30)  # Check every 30 seconds

if __name__ == "__main__":
    main()
'''
    
    with open("monitor_threats.py", "w") as f:
        f.write(script_content)
    
    print("[INFO] Created monitor_threats.py")

def main():
    print("AUTOMATED THREAT DETECTION SETUP")
    print("=" * 50)
    
    # Create monitoring script
    create_realtime_script()
    
    # Test the system
    predictor = AutomatedThreatPredictor()
    
    print("\n[TEST] Processing sample event...")
    result = predictor.process_new_event(
        user="test.user",
        pc="TEST-PC",
        activity_type="HTTP",
        details="Test event"
    )
    
    print(f"\nResult: {json.dumps(result, indent=2)}")
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print("\nTo start automated monitoring:")
    print("1. Fix HEC configuration in Splunk (if needed)")
    print("2. Run: python monitor_threats.py")
    print("\nThe system will:")
    print("- Monitor for new events")
    print("- Automatically predict threat scores")
    print("- Send results to Splunk in real-time")
    print("\nView real-time threats in Splunk:")
    print('index=main sourcetype=insider_threat_realtime')

if __name__ == "__main__":
    main()
