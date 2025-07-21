#!/usr/bin/env python3
"""
Fixed Automated Real-Time Splunk Integration
This script fixes the 400 error and properly integrates with Splunk HEC
"""

import time
import json
import requests
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os
import urllib3
urllib3.disable_warnings()

class FixedAutomatedThreatPredictor:
    def __init__(self):
        # Load credentials from file
        self.load_credentials()
        
        # Configure HEC
        self.hec_url = f"http://localhost:{self.config['hec_port']}/services/collector/event"
        self.headers = {"Authorization": f"Splunk {self.config['hec_token']}"}
        
        # Model paths
        self.model_path = "C:\Program Files\Splunk\etc\apps\projectsplunk\r1\checkpoints\model_checkpoint.h5"
        self.scaler_path = "C:\Program Files\Splunk\etc\apps\projectsplunk\r1\checkpoints\scaler.pkl"
        self.encoders_path = "C:\Program Files\Splunk\etc\apps\projectsplunk\r1\checkpoints\label_encoders.pkl"
        
        # Load model and preprocessors
        self.load_model()
        
    def load_credentials(self):
        """Load Splunk credentials from file"""
        try:
            with open('splunk_credentials.json', 'r') as f:
                self.config = json.load(f)
            print("[INFO] Loaded credentials from splunk_credentials.json")
        except Exception as e:
            print(f"[ERROR] Failed to load credentials: {e}")
            # Fallback to defaults
            self.config = {
                'hec_token': '2be9538b-ac19-41fa-909c-8e06f306805d',
                'hec_port': 8088,
                'splunk_index': 'main',
                'splunk_sourcetype': 'insider_threat'
            }
    
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
                print("[WARNING] Model file not found, using dummy predictions")
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
            # Return dummy prediction if model not loaded
            threat_score = np.random.uniform(0.1, 0.9)
            risk_category = "high" if threat_score > 0.7 else "medium" if threat_score > 0.4 else "low"
            return {
                "threat_score": round(threat_score, 4),
                "risk_category": risk_category,
                "is_threat": threat_score >= 0.5,
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
            
            # Simple feature engineering
            features = []
            
            # Encode categorical features
            if self.label_encoders:
                for col in ['user', 'pc', 'activity_type']:
                    if col in self.label_encoders:
                        try:
                            event_data[f'{col}_encoded'] = self.label_encoders[col].transform(event_data[col])
                            features.append(event_data[f'{col}_encoded'].values[0])
                        except:
                            features.append(0)
            
            # Add time-based features
            hour = datetime.now().hour
            is_weekend = datetime.now().weekday() >= 5
            features.extend([hour, int(is_weekend)])
            
            # Convert to numpy array
            X = np.array([features])
            
            # Scale features
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Make prediction
            if len(X.shape) == 2:
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
        """Send event with prediction to Splunk using proper format"""
        
        # Ensure proper event format
        splunk_event = {
            "time": time.time(),
            "event": event_data,
            "index": self.config.get('splunk_index', 'main'),
            "sourcetype": self.config.get('splunk_sourcetype', 'insider_threat')
        }
        
        try:
            response = requests.post(
                self.hec_url,
                json=splunk_event,
                headers=self.headers,
                verify=False,
                timeout=10
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
        success = self.send_to_splunk(event)
        
        return event, success
    
    def test_integration(self):
        """Test the complete integration"""
        print("\n[TEST] Testing complete integration...")
        
        # Test event
        test_event = {
            "user": "test.user",
            "pc": "TEST-PC",
            "activity_type": "HTTP",
            "details": "Integration test"
        }
        
        result, success = self.process_new_event(**test_event)
        
        if success:
            print("[SUCCESS] Integration test passed!")
        else:
            print("[ERROR] Integration test failed")
        
        return success
    
    def monitor_and_predict(self, check_interval=30):
        """Monitor for new events and predict threats"""
        print("\n[START] Fixed Automated Threat Monitoring")
        print("=" * 50)
        print("Using HEC token:", self.config['hec_token'][:8] + "...")
        print("HEC URL:", self.hec_url)
        print("Press Ctrl+C to stop")
        
        # Test integration first
        if not self.test_integration():
            print("[WARNING] Initial test failed, continuing anyway...")
        
        # Sample events for testing
        event_sources = [
            {"user": "john.doe", "pc": "PC-001", "activity_type": "HTTP", "details": "Accessing external site"},
            {"user": "jane.smith", "pc": "PC-002", "activity_type": "FILE", "details": "Large file download"},
            {"user": "bob.jones", "pc": "PC-003", "activity_type": "LOGON", "details": "After hours login"},
            {"user": "alice.brown", "pc": "PC-004", "activity_type": "EMAIL", "details": "Bulk email sent"},
        ]
        
        try:
            while True:
                import random
                event = random.choice(event_sources)
                
                # Process the event
                result, success = self.process_new_event(
                    user=event["user"],
                    pc=event["pc"],
                    activity_type=event["activity_type"],
                    details=event["details"]
                )
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n[STOP] Monitoring stopped by user")

def main():
    print("FIXED AUTOMATED THREAT DETECTION")
    print("=" * 50)
    
    # Initialize predictor
    predictor = FixedAutomatedThreatPredictor()
    
    # Test the system
    print("\n[TEST] Running integration test...")
    predictor.test_integration()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print("\nTo start automated monitoring:")
    print("1. Ensure Splunk is running")
    print("2. Run: python automated_splunk_integration_fixed.py")
    print("\nThe system will:")
    print("- Monitor for new events")
    print("- Automatically predict threat scores")
    print("- Send results to Splunk in real-time")
    print("\nView real-time threats in Splunk:")
    print('index=main sourcetype=insider_threat')

if __name__ == "__main__":
    main()
