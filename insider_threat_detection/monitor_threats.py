#!/usr/bin/env python3
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
