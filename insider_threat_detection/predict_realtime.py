#!/usr/bin/env python3
"""
Real-time Insider Threat Prediction Script
Predict individual records using the trained model
"""

import os
import sys
import json
import pickle
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from src.utils.logger import get_logger
from src.utils.console_utils import safe_print, print_success, print_error, print_warning

logger = get_logger(__name__)

class RealTimePredictor:
    """Real-time prediction for individual records."""
    
    def __init__(self, model_path='../r1/checkpoints/model_checkpoint.h5'):
        """Initialize the predictor with pre-trained model."""
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        # Load model and artifacts
        self._load_model()
        self._load_artifacts()
    
    def _load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        safe_print("Loading trained model...")
        self.model = tf.keras.models.load_model(self.model_path)
        print_success(f"Model loaded from {self.model_path}")
    
    def _load_artifacts(self):
        """Load preprocessing artifacts."""
        artifacts_dir = os.path.dirname(self.model_path)
        
        # Load scaler
        scaler_path = os.path.join(artifacts_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print_success("Scaler loaded")
        
        # Load label encoders
        encoders_path = os.path.join(artifacts_dir, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            print_success("Label encoders loaded")
        
        # Load feature columns
        columns_path = os.path.join(artifacts_dir, 'feature_columns.pkl')
        if os.path.exists(columns_path):
            with open(columns_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            print_success("Feature columns loaded")
    
    def predict_single_record(self, record_data):
        """
        Predict a single record.
        
        Args:
            record_data: Dictionary with fields:
                - user: Username
                - pc: Computer name
                - date: Datetime string
                - activity_type: Type of activity (HTTP, DEVICE, LOGON, etc.)
                - details: Activity details
        
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([record_data])
        
        # Add ID column if missing
        if 'id' not in df.columns:
            df['id'] = 1
        
        # Preprocess
        safe_print("Preprocessing data...")
        df = self.preprocessor.clean_dates(df)
        df = self.preprocessor.create_time_features(df)
        df = self.preprocessor.handle_missing_values(df)
        
        # Feature engineering
        safe_print("Engineering features...")
        df = self.feature_engineer.create_user_behavior_features(df)
        df = self.feature_engineer.encode_categorical_features(df)
        
        # Select and order features
        if self.feature_columns:
            # Ensure all required columns exist
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_columns]
        
        # Scale features
        if self.scaler:
            df_scaled = self.scaler.transform(df)
        else:
            df_scaled = df.values
        
        # Reshape for LSTM (samples, timesteps, features)
        # For single prediction, we'll use the last 8 timesteps
        timesteps = 8
        if df_scaled.shape[0] < timesteps:
            # Pad with zeros if not enough data
            padding = np.zeros((timesteps - df_scaled.shape[0], df_scaled.shape[1]))
            df_scaled = np.vstack([padding, df_scaled])
        
        X = df_scaled[-timesteps:].reshape(1, timesteps, df_scaled.shape[1])
        
        # Make prediction
        safe_print("Making prediction...")
        prediction = self.model.predict(X, verbose=0)[0][0]
        
        # Determine risk level
        if prediction < 0.3:
            risk_level = "LOW"
            risk_color = "success"
        elif prediction < 0.7:
            risk_level = "MEDIUM"
            risk_color = "warning"
        else:
            risk_level = "HIGH"
            risk_color = "error"
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'user': record_data.get('user', 'unknown'),
            'pc': record_data.get('pc', 'unknown'),
            'activity_type': record_data.get('activity_type', 'unknown'),
            'threat_score': float(prediction),
            'risk_level': risk_level,
            'is_threat': prediction > 0.5
        }
        
        # Display result
        safe_print("\n" + "="*50)
        safe_print("PREDICTION RESULT")
        safe_print("="*50)
        safe_print(f"User: {result['user']}")
        safe_print(f"PC: {result['pc']}")
        safe_print(f"Activity: {result['activity_type']}")
        
        if risk_color == "success":
            print_success(f"Threat Score: {result['threat_score']:.4f}")
            print_success(f"Risk Level: {result['risk_level']}")
            print_success(f"Is Threat: {'YES' if result['is_threat'] else 'NO'}")
        elif risk_color == "warning":
            print_warning(f"Threat Score: {result['threat_score']:.4f}")
            print_warning(f"Risk Level: {result['risk_level']}")
            print_warning(f"Is Threat: {'YES' if result['is_threat'] else 'NO'}")
        else:
            print_error(f"Threat Score: {result['threat_score']:.4f}")
            print_error(f"Risk Level: {result['risk_level']}")
            print_error(f"Is Threat: {'YES' if result['is_threat'] else 'NO'}")
        
        safe_print("="*50 + "\n")
        
        return result
    
    def predict_from_json(self, json_path):
        """Predict from a JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            results = []
            for record in data:
                result = self.predict_single_record(record)
                results.append(result)
            return results
        else:
            return self.predict_single_record(data)
    
    def interactive_prediction(self):
        """Interactive mode for entering data manually."""
        safe_print("\n[INTERACTIVE] INTERACTIVE PREDICTION MODE")
        safe_print("Enter record details for prediction:\n")
        
        record = {}
        record['user'] = input("Username (e.g., ACM2278): ").strip() or "ACM2278"
        record['pc'] = input("PC Name (e.g., PC-7654): ").strip() or "PC-7654"
        record['date'] = input("Date/Time (YYYY-MM-DD HH:MM:SS) [Enter for now]: ").strip()
        
        if not record['date']:
            record['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        safe_print("\nActivity Types: HTTP, DEVICE, LOGON, EMAIL")
        record['activity_type'] = input("Activity Type: ").strip().upper() or "HTTP"
        
        record['details'] = input("Activity Details (e.g., URL, file path): ").strip() or "http://example.com"
        
        return self.predict_single_record(record)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Real-time Insider Threat Prediction')
    parser.add_argument('--model-path', type=str, 
                       default='../r1/checkpoints/model_checkpoint.h5',
                       help='Path to trained model')
    parser.add_argument('--json', type=str, help='Path to JSON file with record(s)')
    parser.add_argument('--interactive', action='store_true', 
                       help='Interactive mode for manual input')
    
    # Individual record parameters
    parser.add_argument('--user', type=str, help='Username')
    parser.add_argument('--pc', type=str, help='PC name')
    parser.add_argument('--date', type=str, help='Date/time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--activity', type=str, help='Activity type')
    parser.add_argument('--details', type=str, help='Activity details')
    
    args = parser.parse_args()
    
    # Initialize predictor
    try:
        predictor = RealTimePredictor(model_path=args.model_path)
    except FileNotFoundError as e:
        print_error(f"Error: {e}")
        print_error("Please ensure the model has been trained first.")
        return
    
    # Determine prediction mode
    if args.json:
        # Predict from JSON file
        results = predictor.predict_from_json(args.json)
        
        # Save results
        output_file = args.json.replace('.json', '_predictions.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print_success(f"Results saved to: {output_file}")
        
    elif args.interactive:
        # Interactive mode
        predictor.interactive_prediction()
        
    elif args.user:
        # Command line arguments
        record = {
            'user': args.user,
            'pc': args.pc or 'PC-0000',
            'date': args.date or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'activity_type': args.activity or 'HTTP',
            'details': args.details or ''
        }
        predictor.predict_single_record(record)
        
    else:
        # Default to interactive mode when no arguments provided
        safe_print("\n[INTERACTIVE] No arguments provided - Starting Interactive Mode")
        safe_print("Choose an option:")
        safe_print("1. Interactive input (enter data manually)")
        safe_print("2. Demo prediction (use sample data)")
        
        choice = input("\nEnter choice (1 or 2) [default: 1]: ").strip() or "1"
        
        if choice == "2":
            # Demo prediction
            safe_print("\n[DEMO] DEMO PREDICTION")
            print_warning("Running demo prediction...\n")
            
            demo_record = {
                'user': 'ACM2278',
                'pc': 'PC-7654',
                'date': '2023-01-15 14:30:00',
                'activity_type': 'HTTP',
                'details': 'http://wikileaks.org/files/secret_docs.zip'
            }
            
            safe_print("Demo Record:")
            for key, value in demo_record.items():
                safe_print(f"  {key}: {value}")
            
            predictor.predict_single_record(demo_record)
        else:
            # Interactive mode
            predictor.interactive_prediction()


if __name__ == "__main__":
    main()
