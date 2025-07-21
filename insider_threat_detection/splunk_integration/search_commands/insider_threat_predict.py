#!/usr/bin/env python3
"""
Custom Splunk Search Command for Insider Threat Detection
Usage: | insider_threat_predict model_path="/path/to/model" threshold=0.5
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators

# Add the parent directory to the path to import our modules
# Handle both Windows and Unix paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..', '..', '..')
sys.path.insert(0, os.path.normpath(parent_dir))

# Also add the bin directory for Splunk deployment
bin_dir = os.path.dirname(__file__)
sys.path.insert(0, bin_dir)

try:
    from src.models.lstm_model import InsiderThreatLSTM
    from src.data.preprocessor import DataPreprocessor
    from src.data.feature_engineer import FeatureEngineer
    from src.utils.checkpoint_manager import CheckpointManager
    from src.utils.explainability import ThreatExplainer
    import tensorflow as tf
except ImportError as e:
    # Fallback for when running in Splunk environment
    # Try to import from the bin directory structure
    try:
        from bin.src.models.lstm_model import InsiderThreatLSTM
        from bin.src.data.preprocessor import DataPreprocessor
        from bin.src.data.feature_engineer import FeatureEngineer
        from bin.src.utils.checkpoint_manager import CheckpointManager
        from bin.src.utils.explainability import ThreatExplainer
        import tensorflow as tf
    except ImportError:
        # Final fallback - create minimal implementations
        class DataPreprocessor:
            def __init__(self): pass
            def clean_dates(self, df): return df
            def create_time_features(self, df): return df
            def handle_missing_values(self, df): return df
            def create_sequences(self, df, cols): return [], []
        
        class FeatureEngineer:
            def __init__(self): pass
            def create_user_behavior_features(self, df): return df
            def encode_categorical_features(self, df): return df
            def detect_anomalies(self, df): return df
            def create_advanced_features(self, df): return df
        
        class CheckpointManager:
            def __init__(self): pass
        
        class ThreatExplainer:
            def __init__(self, model, features): pass
            def explain_prediction(self, data): return {}

@Configuration()
class InsiderThreatPredictCommand(StreamingCommand):
    """
    Custom Splunk search command for insider threat prediction.
    
    Usage:
    | insider_threat_predict model_path="/opt/splunk/etc/apps/insider_threat/models/model.h5" threshold=0.5
    """
    
    model_path = Option(
        doc='Path to the trained model file',
        require=True,
        validate=validators.File()
    )
    
    threshold = Option(
        doc='Prediction threshold (default: 0.5)',
        require=False,
        validate=validators.Float(0.0, 1.0),
        default=0.5
    )
    
    explain = Option(
        doc='Include explainability analysis (default: true)',
        require=False,
        validate=validators.Boolean(),
        default=True
    )
    
    def __init__(self):
        super(InsiderThreatPredictCommand, self).__init__()
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.explainer = None
    
    def prepare(self):
        """Load the model and preprocessing artifacts."""
        try:
            # Load model and artifacts
            self.load_model_artifacts()
            return True
        except Exception as e:
            self.write_error(f"Failed to load model: {str(e)}")
            return False
    
    def load_model_artifacts(self):
        """Load the trained model and preprocessing artifacts."""
        try:
            # Load the model
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Load preprocessing artifacts
            checkpoint_manager = CheckpointManager()
            model_dir = os.path.dirname(self.model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load label encoders
            encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
            if os.path.exists(encoders_path):
                import pickle
                with open(encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
            
            # Load feature columns
            features_path = os.path.join(model_dir, 'feature_columns.pkl')
            if os.path.exists(features_path):
                import pickle
                with open(features_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
            
            # Initialize preprocessor and feature engineer
            self.preprocessor = DataPreprocessor()
            self.feature_engineer = FeatureEngineer()
            
            if self.scaler:
                self.preprocessor.scaler = self.scaler
            if self.label_encoders:
                self.feature_engineer.label_encoders = self.label_encoders
            
            # Initialize explainer if requested
            if self.explain and self.feature_columns:
                self.explainer = ThreatExplainer(self.model, self.feature_columns)
                
        except Exception as e:
            raise Exception(f"Error loading model artifacts: {str(e)}")
    
    def stream(self, records):
        """Process each record through the insider threat detection model."""
        if not self.prepare():
            return
        
        for record in records:
            try:
                # Convert Splunk record to DataFrame
                df = self.splunk_record_to_dataframe(record)
                
                # Preprocess the data
                processed_df = self.preprocess_data(df)
                
                # Make prediction
                prediction_result = self.predict(processed_df)
                
                # Add prediction results to the record
                record.update(prediction_result)
                
                yield record
                
            except Exception as e:
                # Add error information to the record
                record['prediction_error'] = str(e)
                record['threat_score'] = 0.0
                record['is_threat'] = False
                yield record
    
    def splunk_record_to_dataframe(self, record):
        """Convert a Splunk record to a pandas DataFrame."""
        # Map Splunk CIM fields to our model fields
        field_mapping = {
            'user': 'user',
            'src': 'pc',
            '_time': 'date',
            'action': 'activity_type',
            'signature': 'details',
            'process_name': 'details',
            'url': 'details'
        }
        
        # Create a dictionary with mapped fields
        mapped_data = {}
        for splunk_field, model_field in field_mapping.items():
            if splunk_field in record:
                mapped_data[model_field] = record[splunk_field]
        
        # Ensure required fields exist
        required_fields = ['user', 'pc', 'date', 'activity_type', 'details']
        for field in required_fields:
            if field not in mapped_data:
                mapped_data[field] = 'unknown'
        
        # Convert to DataFrame
        df = pd.DataFrame([mapped_data])
        
        # Convert date field
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], unit='s')
            except:
                df['date'] = pd.to_datetime('now')
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for prediction."""
        try:
            # Clean dates
            df = self.preprocessor.clean_dates(df)
            
            # Create time features
            df = self.preprocessor.create_time_features(df)
            
            # Handle missing values
            df = self.preprocessor.handle_missing_values(df)
            
            # Feature engineering
            df = self.feature_engineer.create_user_behavior_features(df)
            df = self.feature_engineer.encode_categorical_features(df)
            df = self.feature_engineer.detect_anomalies(df)
            df = self.feature_engineer.create_advanced_features(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in preprocessing: {str(e)}")
    
    def predict(self, df):
        """Make prediction using the loaded model."""
        try:
            # Ensure we have the required feature columns
            if self.feature_columns:
                available_features = [col for col in self.feature_columns if col in df.columns]
                if len(available_features) < len(self.feature_columns) * 0.8:  # At least 80% of features
                    raise Exception("Insufficient features for prediction")
                df_features = df[available_features]
            else:
                # Use all numeric columns
                df_features = df.select_dtypes(include=[np.number])
            
            # Create sequences (for single record, we'll use a simple approach)
            if len(df_features) == 1:
                # For single record, repeat it to create a sequence
                sequence_length = 8  # Default sequence length
                sequence_data = np.tile(df_features.values, (sequence_length, 1))
                X = np.array([sequence_data])
            else:
                # Multiple records - create sequences normally
                X, _ = self.preprocessor.create_sequences(df, list(df_features.columns))
            
            # Scale features if scaler is available
            if self.scaler and len(X) > 0:
                original_shape = X.shape
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = self.scaler.transform(X_reshaped)
                X = X_scaled.reshape(original_shape)
            
            # Make prediction
            if len(X) > 0:
                predictions = self.model.predict(X, verbose=0)
                threat_score = float(predictions[0][0]) if len(predictions) > 0 else 0.0
            else:
                threat_score = 0.0
            
            # Determine if it's a threat based on threshold
            is_threat = threat_score >= self.threshold
            
            # Calculate risk category
            if threat_score >= 0.8:
                risk_category = "high"
            elif threat_score >= 0.5:
                risk_category = "medium"
            elif threat_score >= 0.2:
                risk_category = "low"
            else:
                risk_category = "info"
            
            result = {
                'threat_score': round(threat_score, 4),
                'is_threat': is_threat,
                'risk_category': risk_category,
                'model_version': '1.0',
                'prediction_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Add explainability if enabled and threat detected
            if self.explain and self.explainer and is_threat:
                try:
                    explanation = self.explainer.explain_prediction(X[0] if len(X) > 0 else np.array([]))
                    
                    # Add top contributing features
                    if explanation.get('explanations'):
                        result['threat_explanation'] = '; '.join(explanation['explanations'][:3])
                    
                    if explanation.get('risk_factors'):
                        result['risk_factors'] = ', '.join(explanation['risk_factors'])
                    
                    if explanation.get('top_features'):
                        top_feature = explanation['top_features'][0]
                        result['primary_risk_indicator'] = top_feature.get('feature', 'unknown')
                        
                except Exception as e:
                    self.write_warning(f"Explainability failed: {e}")
                    result['threat_explanation'] = "Analysis not available"
            
            return result
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

dispatch(InsiderThreatPredictCommand, sys.argv, sys.stdin, sys.stdout, __name__)
