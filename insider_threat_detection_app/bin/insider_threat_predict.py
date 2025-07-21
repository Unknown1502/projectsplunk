#!/usr/bin/env python3
"""
Splunk Custom Search Command for Insider Threat Detection
Refactored for proper Splunk app structure
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators

# Add the lib directory to Python path
app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_dir = os.path.join(app_root, 'bin', 'lib')
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

# Import our modules
try:
    from src.models.lstm_model import InsiderThreatLSTM
    from src.data.preprocessor import DataPreprocessor
    from src.data.feature_engineer import FeatureEngineer
    from src.utils.checkpoint_manager import CheckpointManager
    from src.utils.explainability import ThreatExplainer
    import tensorflow as tf
except ImportError as e:
    # Log the error for debugging
    import traceback
    sys.stderr.write(f"Import error: {str(e)}\n")
    sys.stderr.write(f"Python path: {sys.path}\n")
    sys.stderr.write(traceback.format_exc())
    raise

@Configuration()
class InsiderThreatPredictCommand(StreamingCommand):
    """
    Custom Splunk search command for insider threat prediction.
    
    Usage:
    | insider_threat_predict [model_name="latest"] [threshold=0.5] [explain=true]
    
    Examples:
    | insider_threat_predict
    | insider_threat_predict model_name="model_v2" threshold=0.7
    | insider_threat_predict threshold=0.8 explain=false
    """
    
    model_name = Option(
        doc='Name of the model to use (default: latest)',
        require=False,
        validate=validators.Match("model_name", r"^[\w\-\.]+$"),
        default="latest"
    )
    
    threshold = Option(
        doc='Prediction threshold (default: 0.5)',
        require=False,
        validate=validators.Float(minimum=0.0, maximum=1.0),
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
        self._initialized = False
    
    def prepare(self):
        """Load the model and preprocessing artifacts."""
        if self._initialized:
            return True
            
        try:
            self.load_model_artifacts()
            self._initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def load_model_artifacts(self):
        """Load the trained model and preprocessing artifacts."""
        try:
            # Determine model path
            models_dir = os.path.join(self.service.app_path, 'bin', 'models')
            
            if self.model_name == "latest":
                # Find the latest model
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
                if not model_files:
                    raise Exception("No model files found in models directory")
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                model_path = os.path.join(models_dir, model_files[0])
            else:
                model_path = os.path.join(models_dir, f"{self.model_name}.h5")
            
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found: {model_path}")
            
            # Load the model
            self.logger.info(f"Loading model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Load preprocessing artifacts
            model_dir = os.path.dirname(model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info("Loaded scaler")
            
            # Load label encoders
            encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
            if os.path.exists(encoders_path):
                import pickle
                with open(encoders_path, 'rb') as f:
                    self.label_encoders = pickle.load(f)
                self.logger.info("Loaded label encoders")
            
            # Load feature columns
            features_path = os.path.join(model_dir, 'feature_columns.pkl')
            if os.path.exists(features_path):
                import pickle
                with open(features_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                self.logger.info(f"Loaded {len(self.feature_columns)} feature columns")
            
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
                self.logger.info("Initialized explainer")
                
        except Exception as e:
            self.logger.error(f"Error loading model artifacts: {str(e)}")
            raise
    
    def stream(self, records):
        """Process each record through the insider threat detection model."""
        if not self.prepare():
            for record in records:
                record['prediction_error'] = "Failed to load model"
                record['threat_score'] = 0.0
                record['is_threat'] = False
                yield record
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
                self.logger.error(f"Prediction error: {str(e)}")
                record['prediction_error'] = str(e)
                record['threat_score'] = 0.0
                record['is_threat'] = False
                yield record
    
    def splunk_record_to_dataframe(self, record):
        """Convert a Splunk record to a pandas DataFrame."""
        # Map Splunk CIM fields to our model fields
        field_mapping = {
            'user': 'user',
            'src_user': 'user',
            'src': 'pc',
            'src_host': 'pc',
            'dest': 'pc',
            'dest_host': 'pc',
            '_time': 'date',
            'action': 'activity_type',
            'signature': 'details',
            'process': 'details',
            'process_name': 'details',
            'url': 'details',
            'file_name': 'details',
            'file_path': 'details'
        }
        
        # Create a dictionary with mapped fields
        mapped_data = {}
        for splunk_field, model_field in field_mapping.items():
            if splunk_field in record:
                value = record[splunk_field]
                if value and str(value).strip():
                    if model_field not in mapped_data or not mapped_data[model_field]:
                        mapped_data[model_field] = value
        
        # Ensure required fields exist
        required_fields = ['user', 'pc', 'date', 'activity_type', 'details']
        for field in required_fields:
            if field not in mapped_data:
                if field == 'date' and '_time' in record:
                    mapped_data[field] = record['_time']
                else:
                    mapped_data[field] = 'unknown'
        
        # Convert to DataFrame
        df = pd.DataFrame([mapped_data])
        
        # Convert date field
        if 'date' in df.columns:
            try:
                # Try to parse as Unix timestamp first
                df['date'] = pd.to_datetime(df['date'], unit='s')
            except:
                try:
                    # Try to parse as string
                    df['date'] = pd.to_datetime(df['date'])
                except:
                    # Default to current time
                    df['date'] = pd.Timestamp.now()
        
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
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def predict(self, df):
        """Make prediction using the loaded model."""
        try:
            # Ensure we have the required feature columns
            if self.feature_columns:
                available_features = [col for col in self.feature_columns if col in df.columns]
                if len(available_features) < len(self.feature_columns) * 0.5:  # At least 50% of features
                    self.logger.warning(f"Only {len(available_features)} of {len(self.feature_columns)} features available")
                
                # Add missing features with default values
                for feature in self.feature_columns:
                    if feature not in df.columns:
                        df[feature] = 0
                
                df_features = df[self.feature_columns]
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
                risk_category = "critical"
            elif threat_score >= 0.6:
                risk_category = "high"
            elif threat_score >= 0.4:
                risk_category = "medium"
            elif threat_score >= 0.2:
                risk_category = "low"
            else:
                risk_category = "info"
            
            result = {
                'threat_score': round(threat_score, 4),
                'is_threat': is_threat,
                'risk_category': risk_category,
                'model_name': self.model_name,
                'threshold_used': self.threshold,
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
                        top_features = explanation['top_features'][:3]
                        for i, feature_info in enumerate(top_features):
                            result[f'top_feature_{i+1}'] = feature_info.get('feature', 'unknown')
                            result[f'top_feature_{i+1}_impact'] = round(feature_info.get('importance', 0), 3)
                        
                except Exception as e:
                    self.logger.warning(f"Explainability failed: {e}")
                    result['threat_explanation'] = "Analysis not available"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

# Entry point for Splunk
if __name__ == "__main__":
    dispatch(InsiderThreatPredictCommand, sys.argv, sys.stdin, sys.stdout, __name__)
