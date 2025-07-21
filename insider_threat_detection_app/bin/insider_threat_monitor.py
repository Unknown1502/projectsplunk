#!/usr/bin/env python3
"""
Splunk Custom Search Command for Real-time Insider Threat Monitoring
"""

import sys
import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    import tensorflow as tf
except ImportError as e:
    import traceback
    sys.stderr.write(f"Import error: {str(e)}\n")
    sys.stderr.write(traceback.format_exc())
    raise

@Configuration()
class InsiderThreatMonitorCommand(StreamingCommand):
    """
    Custom Splunk search command for real-time insider threat monitoring.
    
    Usage:
    | insider_threat_monitor [threshold=0.7] [window=5m] [alert_on="high"] [track_users=true]
    
    Examples:
    | insider_threat_monitor
    | insider_threat_monitor threshold=0.8 window=10m
    | insider_threat_monitor alert_on="critical" track_users=false
    """
    
    threshold = Option(
        doc='Alert threshold for threat score (default: 0.7)',
        require=False,
        validate=validators.Float(minimum=0.0, maximum=1.0),
        default=0.7
    )
    
    window = Option(
        doc='Time window for aggregation (default: 5m)',
        require=False,
        validate=validators.Match("window", r"^\d+[smhd]$"),
        default="5m"
    )
    
    alert_on = Option(
        doc='Risk level to alert on: info|low|medium|high|critical (default: high)',
        require=False,
        validate=validators.Match("alert_on", r"^(info|low|medium|high|critical)$"),
        default="high"
    )
    
    track_users = Option(
        doc='Track user behavior patterns (default: true)',
        require=False,
        validate=validators.Boolean(),
        default=True
    )
    
    aggregate = Option(
        doc='Aggregate results by user (default: true)',
        require=False,
        validate=validators.Boolean(),
        default=True
    )
    
    def __init__(self):
        super(InsiderThreatMonitorCommand, self).__init__()
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.user_history = {}
        self.alert_levels = {
            'info': 0.2,
            'low': 0.4,
            'medium': 0.6,
            'high': 0.7,
            'critical': 0.8
        }
        self._initialized = False
    
    def prepare(self):
        """Load the model and initialize monitoring."""
        if self._initialized:
            return True
            
        try:
            self.load_model()
            self._initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {str(e)}")
            return False
    
    def load_model(self):
        """Load the latest model for monitoring."""
        try:
            # Find the latest model
            models_dir = os.path.join(self.service.app_path, 'bin', 'models')
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
            
            if not model_files:
                raise Exception("No model files found")
            
            # Get the most recent model
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            model_path = os.path.join(models_dir, model_files[0])
            
            self.logger.info(f"Loading model: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Initialize preprocessors
            self.preprocessor = DataPreprocessor()
            self.feature_engineer = FeatureEngineer()
            
            # Load preprocessing artifacts if available
            self._load_preprocessing_artifacts(models_dir)
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_preprocessing_artifacts(self, models_dir):
        """Load preprocessing artifacts."""
        import pickle
        
        # Load scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.preprocessor.scaler = pickle.load(f)
        
        # Load label encoders
        encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                self.feature_engineer.label_encoders = pickle.load(f)
    
    def stream(self, records):
        """Process streaming records for monitoring."""
        if not self.prepare():
            for record in records:
                record['monitor_error'] = "Failed to initialize monitoring"
                yield record
            return
        
        # Parse window duration
        window_seconds = self._parse_window(self.window)
        current_time = datetime.now()
        
        # Process records and collect for aggregation
        processed_records = []
        
        for record in records:
            try:
                # Add monitoring fields
                record['monitor_timestamp'] = current_time.isoformat()
                record['monitor_window'] = self.window
                
                # Get user from record
                user = record.get('user') or record.get('src_user') or 'unknown'
                
                # Convert to DataFrame and predict
                df = self._record_to_dataframe(record)
                threat_score = self._predict_threat(df)
                
                # Determine risk level
                risk_level = self._get_risk_level(threat_score)
                
                # Add monitoring results
                record['threat_score'] = round(threat_score, 4)
                record['risk_level'] = risk_level
                record['alert_threshold'] = self.threshold
                
                # Check if alert should be triggered
                alert_triggered = threat_score >= self.alert_levels.get(self.alert_on, 0.7)
                record['alert_triggered'] = alert_triggered
                
                # Track user history if enabled
                if self.track_users:
                    self._update_user_history(user, threat_score, current_time)
                    user_stats = self._get_user_statistics(user, window_seconds)
                    record.update(user_stats)
                
                processed_records.append(record)
                
            except Exception as e:
                self.logger.error(f"Error processing record: {str(e)}")
                record['monitor_error'] = str(e)
                record['threat_score'] = 0.0
                processed_records.append(record)
        
        # Aggregate results if requested
        if self.aggregate and processed_records:
            yield from self._aggregate_results(processed_records, window_seconds)
        else:
            yield from processed_records
    
    def _parse_window(self, window_str):
        """Parse window string to seconds."""
        unit = window_str[-1]
        value = int(window_str[:-1])
        
        multipliers = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400
        }
        
        return value * multipliers.get(unit, 60)
    
    def _record_to_dataframe(self, record):
        """Convert record to DataFrame for prediction."""
        # Similar to predict command but simplified
        data = {
            'user': record.get('user', 'unknown'),
            'pc': record.get('src', record.get('src_host', 'unknown')),
            'date': record.get('_time', datetime.now().timestamp()),
            'activity_type': record.get('action', 'unknown'),
            'details': record.get('signature', record.get('process_name', 'unknown'))
        }
        
        df = pd.DataFrame([data])
        
        # Convert timestamp
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], unit='s')
            except:
                df['date'] = pd.Timestamp.now()
        
        return df
    
    def _predict_threat(self, df):
        """Make threat prediction."""
        try:
            # Preprocess
            df = self.preprocessor.clean_dates(df)
            df = self.preprocessor.create_time_features(df)
            df = self.preprocessor.handle_missing_values(df)
            
            # Feature engineering
            df = self.feature_engineer.create_user_behavior_features(df)
            df = self.feature_engineer.encode_categorical_features(df)
            
            # Get numeric features
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Create sequence
            sequence_length = 8
            sequence_data = np.tile(numeric_df.values, (sequence_length, 1))
            X = np.array([sequence_data])
            
            # Scale if scaler available
            if hasattr(self.preprocessor, 'scaler') and self.preprocessor.scaler:
                original_shape = X.shape
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = self.preprocessor.scaler.transform(X_reshaped)
                X = X_scaled.reshape(original_shape)
            
            # Predict
            predictions = self.model.predict(X, verbose=0)
            return float(predictions[0][0])
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return 0.0
    
    def _get_risk_level(self, threat_score):
        """Determine risk level from threat score."""
        if threat_score >= 0.8:
            return "critical"
        elif threat_score >= 0.7:
            return "high"
        elif threat_score >= 0.6:
            return "medium"
        elif threat_score >= 0.4:
            return "low"
        else:
            return "info"
    
    def _update_user_history(self, user, threat_score, timestamp):
        """Update user history for tracking."""
        if user not in self.user_history:
            self.user_history[user] = []
        
        # Add new entry
        self.user_history[user].append({
            'timestamp': timestamp,
            'threat_score': threat_score
        })
        
        # Keep only recent history (last hour)
        cutoff_time = timestamp - timedelta(hours=1)
        self.user_history[user] = [
            entry for entry in self.user_history[user]
            if entry['timestamp'] > cutoff_time
        ]
    
    def _get_user_statistics(self, user, window_seconds):
        """Calculate user statistics over the window."""
        if user not in self.user_history or not self.user_history[user]:
            return {
                'user_avg_threat_score': 0.0,
                'user_max_threat_score': 0.0,
                'user_threat_count': 0,
                'user_risk_trend': 'stable'
            }
        
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=window_seconds)
        
        # Get scores in window
        window_scores = [
            entry['threat_score'] for entry in self.user_history[user]
            if entry['timestamp'] >= window_start
        ]
        
        if not window_scores:
            return {
                'user_avg_threat_score': 0.0,
                'user_max_threat_score': 0.0,
                'user_threat_count': 0,
                'user_risk_trend': 'stable'
            }
        
        # Calculate statistics
        avg_score = np.mean(window_scores)
        max_score = np.max(window_scores)
        threat_count = sum(1 for score in window_scores if score >= self.threshold)
        
        # Determine trend
        if len(window_scores) >= 3:
            recent_avg = np.mean(window_scores[-3:])
            older_avg = np.mean(window_scores[:-3])
            if recent_avg > older_avg * 1.2:
                trend = 'increasing'
            elif recent_avg < older_avg * 0.8:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'user_avg_threat_score': round(avg_score, 4),
            'user_max_threat_score': round(max_score, 4),
            'user_threat_count': threat_count,
            'user_risk_trend': trend
        }
    
    def _aggregate_results(self, records, window_seconds):
        """Aggregate results by user."""
        user_aggregates = {}
        
        for record in records:
            user = record.get('user', 'unknown')
            
            if user not in user_aggregates:
                user_aggregates[user] = {
                    '_time': record.get('_time'),
                    'user': user,
                    'event_count': 0,
                    'threat_scores': [],
                    'risk_levels': [],
                    'alert_count': 0,
                    'hosts': set(),
                    'activities': set()
                }
            
            agg = user_aggregates[user]
            agg['event_count'] += 1
            agg['threat_scores'].append(record.get('threat_score', 0))
            agg['risk_levels'].append(record.get('risk_level', 'info'))
            
            if record.get('alert_triggered', False):
                agg['alert_count'] += 1
            
            # Collect unique hosts and activities
            host = record.get('src', record.get('src_host'))
            if host:
                agg['hosts'].add(host)
            
            activity = record.get('action', record.get('activity_type'))
            if activity:
                agg['activities'].add(activity)
        
        # Generate aggregated records
        for user, agg in user_aggregates.items():
            # Calculate aggregate metrics
            threat_scores = agg['threat_scores']
            
            yield {
                '_time': agg['_time'],
                'user': user,
                'monitor_type': 'aggregate',
                'window': self.window,
                'event_count': agg['event_count'],
                'avg_threat_score': round(np.mean(threat_scores), 4) if threat_scores else 0.0,
                'max_threat_score': round(np.max(threat_scores), 4) if threat_scores else 0.0,
                'min_threat_score': round(np.min(threat_scores), 4) if threat_scores else 0.0,
                'alert_count': agg['alert_count'],
                'unique_hosts': len(agg['hosts']),
                'unique_activities': len(agg['activities']),
                'hosts': ', '.join(sorted(agg['hosts'])),
                'activities': ', '.join(sorted(agg['activities'])),
                'primary_risk_level': max(set(agg['risk_levels']), key=agg['risk_levels'].count),
                'requires_investigation': agg['alert_count'] > 0 or any(score >= self.threshold for score in threat_scores)
            }

# Entry point for Splunk
if __name__ == "__main__":
    dispatch(InsiderThreatMonitorCommand, sys.argv, sys.stdin, sys.stdout, __name__)
