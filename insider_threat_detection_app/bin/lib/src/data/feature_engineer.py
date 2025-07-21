"""Feature engineering utilities for insider threat detection."""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import ANOMALY_CONTAMINATION, THREAT_PERCENTILE
from config.model_config import (
    CATEGORICAL_COLUMNS, LOW_CARDINALITY_COLUMNS, 
    USER_STATS_CONFIG, ANOMALY_FEATURES, THREAT_SCORING_WEIGHTS
)
from src.utils.logger import get_logger
from src.utils.explainability import PeerGroupAnalyzer


class FeatureEngineer:
    """Handles advanced feature engineering for insider threat detection."""
    
    def __init__(self):
        self.logger = get_logger("feature_engineer")
        self.label_encoders = {}
        self.peer_analyzer = PeerGroupAnalyzer()
    
    def create_user_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user behavior analysis features."""
        self.logger.info("Creating user behavior features...")
        
        # Calculate user statistics
        user_stats = df.groupby('user').agg(USER_STATS_CONFIG['aggregations']).reset_index()
        
        # Flatten column names
        user_stats.columns = USER_STATS_CONFIG['column_names']
        
        # Fill NaN values in hour_std
        user_stats['hour_std'] = user_stats['hour_std'].fillna(0)
        
        # Calculate activity diversity (entropy)
        user_stats['activity_entropy'] = user_stats['activity_dist'].apply(
            self._calculate_entropy
        )
        user_stats = user_stats.drop('activity_dist', axis=1)
        
        # Merge back with main dataframe
        df = df.merge(user_stats, on='user', how='left')
        
        self.logger.info("User behavior features created successfully")
        return df
    
    def _calculate_entropy(self, activity_dict: Dict) -> float:
        """Calculate entropy for activity distribution."""
        if not isinstance(activity_dict, dict):
            return 0
        
        total = sum(activity_dict.values())
        if total == 0:
            return 0
        
        entropy = 0
        for count in activity_dict.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables with frequency and label encoding."""
        self.logger.info("Encoding categorical features...")
        
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
                
            # Handle missing values before encoding
            df[col] = df[col].fillna('unknown')
            
            # Frequency encoding for high cardinality features
            freq_encoding = df[col].value_counts().to_dict()
            df[f'{col}_freq'] = df[col].map(freq_encoding)
            
            # Label encoding for low cardinality features
            if col in LOW_CARDINALITY_COLUMNS:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    df[col].astype(str)
                )
        
        self.logger.info("Categorical encoding completed")
        return df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using Isolation Forest."""
        self.logger.info("Performing anomaly detection...")
        
        # Handle missing values in anomaly features
        for col in ANOMALY_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Prepare data for anomaly detection
        available_features = [col for col in ANOMALY_FEATURES if col in df.columns]
        
        if not available_features:
            self.logger.warning("No features available for anomaly detection")
            df['anomaly_score'] = 0
            return df
        
        anomaly_data = df[available_features]
        
        # Check if we have valid data
        if len(anomaly_data) > 0 and not anomaly_data.isna().all().all():
            try:
                iso_forest = IsolationForest(
                    contamination=ANOMALY_CONTAMINATION, 
                    random_state=42
                )
                anomaly_predictions = iso_forest.fit_predict(anomaly_data)
                df['anomaly_score'] = (anomaly_predictions == -1).astype(int)
                
                anomaly_count = df['anomaly_score'].sum()
                self.logger.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection: {e}")
                df['anomaly_score'] = 0
        else:
            self.logger.warning("Unable to perform anomaly detection, setting anomaly_score to 0")
            df['anomaly_score'] = 0
        
        return df
    
    def create_threat_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated threat labels based on multiple factors."""
        self.logger.info("Creating threat labels...")
        
        threat_score = np.zeros(len(df))
        
        # Features to normalize for scoring
        features_to_normalize = ['total_activities', 'unique_pcs', 'off_hours_ratio',
                               'activity_entropy', 'hour_std']
        
        # Normalize features and add to threat score
        for feature in features_to_normalize:
            if feature in df.columns:
                values = df[feature].fillna(0)
                if values.std() > 0:
                    normalized = (values - values.mean()) / values.std()
                    threat_score += np.abs(normalized) * THREAT_SCORING_WEIGHTS['normalized_features_weight']
        
        # Add anomaly score
        if 'anomaly_score' in df.columns:
            threat_score += df['anomaly_score'] * THREAT_SCORING_WEIGHTS['anomaly_score_weight']
        
        # Add specific behavioral patterns
        if 'is_off_hours' in df.columns:
            threat_score += (df['is_off_hours'] == 1).astype(int) * THREAT_SCORING_WEIGHTS['off_hours_weight']
        
        if 'is_weekend' in df.columns:
            threat_score += (df['is_weekend'] == 1).astype(int) * THREAT_SCORING_WEIGHTS['weekend_weight']
        
        # Convert to binary labels (top percentile are threats)
        threshold = np.percentile(threat_score, THREAT_PERCENTILE)
        df['is_threat'] = (threat_score > threshold).astype(int)
        
        threat_ratio = df['is_threat'].mean()
        self.logger.info(f"Threat ratio: {threat_ratio:.3f}")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional advanced features."""
        self.logger.info("Creating advanced features...")
        
        # Time-based patterns
        df['hour_category'] = pd.cut(df['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['night', 'morning', 'afternoon', 'evening'],
                                   include_lowest=True)
        
        # Activity frequency per user per day
        df['date_only'] = df['date'].dt.date
        daily_activity = df.groupby(['user', 'date_only']).size().reset_index(name='daily_activity_count')
        df = df.merge(daily_activity, on=['user', 'date_only'], how='left')
        df = df.drop('date_only', axis=1)
        
        # PC switching behavior
        df['pc_changes'] = df.groupby('user')['pc'].transform(
            lambda x: (x != x.shift()).cumsum()
        )
        
        # Activity type switching
        df['activity_changes'] = df.groupby('user')['activity_type'].transform(
            lambda x: (x != x.shift()).cumsum()
        )
        
        # Time since last activity (in hours)
        df['time_since_last'] = df.groupby('user')['date'].diff().dt.total_seconds() / 3600
        df['time_since_last'] = df['time_since_last'].fillna(0)
        
        self.logger.info("Advanced features created successfully")
        return df
    
    def get_feature_importance_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of feature importance and statistics."""
        numerical_features = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            "total_features": len(df.columns),
            "numerical_features": len(numerical_features),
            "categorical_features": len(df.columns) - len(numerical_features),
            "feature_statistics": {}
        }
        
        # Calculate basic statistics for numerical features
        for col in numerical_features:
            if col in df.columns:
                summary["feature_statistics"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "missing_count": int(df[col].isnull().sum())
                }
        
        return summary
    
    def validate_features(self, df: pd.DataFrame, required_features: List[str]) -> Dict:
        """Validate that all required features are present and valid."""
        validation_results = {
            "missing_features": [],
            "invalid_features": [],
            "feature_quality": {}
        }
        
        for feature in required_features:
            if feature not in df.columns:
                validation_results["missing_features"].append(feature)
            else:
                # Check for high percentage of missing values
                missing_pct = (df[feature].isnull().sum() / len(df)) * 100
                
                # Check for constant values
                is_constant = df[feature].nunique() <= 1
                
                validation_results["feature_quality"][feature] = {
                    "missing_percentage": missing_pct,
                    "is_constant": is_constant,
                    "unique_values": df[feature].nunique()
                }
                
                if missing_pct > 90 or is_constant:
                    validation_results["invalid_features"].append(feature)
        
        validation_results["is_valid"] = (
            len(validation_results["missing_features"]) == 0 and
            len(validation_results["invalid_features"]) == 0
        )
        
        return validation_results
    
    def create_peer_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on peer group analysis."""
        self.logger.info("Creating peer group features...")
        
        try:
            # Build peer groups if not already built
            if not self.peer_analyzer.peer_groups:
                self.peer_analyzer.build_peer_groups(df)
            
            # Calculate peer deviation scores for each user
            peer_deviations = []
            
            for user in df['user'].unique():
                user_data = df[df['user'] == user]
                
                # Calculate user statistics
                user_stats = {
                    'avg_hour': user_data['hour'].mean() if 'hour' in user_data else 12,
                    'weekend_ratio': user_data['is_weekend'].mean() if 'is_weekend' in user_data else 0,
                    'off_hours_ratio': user_data['is_off_hours'].mean() if 'is_off_hours' in user_data else 0,
                    'unique_pcs': user_data['pc'].nunique() if 'pc' in user_data else 1
                }
                
                # Analyze against peers
                analysis = self.peer_analyzer.analyze_against_peers(user, user_stats)
                
                peer_deviations.append({
                    'user': user,
                    'peer_deviation_score': analysis.get('overall_deviation_score', 0),
                    'peer_risk_level': analysis.get('risk_level', 'unknown')
                })
            
            # Convert to DataFrame and merge
            peer_df = pd.DataFrame(peer_deviations)
            df = df.merge(peer_df, on='user', how='left')
            
            # Fill missing values
            df['peer_deviation_score'] = df['peer_deviation_score'].fillna(0)
            df['peer_risk_level'] = df['peer_risk_level'].fillna('unknown')
            
            # Encode peer risk level
            risk_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3, 'unknown': 0}
            df['peer_risk_encoded'] = df['peer_risk_level'].map(risk_mapping)
            
            self.logger.info("Peer group features created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating peer group features: {e}")
            # Add default values if peer analysis fails
            df['peer_deviation_score'] = 0
            df['peer_risk_encoded'] = 0
        
        return df
