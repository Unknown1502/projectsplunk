"""Explainability utilities for insider threat detection."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import shap
from lime.lime_tabular import LimeTabularExplainer
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger


class ThreatExplainer:
    """Provides explainability for threat predictions."""
    
    def __init__(self, model, feature_names: List[str], training_data: np.ndarray = None):
        """
        Initialize the explainer.
        
        Args:
            model: The trained model
            feature_names: List of feature names
            training_data: Training data for SHAP/LIME (optional)
        """
        self.logger = get_logger("threat_explainer")
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Initialize explainers if training data is provided
        if training_data is not None:
            self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers."""
        try:
            # Initialize SHAP explainer
            self.logger.info("Initializing SHAP explainer...")
            # Use DeepExplainer for neural networks
            self.shap_explainer = shap.DeepExplainer(
                self.model,
                self.training_data[:100]  # Use subset for efficiency
            )
            
            # Initialize LIME explainer
            self.logger.info("Initializing LIME explainer...")
            self.lime_explainer = LimeTabularExplainer(
                self.training_data,
                feature_names=self.feature_names,
                class_names=['Normal', 'Threat'],
                mode='classification'
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing explainers: {e}")
    
    def explain_prediction(self, instance: np.ndarray, method: str = 'simple') -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            instance: Single instance to explain (shape: [sequence_length, features])
            method: 'simple', 'shap', or 'lime'
            
        Returns:
            Dictionary with explanation details
        """
        try:
            if method == 'simple':
                return self._simple_explanation(instance)
            elif method == 'shap' and self.shap_explainer:
                return self._shap_explanation(instance)
            elif method == 'lime' and self.lime_explainer:
                return self._lime_explanation(instance)
            else:
                return self._simple_explanation(instance)
                
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {e}")
            return self._simple_explanation(instance)
    
    def _simple_explanation(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Provide simple rule-based explanation.
        
        This method analyzes the input features and identifies
        which ones are most likely contributing to the threat score.
        """
        # Get the latest timestep (most recent activity)
        latest_features = instance[-1] if len(instance.shape) > 1 else instance
        
        # Calculate feature importance based on deviation from normal
        feature_importance = {}
        explanations = []
        
        # Define normal ranges for key features
        normal_ranges = {
            'hour': (9, 17),  # Normal working hours
            'off_hours_ratio': (0, 0.2),  # Low off-hours activity
            'weekend_ratio': (0, 0.1),  # Low weekend activity
            'anomaly_score': (0, 0),  # No anomalies
            'unique_pcs': (1, 2),  # Usually 1-2 PCs
            'activity_entropy': (0.5, 2.0),  # Moderate diversity
            'threat_score': (0, 0.3)  # Low threat score
        }
        
        # Analyze each feature
        for i, feature_name in enumerate(self.feature_names):
            if i < len(latest_features):
                value = latest_features[i]
                
                # Check if feature is outside normal range
                if feature_name in normal_ranges:
                    min_val, max_val = normal_ranges[feature_name]
                    if value < min_val or value > max_val:
                        deviation = abs(value - (min_val + max_val) / 2)
                        feature_importance[feature_name] = deviation
                        
                        # Generate human-readable explanation
                        if feature_name == 'hour' and (value < 9 or value > 17):
                            explanations.append(f"Activity detected outside normal hours ({value}:00)")
                        elif feature_name == 'off_hours_ratio' and value > 0.2:
                            explanations.append(f"High off-hours activity ratio: {value:.2%}")
                        elif feature_name == 'weekend_ratio' and value > 0.1:
                            explanations.append(f"Elevated weekend activity: {value:.2%}")
                        elif feature_name == 'anomaly_score' and value > 0:
                            explanations.append("Anomalous behavior pattern detected")
                        elif feature_name == 'unique_pcs' and value > 2:
                            explanations.append(f"Multiple PCs accessed: {int(value)} devices")
                        elif feature_name == 'activity_entropy' and value > 2.0:
                            explanations.append("Unusual diversity in activity patterns")
        
        # Sort features by importance
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate risk factors
        risk_factors = []
        if any('hour' in f[0] for f in top_features):
            risk_factors.append("Unusual access times")
        if any('pc' in f[0] for f in top_features):
            risk_factors.append("Multiple device usage")
        if any('entropy' in f[0] for f in top_features):
            risk_factors.append("Irregular activity patterns")
        if any('anomaly' in f[0] for f in top_features):
            risk_factors.append("Statistical anomaly detected")
        
        return {
            'method': 'rule-based',
            'top_features': [{'feature': f[0], 'importance': f[1]} for f in top_features],
            'explanations': explanations,
            'risk_factors': risk_factors,
            'confidence': 'high' if len(explanations) >= 3 else 'medium'
        }
    
    def _shap_explanation(self, instance: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP-based explanation."""
        try:
            # Reshape for SHAP if needed
            if len(instance.shape) == 2:
                instance = instance.reshape(1, -1)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(instance)
            
            # Get feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            feature_importance = np.abs(shap_values[0])
            top_indices = np.argsort(feature_importance)[-5:][::-1]
            
            top_features = []
            explanations = []
            
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    importance = feature_importance[idx]
                    value = instance[0][idx]
                    
                    top_features.append({
                        'feature': feature_name,
                        'importance': float(importance),
                        'value': float(value),
                        'impact': 'increases' if shap_values[0][idx] > 0 else 'decreases'
                    })
                    
                    explanations.append(
                        f"{feature_name} = {value:.2f} {top_features[-1]['impact']} threat probability"
                    )
            
            return {
                'method': 'shap',
                'top_features': top_features,
                'explanations': explanations,
                'confidence': 'high'
            }
            
        except Exception as e:
            self.logger.error(f"SHAP explanation failed: {e}")
            return self._simple_explanation(instance)
    
    def _lime_explanation(self, instance: np.ndarray) -> Dict[str, Any]:
        """Generate LIME-based explanation."""
        try:
            # Flatten instance for LIME
            if len(instance.shape) > 1:
                instance_flat = instance.flatten()
            else:
                instance_flat = instance
            
            # Create prediction function wrapper
            def predict_proba(X):
                # Reshape back to original shape if needed
                if len(instance.shape) > 1:
                    seq_len = instance.shape[0]
                    feat_len = instance.shape[1]
                    X_reshaped = X.reshape(-1, seq_len, feat_len)
                else:
                    X_reshaped = X
                
                predictions = self.model.predict(X_reshaped)
                # Return probabilities for both classes
                return np.column_stack([1 - predictions, predictions])
            
            # Generate explanation
            exp = self.lime_explainer.explain_instance(
                instance_flat,
                predict_proba,
                num_features=5
            )
            
            # Extract top features
            top_features = []
            explanations = []
            
            for feature_idx, importance in exp.as_list():
                # Parse feature index from LIME format
                if '<=' in str(feature_idx):
                    parts = str(feature_idx).split()
                    if len(parts) > 0 and parts[0].isdigit():
                        idx = int(parts[0])
                        if idx < len(self.feature_names):
                            feature_name = self.feature_names[idx]
                            top_features.append({
                                'feature': feature_name,
                                'importance': abs(importance),
                                'impact': 'increases' if importance > 0 else 'decreases'
                            })
                            explanations.append(f"{feature_name} {top_features[-1]['impact']} threat probability")
            
            return {
                'method': 'lime',
                'top_features': top_features,
                'explanations': explanations,
                'confidence': 'medium'
            }
            
        except Exception as e:
            self.logger.error(f"LIME explanation failed: {e}")
            return self._simple_explanation(instance)
    
    def generate_threat_report(self, instance: np.ndarray, prediction: float, 
                             user_info: Dict = None) -> str:
        """
        Generate a human-readable threat report.
        
        Args:
            instance: Input features
            prediction: Threat score
            user_info: Additional user information
            
        Returns:
            Formatted threat report
        """
        explanation = self.explain_prediction(instance)
        
        report = f"""
INSIDER THREAT ANALYSIS REPORT
==============================

Threat Score: {prediction:.2%}
Risk Level: {'HIGH' if prediction > 0.7 else 'MEDIUM' if prediction > 0.4 else 'LOW'}
Analysis Method: {explanation['method'].upper()}

KEY RISK INDICATORS:
"""
        
        for i, exp in enumerate(explanation['explanations'][:5], 1):
            report += f"{i}. {exp}\n"
        
        if explanation.get('risk_factors'):
            report += "\nRISK FACTORS:\n"
            for factor in explanation['risk_factors']:
                report += f"• {factor}\n"
        
        if explanation.get('top_features'):
            report += "\nTOP CONTRIBUTING FEATURES:\n"
            for feat in explanation['top_features'][:3]:
                report += f"• {feat['feature']}: "
                if 'importance' in feat:
                    report += f"importance score = {feat['importance']:.3f}"
                if 'impact' in feat:
                    report += f" ({feat['impact']} risk)"
                report += "\n"
        
        if user_info:
            report += f"\nUSER INFORMATION:\n"
            report += f"User: {user_info.get('user', 'Unknown')}\n"
            report += f"Department: {user_info.get('department', 'Unknown')}\n"
            report += f"Last Activity: {user_info.get('last_activity', 'Unknown')}\n"
        
        report += f"\nCONFIDENCE: {explanation.get('confidence', 'medium').upper()}\n"
        report += "=" * 30
        
        return report


class PeerGroupAnalyzer:
    """Analyzes user behavior against peer groups."""
    
    def __init__(self):
        self.logger = get_logger("peer_group_analyzer")
        self.peer_groups = {}
        self.group_statistics = {}
    
    def build_peer_groups(self, df: pd.DataFrame, group_by: List[str] = None) -> None:
        """
        Build peer groups from historical data.
        
        Args:
            df: DataFrame with user activity data
            group_by: Columns to group by (e.g., ['department', 'role'])
        """
        if group_by is None:
            # Default grouping by activity patterns
            group_by = ['activity_type']
        
        self.logger.info(f"Building peer groups by: {group_by}")
        
        try:
            # Calculate user statistics
            user_stats = df.groupby('user').agg({
                'hour': ['mean', 'std'],
                'is_weekend': 'mean',
                'is_off_hours': 'mean',
                'pc': 'nunique',
                'activity_type': lambda x: x.value_counts().to_dict()
            }).reset_index()
            
            # Flatten column names
            user_stats.columns = ['user', 'avg_hour', 'hour_std', 
                                 'weekend_ratio', 'off_hours_ratio', 
                                 'unique_pcs', 'activity_distribution']
            
            # Create peer groups using clustering
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for clustering
            clustering_features = ['avg_hour', 'hour_std', 'weekend_ratio', 
                                 'off_hours_ratio', 'unique_pcs']
            
            X = user_stats[clustering_features].fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Determine optimal number of clusters
            n_clusters = min(5, len(user_stats) // 10)  # Max 5 groups
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            user_stats['peer_group'] = kmeans.fit_predict(X_scaled)
            
            # Store peer groups
            for _, row in user_stats.iterrows():
                self.peer_groups[row['user']] = row['peer_group']
            
            # Calculate group statistics
            for group_id in range(n_clusters):
                group_data = user_stats[user_stats['peer_group'] == group_id]
                
                self.group_statistics[group_id] = {
                    'size': len(group_data),
                    'avg_hour_mean': group_data['avg_hour'].mean(),
                    'avg_hour_std': group_data['avg_hour'].std(),
                    'weekend_ratio_mean': group_data['weekend_ratio'].mean(),
                    'weekend_ratio_std': group_data['weekend_ratio'].std(),
                    'off_hours_ratio_mean': group_data['off_hours_ratio'].mean(),
                    'off_hours_ratio_std': group_data['off_hours_ratio'].std(),
                    'unique_pcs_mean': group_data['unique_pcs'].mean(),
                    'unique_pcs_std': group_data['unique_pcs'].std()
                }
            
            self.logger.info(f"Created {n_clusters} peer groups")
            
        except Exception as e:
            self.logger.error(f"Error building peer groups: {e}")
    
    def analyze_against_peers(self, user: str, user_stats: Dict) -> Dict[str, Any]:
        """
        Analyze user behavior against their peer group.
        
        Args:
            user: User identifier
            user_stats: Dictionary of user statistics
            
        Returns:
            Analysis results with deviation scores
        """
        try:
            # Get user's peer group
            peer_group = self.peer_groups.get(user, 0)
            group_stats = self.group_statistics.get(peer_group, {})
            
            if not group_stats:
                return {'error': 'No peer group statistics available'}
            
            deviations = {}
            anomalies = []
            
            # Calculate deviations for each metric
            metrics = [
                ('avg_hour', 'Average Hour'),
                ('weekend_ratio', 'Weekend Activity'),
                ('off_hours_ratio', 'Off-Hours Activity'),
                ('unique_pcs', 'Device Usage')
            ]
            
            for metric, display_name in metrics:
                if metric in user_stats and f'{metric}_mean' in group_stats:
                    user_value = user_stats[metric]
                    group_mean = group_stats[f'{metric}_mean']
                    group_std = group_stats.get(f'{metric}_std', 1)
                    
                    # Calculate z-score
                    if group_std > 0:
                        z_score = (user_value - group_mean) / group_std
                    else:
                        z_score = 0
                    
                    deviations[metric] = {
                        'user_value': user_value,
                        'group_mean': group_mean,
                        'z_score': z_score,
                        'deviation_level': self._classify_deviation(z_score)
                    }
                    
                    # Flag significant anomalies
                    if abs(z_score) > 2:
                        anomalies.append(
                            f"{display_name}: {user_value:.2f} "
                            f"(group average: {group_mean:.2f}, "
                            f"{'above' if z_score > 0 else 'below'} normal)"
                        )
            
            # Calculate overall peer deviation score
            deviation_scores = [abs(d['z_score']) for d in deviations.values()]
            overall_deviation = np.mean(deviation_scores) if deviation_scores else 0
            
            return {
                'peer_group': peer_group,
                'group_size': group_stats.get('size', 0),
                'deviations': deviations,
                'anomalies': anomalies,
                'overall_deviation_score': overall_deviation,
                'risk_level': self._classify_risk(overall_deviation),
                'recommendation': self._generate_recommendation(overall_deviation, anomalies)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing against peers: {e}")
            return {'error': str(e)}
    
    def _classify_deviation(self, z_score: float) -> str:
        """Classify deviation level based on z-score."""
        abs_z = abs(z_score)
        if abs_z < 1:
            return 'normal'
        elif abs_z < 2:
            return 'moderate'
        elif abs_z < 3:
            return 'significant'
        else:
            return 'extreme'
    
    def _classify_risk(self, deviation_score: float) -> str:
        """Classify risk level based on overall deviation."""
        if deviation_score < 1:
            return 'low'
        elif deviation_score < 2:
            return 'medium'
        elif deviation_score < 3:
            return 'high'
        else:
            return 'critical'
    
    def _generate_recommendation(self, deviation_score: float, anomalies: List[str]) -> str:
        """Generate recommendation based on analysis."""
        if deviation_score < 1:
            return "User behavior is consistent with peer group. Continue normal monitoring."
        elif deviation_score < 2:
            return "Minor deviations detected. Increase monitoring frequency."
        elif deviation_score < 3:
            return "Significant deviations from peer group. Investigate recent activities."
        else:
            return "Critical deviation from peer behavior. Immediate investigation required."
