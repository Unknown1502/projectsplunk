#!/usr/bin/env python3
"""
Splunk Custom Search Command for Insider Threat Risk Scoring
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators

# Add the lib directory to Python path
app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_dir = os.path.join(app_root, 'bin', 'lib')
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

@Configuration()
class InsiderThreatScoreCommand(StreamingCommand):
    """
    Custom Splunk search command for risk scoring based on threat predictions.
    
    Usage:
    | insider_threat_score [score_field="threat_score"] [output_field="risk_category"]
    
    Examples:
    | insider_threat_score
    | insider_threat_score score_field="ml_score" output_field="risk_level"
    """
    
    score_field = Option(
        doc='Field containing the threat score (default: threat_score)',
        require=False,
        default="threat_score"
    )
    
    output_field = Option(
        doc='Output field name for risk category (default: risk_category)',
        require=False,
        default="risk_category"
    )
    
    add_risk_score = Option(
        doc='Add normalized risk score (0-100) (default: true)',
        require=False,
        validate=validators.Boolean(),
        default=True
    )
    
    add_risk_factors = Option(
        doc='Add risk factor analysis (default: true)',
        require=False,
        validate=validators.Boolean(),
        default=True
    )
    
    def stream(self, records):
        """Process each record to add risk scoring."""
        for record in records:
            try:
                # Get threat score
                threat_score = self._get_threat_score(record)
                
                # Calculate risk category
                risk_category = self._calculate_risk_category(threat_score)
                record[self.output_field] = risk_category
                
                # Add normalized risk score if requested
                if self.add_risk_score:
                    record['risk_score'] = round(threat_score * 100, 2)
                
                # Add risk factors if requested
                if self.add_risk_factors:
                    risk_factors = self._analyze_risk_factors(record, threat_score)
                    record.update(risk_factors)
                
                # Add risk metadata
                record['risk_evaluated_at'] = datetime.now().isoformat()
                record['risk_score_version'] = '2.0'
                
                yield record
                
            except Exception as e:
                self.logger.error(f"Error processing record: {str(e)}")
                record['risk_error'] = str(e)
                record[self.output_field] = 'unknown'
                yield record
    
    def _get_threat_score(self, record):
        """Extract threat score from record."""
        # Try to get from specified field
        if self.score_field in record:
            try:
                score = float(record[self.score_field])
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except (ValueError, TypeError):
                pass
        
        # Try common field names
        for field in ['threat_score', 'ml_score', 'risk_score', 'score']:
            if field in record:
                try:
                    score = float(record[field])
                    # Handle different scales
                    if score > 1.0:
                        score = score / 100.0  # Assume percentage
                    return max(0.0, min(1.0, score))
                except (ValueError, TypeError):
                    continue
        
        # Default to 0
        return 0.0
    
    def _calculate_risk_category(self, threat_score):
        """Calculate risk category from threat score."""
        if threat_score >= 0.8:
            return "critical"
        elif threat_score >= 0.6:
            return "high"
        elif threat_score >= 0.4:
            return "medium"
        elif threat_score >= 0.2:
            return "low"
        else:
            return "info"
    
    def _analyze_risk_factors(self, record, threat_score):
        """Analyze and identify risk factors."""
        risk_factors = {
            'risk_factor_count': 0,
            'risk_factors': [],
            'risk_indicators': []
        }
        
        # Time-based risk factors
        if '_time' in record:
            try:
                event_time = pd.to_datetime(record['_time'], unit='s')
                hour = event_time.hour
                
                # After hours activity
                if hour < 6 or hour > 20:
                    risk_factors['risk_factors'].append('after_hours_activity')
                    risk_factors['risk_factor_count'] += 1
                
                # Weekend activity
                if event_time.weekday() >= 5:
                    risk_factors['risk_factors'].append('weekend_activity')
                    risk_factors['risk_factor_count'] += 1
                    
            except:
                pass
        
        # User-based risk factors
        user = record.get('user', record.get('src_user', ''))
        if user:
            # Service accounts
            if any(term in user.lower() for term in ['service', 'svc', 'admin', 'system']):
                risk_factors['risk_factors'].append('privileged_account')
                risk_factors['risk_factor_count'] += 1
        
        # Activity-based risk factors
        action = record.get('action', record.get('activity_type', ''))
        if action:
            action_lower = action.lower()
            
            # High-risk activities
            high_risk_actions = ['delete', 'remove', 'download', 'export', 'copy', 'transfer']
            if any(term in action_lower for term in high_risk_actions):
                risk_factors['risk_factors'].append('high_risk_action')
                risk_factors['risk_factor_count'] += 1
            
            # Authentication events
            auth_actions = ['login', 'logon', 'auth', 'failed']
            if any(term in action_lower for term in auth_actions):
                risk_factors['risk_indicators'].append('authentication_event')
        
        # Volume-based risk factors
        if 'bytes' in record or 'size' in record:
            try:
                bytes_value = float(record.get('bytes', record.get('size', 0)))
                if bytes_value > 1073741824:  # 1GB
                    risk_factors['risk_factors'].append('large_data_transfer')
                    risk_factors['risk_factor_count'] += 1
            except:
                pass
        
        # Source-based risk factors
        src = record.get('src', record.get('src_ip', ''))
        if src:
            # External IPs (simple check)
            if not any(src.startswith(prefix) for prefix in ['10.', '172.', '192.168.']):
                risk_factors['risk_indicators'].append('external_source')
        
        # Threat score based factors
        if threat_score >= 0.8:
            risk_factors['risk_indicators'].append('ml_high_confidence')
        
        # Previous alerts
        if record.get('alert_count', 0) > 0:
            risk_factors['risk_factors'].append('previous_alerts')
            risk_factors['risk_factor_count'] += 1
        
        # Format risk factors for output
        risk_factors['risk_factors_list'] = ', '.join(risk_factors['risk_factors']) if risk_factors['risk_factors'] else 'none'
        risk_factors['risk_indicators_list'] = ', '.join(risk_factors['risk_indicators']) if risk_factors['risk_indicators'] else 'none'
        
        # Calculate adjusted risk score
        base_score = threat_score * 100
        factor_multiplier = 1 + (risk_factors['risk_factor_count'] * 0.1)  # 10% per factor
        risk_factors['adjusted_risk_score'] = round(min(100, base_score * factor_multiplier), 2)
        
        return risk_factors

# Entry point for Splunk
if __name__ == "__main__":
    dispatch(InsiderThreatScoreCommand, sys.argv, sys.stdin, sys.stdout, __name__)
