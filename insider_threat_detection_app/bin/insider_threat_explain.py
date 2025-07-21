#!/usr/bin/env python3
"""
Splunk Custom Search Command for Insider Threat Explainability
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
class InsiderThreatExplainCommand(StreamingCommand):
    """
    Custom Splunk search command for explaining insider threat predictions.
    
    Usage:
    | insider_threat_explain [detail_level="medium"] [max_factors=5]
    
    Examples:
    | insider_threat_explain
    | insider_threat_explain detail_level="high" max_factors=10
    """
    
    detail_level = Option(
        doc='Level of explanation detail: low|medium|high (default: medium)',
        require=False,
        validate=validators.Match("detail_level", r"^(low|medium|high)$"),
        default="medium"
    )
    
    max_factors = Option(
        doc='Maximum number of factors to show (default: 5)',
        require=False,
        validate=validators.Integer(minimum=1, maximum=20),
        default=5
    )
    
    include_recommendations = Option(
        doc='Include security recommendations (default: true)',
        require=False,
        validate=validators.Boolean(),
        default=True
    )
    
    def stream(self, records):
        """Process each record to add explainability."""
        for record in records:
            try:
                # Get threat score and risk level
                threat_score = self._get_threat_score(record)
                risk_level = record.get('risk_category', record.get('risk_level', 'unknown'))
                
                # Generate explanation based on detail level
                if self.detail_level == "low":
                    explanation = self._generate_simple_explanation(record, threat_score, risk_level)
                elif self.detail_level == "high":
                    explanation = self._generate_detailed_explanation(record, threat_score, risk_level)
                else:
                    explanation = self._generate_medium_explanation(record, threat_score, risk_level)
                
                # Add explanation to record
                record.update(explanation)
                
                # Add recommendations if requested
                if self.include_recommendations:
                    recommendations = self._generate_recommendations(record, threat_score, risk_level)
                    record.update(recommendations)
                
                yield record
                
            except Exception as e:
                self.logger.error(f"Error explaining record: {str(e)}")
                record['explanation_error'] = str(e)
                record['threat_explanation'] = "Unable to generate explanation"
                yield record
    
    def _get_threat_score(self, record):
        """Extract threat score from record."""
        for field in ['threat_score', 'ml_score', 'risk_score', 'score']:
            if field in record:
                try:
                    score = float(record[field])
                    if score > 1.0:
                        score = score / 100.0
                    return max(0.0, min(1.0, score))
                except:
                    continue
        return 0.0
    
    def _generate_simple_explanation(self, record, threat_score, risk_level):
        """Generate simple explanation."""
        explanation = {
            'threat_explanation': f"User activity scored {threat_score:.2%} with {risk_level} risk.",
            'explanation_detail': 'simple'
        }
        
        # Add primary reason
        if threat_score >= 0.8:
            explanation['primary_reason'] = "Multiple high-risk indicators detected"
        elif threat_score >= 0.6:
            explanation['primary_reason'] = "Suspicious activity pattern identified"
        elif threat_score >= 0.4:
            explanation['primary_reason'] = "Moderate deviation from normal behavior"
        else:
            explanation['primary_reason'] = "Minor anomaly detected"
        
        return explanation
    
    def _generate_medium_explanation(self, record, threat_score, risk_level):
        """Generate medium-detail explanation."""
        explanation = self._generate_simple_explanation(record, threat_score, risk_level)
        explanation['explanation_detail'] = 'medium'
        
        # Analyze contributing factors
        factors = []
        
        # Time-based factors
        if '_time' in record:
            try:
                event_time = pd.to_datetime(record['_time'], unit='s')
                hour = event_time.hour
                
                if hour < 6 or hour > 20:
                    factors.append({
                        'factor': 'after_hours_activity',
                        'description': f'Activity at {hour:02d}:00 outside business hours',
                        'impact': 'high'
                    })
                
                if event_time.weekday() >= 5:
                    factors.append({
                        'factor': 'weekend_activity',
                        'description': f'Activity on {event_time.strftime("%A")}',
                        'impact': 'medium'
                    })
            except:
                pass
        
        # User behavior factors
        user = record.get('user', record.get('src_user', ''))
        if user:
            if record.get('user_avg_threat_score', 0) > 0.5:
                factors.append({
                    'factor': 'high_average_risk',
                    'description': f'User {user} has elevated average risk score',
                    'impact': 'high'
                })
            
            if record.get('user_risk_trend') == 'increasing':
                factors.append({
                    'factor': 'increasing_risk_trend',
                    'description': 'Risk score trending upward',
                    'impact': 'medium'
                })
        
        # Activity factors
        action = record.get('action', record.get('activity_type', ''))
        if action:
            action_lower = action.lower()
            if any(term in action_lower for term in ['delete', 'remove', 'export']):
                factors.append({
                    'factor': 'sensitive_action',
                    'description': f'Performed sensitive action: {action}',
                    'impact': 'high'
                })
        
        # Volume factors
        if 'bytes' in record:
            try:
                bytes_value = float(record['bytes'])
                if bytes_value > 1073741824:  # 1GB
                    factors.append({
                        'factor': 'large_data_volume',
                        'description': f'Transferred {bytes_value/1073741824:.1f}GB of data',
                        'impact': 'high'
                    })
            except:
                pass
        
        # Sort factors by impact
        impact_order = {'high': 3, 'medium': 2, 'low': 1}
        factors.sort(key=lambda x: impact_order.get(x['impact'], 0), reverse=True)
        
        # Limit to max_factors
        factors = factors[:self.max_factors]
        
        # Format factors for output
        if factors:
            explanation['contributing_factors'] = len(factors)
            explanation['factor_details'] = json.dumps(factors)
            explanation['top_factors'] = ', '.join([f['factor'] for f in factors[:3]])
        else:
            explanation['contributing_factors'] = 0
            explanation['factor_details'] = '[]'
            explanation['top_factors'] = 'none identified'
        
        return explanation
    
    def _generate_detailed_explanation(self, record, threat_score, risk_level):
        """Generate detailed explanation."""
        explanation = self._generate_medium_explanation(record, threat_score, risk_level)
        explanation['explanation_detail'] = 'detailed'
        
        # Add statistical context
        explanation['statistical_context'] = self._generate_statistical_context(record, threat_score)
        
        # Add behavioral analysis
        explanation['behavioral_analysis'] = self._generate_behavioral_analysis(record)
        
        # Add temporal patterns
        explanation['temporal_patterns'] = self._analyze_temporal_patterns(record)
        
        # Generate narrative explanation
        narrative = self._generate_narrative(record, threat_score, risk_level, explanation)
        explanation['narrative_explanation'] = narrative
        
        return explanation
    
    def _generate_statistical_context(self, record, threat_score):
        """Generate statistical context for the threat score."""
        context = []
        
        # Compare to average
        avg_score = record.get('user_avg_threat_score', 0)
        if avg_score > 0:
            deviation = ((threat_score - avg_score) / avg_score) * 100 if avg_score > 0 else 0
            context.append(f"Score is {abs(deviation):.0f}% {'above' if deviation > 0 else 'below'} user average")
        
        # Percentile context
        if threat_score >= 0.9:
            context.append("Score in top 10% of all threats")
        elif threat_score >= 0.75:
            context.append("Score in top 25% of all threats")
        
        # Trend context
        trend = record.get('user_risk_trend', '')
        if trend == 'increasing':
            context.append("Part of an increasing risk pattern")
        elif trend == 'decreasing':
            context.append("Risk decreasing but still elevated")
        
        return '; '.join(context) if context else "No statistical context available"
    
    def _generate_behavioral_analysis(self, record):
        """Analyze user behavior patterns."""
        analysis = []
        
        # Activity diversity
        unique_activities = record.get('unique_activities', 0)
        if unique_activities > 5:
            analysis.append(f"Unusually diverse activity set ({unique_activities} different actions)")
        
        # Host diversity
        unique_hosts = record.get('unique_hosts', 0)
        if unique_hosts > 3:
            analysis.append(f"Activity across multiple systems ({unique_hosts} hosts)")
        
        # Event frequency
        event_count = record.get('event_count', 0)
        if event_count > 100:
            analysis.append(f"High activity volume ({event_count} events)")
        
        return '; '.join(analysis) if analysis else "Normal behavioral patterns"
    
    def _analyze_temporal_patterns(self, record):
        """Analyze temporal patterns in the activity."""
        patterns = []
        
        if '_time' in record:
            try:
                event_time = pd.to_datetime(record['_time'], unit='s')
                
                # Time of day pattern
                hour = event_time.hour
                if 0 <= hour < 6:
                    patterns.append("Late night activity (00:00-06:00)")
                elif 6 <= hour < 9:
                    patterns.append("Early morning activity")
                elif 20 <= hour < 24:
                    patterns.append("Evening activity")
                
                # Day of week pattern
                if event_time.weekday() == 0:
                    patterns.append("Monday - potential catch-up activity")
                elif event_time.weekday() == 4:
                    patterns.append("Friday - potential data exfiltration risk")
                elif event_time.weekday() >= 5:
                    patterns.append("Weekend activity")
                
            except:
                pass
        
        return '; '.join(patterns) if patterns else "No significant temporal patterns"
    
    def _generate_narrative(self, record, threat_score, risk_level, explanation):
        """Generate a narrative explanation."""
        user = record.get('user', 'Unknown user')
        action = record.get('action', record.get('activity_type', 'performed activity'))
        
        narrative_parts = []
        
        # Opening
        narrative_parts.append(f"{user} {action} with a threat score of {threat_score:.2%} ({risk_level} risk).")
        
        # Contributing factors
        if explanation.get('contributing_factors', 0) > 0:
            narrative_parts.append(f"This assessment is based on {explanation['contributing_factors']} risk factors, primarily: {explanation.get('top_factors', 'various indicators')}.")
        
        # Statistical context
        if 'statistical_context' in explanation and explanation['statistical_context'] != "No statistical context available":
            narrative_parts.append(explanation['statistical_context'] + ".")
        
        # Behavioral analysis
        if 'behavioral_analysis' in explanation and explanation['behavioral_analysis'] != "Normal behavioral patterns":
            narrative_parts.append("Behavioral analysis shows: " + explanation['behavioral_analysis'] + ".")
        
        # Temporal patterns
        if 'temporal_patterns' in explanation and explanation['temporal_patterns'] != "No significant temporal patterns":
            narrative_parts.append("Temporal analysis reveals: " + explanation['temporal_patterns'] + ".")
        
        return " ".join(narrative_parts)
    
    def _generate_recommendations(self, record, threat_score, risk_level):
        """Generate security recommendations based on the threat analysis."""
        recommendations = {
            'recommendation_count': 0,
            'recommendations': [],
            'priority_action': '',
            'investigation_required': False
        }
        
        # High-risk recommendations
        if threat_score >= 0.8 or risk_level == "critical":
            recommendations['investigation_required'] = True
            recommendations['priority_action'] = "Immediate investigation required"
            recommendations['recommendations'].extend([
                "Immediately review user's recent activities",
                "Check for data exfiltration attempts",
                "Verify user identity and authorization",
                "Consider temporary access suspension pending investigation"
            ])
        
        # Medium-risk recommendations
        elif threat_score >= 0.6 or risk_level == "high":
            recommendations['investigation_required'] = True
            recommendations['priority_action'] = "Investigate within 24 hours"
            recommendations['recommendations'].extend([
                "Review user's access permissions",
                "Monitor user activity closely for next 48 hours",
                "Check for policy violations",
                "Verify recent system access patterns"
            ])
        
        # Specific recommendations based on factors
        
        # After-hours activity
        if record.get('_time'):
            try:
                event_time = pd.to_datetime(record['_time'], unit='s')
                if event_time.hour < 6 or event_time.hour > 20:
                    recommendations['recommendations'].append("Verify business justification for after-hours access")
            except:
                pass
        
        # Large data transfer
        if 'bytes' in record:
            try:
                bytes_value = float(record['bytes'])
                if bytes_value > 1073741824:  # 1GB
                    recommendations['recommendations'].append("Review data transfer logs and destination")
                    recommendations['recommendations'].append("Verify data classification and handling policies")
            except:
                pass
        
        # Service account activity
        user = record.get('user', record.get('src_user', ''))
        if user and any(term in user.lower() for term in ['service', 'svc', 'admin']):
            recommendations['recommendations'].append("Verify service account usage is automated and expected")
            recommendations['recommendations'].append("Check for interactive service account usage")
        
        # Failed authentication
        action = record.get('action', record.get('activity_type', ''))
        if action and 'fail' in action.lower():
            recommendations['recommendations'].append("Check for brute force attempts")
            recommendations['recommendations'].append("Verify account lockout policies are enforced")
        
        # Limit recommendations
        recommendations['recommendations'] = recommendations['recommendations'][:self.max_factors]
        recommendations['recommendation_count'] = len(recommendations['recommendations'])
        
        # Format for output
        if recommendations['recommendations']:
            recommendations['recommendations_list'] = ' | '.join(recommendations['recommendations'])
        else:
            recommendations['recommendations_list'] = "Continue normal monitoring"
            recommendations['priority_action'] = "No immediate action required"
        
        return recommendations

# Entry point for Splunk
if __name__ == "__main__":
    dispatch(InsiderThreatExplainCommand, sys.argv, sys.stdin, sys.stdout, __name__)
