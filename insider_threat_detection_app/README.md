# Insider Threat Detection App for Splunk

## Overview

The Insider Threat Detection App is an advanced machine learning-powered solution for identifying and preventing insider threats in your organization. Using LSTM/GRU neural networks, this app analyzes user behavior patterns to detect anomalies and potential security risks in real-time.

## Features

- **Machine Learning Models**: Pre-trained LSTM/GRU models for accurate threat detection
- **Real-time Monitoring**: Continuous analysis of user activities
- **Custom SPL Commands**: Five powerful commands for threat analysis
- **Risk Scoring**: Automated risk assessment and categorization
- **Explainable AI**: Detailed explanations for threat predictions
- **Interactive Dashboards**: Comprehensive visualization of threats
- **CIM Compliance**: Full Common Information Model integration

## Installation

1. Download the app package
2. Install via Splunk Web:
   - Navigate to Apps → Manage Apps
   - Click "Install app from file"
   - Upload the app package
3. Restart Splunk if prompted

## Quick Start

### 1. Basic Threat Detection
```spl
index=security | insider_threat_predict
```

### 2. Real-time Monitoring
```spl
index=security | insider_threat_monitor threshold=0.7 window=5m
```

### 3. Risk Scoring
```spl
index=security | insider_threat_score | where risk_category="high"
```

### 4. Threat Explanation
```spl
index=security | insider_threat_predict | insider_threat_explain detail_level="high"
```

## Custom Commands

### insider_threat_predict
Performs ML-based threat prediction on events.

**Syntax:**
```spl
| insider_threat_predict [model_name=<string>] [threshold=<float>] [explain=<bool>]
```

**Parameters:**
- `model_name`: Model to use (default: "latest")
- `threshold`: Threat threshold 0.0-1.0 (default: 0.5)
- `explain`: Include explanations (default: true)

### insider_threat_monitor
Real-time monitoring with aggregation capabilities.

**Syntax:**
```spl
| insider_threat_monitor [threshold=<float>] [window=<string>] [alert_on=<string>]
```

**Parameters:**
- `threshold`: Alert threshold (default: 0.7)
- `window`: Time window (default: "5m")
- `alert_on`: Risk level to alert (default: "high")

### insider_threat_score
Advanced risk scoring with factor analysis.

**Syntax:**
```spl
| insider_threat_score [score_field=<string>] [output_field=<string>]
```

**Parameters:**
- `score_field`: Input score field (default: "threat_score")
- `output_field`: Output category field (default: "risk_category")

### insider_threat_explain
Provides detailed explanations for predictions.

**Syntax:**
```spl
| insider_threat_explain [detail_level=<string>] [max_factors=<int>]
```

**Parameters:**
- `detail_level`: "low", "medium", or "high" (default: "medium")
- `max_factors`: Maximum factors to show (default: 5)

### insider_threat_train
Train new models on your data.

**Syntax:**
```spl
| insider_threat_train [data_source=<string>] [epochs=<int>] [model_name=<string>]
```

**Parameters:**
- `data_source`: "lookup" or "index" (default: "lookup")
- `epochs`: Training epochs (default: 30)
- `model_name`: Output model name (default: auto-generated)

## Data Requirements

### Required Fields
- `user` or `src_user`: User identifier
- `_time`: Event timestamp
- `action` or `activity_type`: Activity performed

### Optional Fields
- `src` or `src_host`: Source host
- `bytes`: Data volume
- `signature` or `details`: Activity details

## Dashboards

### Main Dashboard
Access via **Apps → Insider Threat Detection**

Features:
- Threat Overview (24-hour summary)
- High Risk Users
- Model Performance Metrics
- Threat Score Distribution
- Risk Categories Over Time
- Recent High-Risk Activities

## Alerts

Pre-configured alerts include:
- High Risk Detection
- Anomalous User Behavior
- Model Performance Degradation
- Data Quality Issues

## Configuration

### Model Management
Models are stored in: `$SPLUNK_HOME/etc/apps/insider_threat_detection/bin/models/`

### Lookup Tables
- `insider_threat_users.csv`: User risk profiles
- `activity_risk_mappings.csv`: Activity risk scores

### Index Configuration
Default index: `insider_threat`

To use a different index, modify `inputs.conf` in local directory.

## Best Practices

1. **Data Volume**: Ensure sufficient historical data (30+ days) for accurate predictions
2. **Model Updates**: Retrain models monthly or when accuracy drops below 85%
3. **Threshold Tuning**: Adjust thresholds based on your environment
4. **Alert Fatigue**: Start with higher thresholds and gradually lower them

## Troubleshooting

### Common Issues

1. **No predictions generated**
   - Check if model files exist in `bin/models/`
   - Verify Python dependencies are installed

2. **Low accuracy**
   - Retrain model with recent data
   - Check data quality and completeness

3. **Performance issues**
   - Reduce batch size in real-time monitoring
   - Consider summary indexing for large datasets

### Debug Mode
Enable debug logging:
```spl
index=_internal source=*insider_threat*.log
```

## Security Considerations

- Models and lookups require appropriate permissions
- Sensitive predictions should be restricted to security team
- Regular audit of app usage recommended

## Support

For issues or questions:
1. Check the app logs
2. Review this documentation
3. Contact your Splunk administrator

## Version History

- v2.0.0: Complete rewrite with Splunk app standards
- v1.0.0: Initial release

## License

This app is provided as-is for security operations use.
