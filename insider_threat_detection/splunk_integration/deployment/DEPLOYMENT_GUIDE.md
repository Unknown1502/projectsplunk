# Splunk Enterprise Integration - Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Insider Threat Detection system with Splunk Enterprise.

## Prerequisites

- Splunk Enterprise 8.0+ or Splunk Cloud
- Python 3.7+ with TensorFlow 2.x
- Administrative access to Splunk
- Trained insider threat detection model

## Deployment Options

### Option 1: Custom Search Command (Recommended)

This approach wraps your existing model as a custom Splunk search command.

#### Step 1: Prepare the Splunk App

```bash
# Copy the Splunk app to your Splunk installation
cp -r splunk_integration/splunk_app /opt/splunk/etc/apps/insider_threat_detection

# Set proper permissions
chown -R splunk:splunk /opt/splunk/etc/apps/insider_threat_detection
chmod +x /opt/splunk/etc/apps/insider_threat_detection/bin/*.py
```

#### Step 2: Install Python Dependencies

```bash
# Install required Python packages in Splunk's Python environment
/opt/splunk/bin/splunk cmd python -m pip install tensorflow pandas numpy scikit-learn
```

#### Step 3: Copy Model Files

```bash
# Create models directory in the app
mkdir -p /opt/splunk/etc/apps/insider_threat_detection/models

# Copy your trained model and artifacts
cp r1/checkpoints/model_checkpoint.h5 /opt/splunk/etc/apps/insider_threat_detection/models/
cp r1/checkpoints/scaler.pkl /opt/splunk/etc/apps/insider_threat_detection/models/
cp r1/checkpoints/label_encoders.pkl /opt/splunk/etc/apps/insider_threat_detection/models/
cp r1/checkpoints/feature_columns.pkl /opt/splunk/etc/apps/insider_threat_detection/models/
```

#### Step 4: Copy Search Command

```bash
# Copy the custom search command
cp splunk_integration/search_commands/insider_threat_predict.py /opt/splunk/etc/apps/insider_threat_detection/bin/
```

#### Step 5: Restart Splunk

```bash
/opt/splunk/bin/splunk restart
```

#### Step 6: Test the Integration

```spl
# Test the custom search command
index=main sourcetype=your_data 
| head 10 
| insider_threat_predict model_path="/opt/splunk/etc/apps/insider_threat_detection/models/model_checkpoint.h5" threshold=0.5
```

### Option 2: MLTK Integration

Convert your model to work with Splunk's Machine Learning Toolkit.

#### Step 1: Install MLTK

```bash
# Download and install MLTK from Splunkbase
# https://splunkbase.splunk.com/app/2890/
```

#### Step 2: Create MLTK-Compatible Model

```python
# Use the provided MLTK conversion script
python splunk_integration/mltk/convert_to_mltk.py
```

#### Step 3: Deploy Model in MLTK

```spl
# Train a new model using MLTK syntax
| inputlookup insider_threat_data.csv
| fit RandomForestClassifier threat_score from user pc activity_type into insider_threat_model
```

### Option 3: REST API Integration

Deploy your model as an external service and integrate via REST API.

#### Step 1: Deploy Model as REST API

```bash
# Use the provided Flask API wrapper
python splunk_integration/api/model_api.py
```

#### Step 2: Configure Splunk REST Calls

```spl
# Call external model via REST
| rest url="http://your-model-server:5000/predict" 
  method="POST" 
  body="{\"data\": \"$data$\"}"
```

## Data Preparation

### CIM Field Mapping

Ensure your data follows CIM standards:

```spl
# Map your fields to CIM
| eval user=coalesce(username, userid, user)
| eval src=coalesce(source_ip, src_ip, client_ip)
| eval dest=coalesce(dest_host, destination, pc)
| eval action=coalesce(activity, event_type, action)
```

### Data Ingestion

Configure your data inputs to use the correct sourcetype:

```ini
# inputs.conf
[monitor:///path/to/insider/threat/logs]
sourcetype = insider_threat
index = security
```

## Dashboard Deployment

### Step 1: Import Dashboard

```bash
# Copy dashboard XML to Splunk
cp splunk_integration/dashboards/insider_threat_dashboard.xml /opt/splunk/etc/apps/insider_threat_detection/local/data/ui/views/
```

### Step 2: Configure Permissions

```bash
# Set dashboard permissions in Splunk Web UI
# Settings > User Interface > Views > insider_threat_dashboard
```

## Alert Configuration

### Step 1: Import Alert Configurations

```bash
# Copy alert configurations
cp splunk_integration/alerts/insider_threat_alerts.conf /opt/splunk/etc/apps/insider_threat_detection/local/
```

### Step 2: Configure Email Settings

```ini
# Configure email settings in alert_actions.conf
[email]
server = your-smtp-server.com
from = splunk-alerts@company.com
```

### Step 3: Test Alerts

```spl
# Test alert search manually
index=main sourcetype=insider_threat 
| insider_threat_predict model_path="/opt/splunk/etc/apps/insider_threat_detection/models/model_checkpoint.h5" threshold=0.5 
| where threat_score >= 0.8
```

## Performance Optimization

### Model Optimization

1. **Batch Processing**: Process events in batches for better performance
2. **Model Caching**: Cache model in memory to avoid repeated loading
3. **Feature Selection**: Use only essential features for faster prediction

### Splunk Optimization

1. **Index Optimization**: Use appropriate indexes and sourcetypes
2. **Search Optimization**: Use efficient SPL queries
3. **Resource Allocation**: Allocate sufficient resources for ML processing

## Monitoring and Maintenance

### Health Checks

```spl
# Monitor model performance
| rest /services/apps/local/insider_threat_detection
| eval status=if(disabled=0, "enabled", "disabled")
```

### Model Updates

```bash
# Update model files
cp new_model.h5 /opt/splunk/etc/apps/insider_threat_detection/models/model_checkpoint.h5
/opt/splunk/bin/splunk restart
```

### Log Monitoring

```spl
# Monitor search command logs
index=_internal source=*splunkd.log* component=SearchProcessor insider_threat_predict
```

## Troubleshooting

### Common Issues

1. **Python Import Errors**: Ensure all dependencies are installed in Splunk's Python environment
2. **Model Loading Errors**: Check file paths and permissions
3. **Performance Issues**: Monitor resource usage and optimize queries

### Debug Mode

```python
# Enable debug logging in the search command
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Support

For technical support:
- Check Splunk logs: `index=_internal source=*splunkd.log*`
- Review search command logs
- Contact your Splunk administrator

## Security Considerations

1. **Model Security**: Protect model files with appropriate permissions
2. **Data Privacy**: Ensure compliance with data protection regulations
3. **Access Control**: Implement proper role-based access control
4. **Audit Logging**: Enable audit logging for all ML activities

## Next Steps

1. **Validation**: Test the integration with sample data
2. **Training**: Train security team on new dashboards and alerts
3. **Tuning**: Fine-tune thresholds based on false positive rates
4. **Scaling**: Plan for scaling to handle increased data volumes

## Example SPL Queries

### Basic Threat Detection
```spl
index=security sourcetype=insider_threat 
| insider_threat_predict model_path="/opt/splunk/etc/apps/insider_threat_detection/models/model_checkpoint.h5" threshold=0.5
| where is_threat=true
| table _time user pc activity_type threat_score risk_category
```

### User Risk Analysis
```spl
index=security sourcetype=insider_threat 
| insider_threat_predict model_path="/opt/splunk/etc/apps/insider_threat_detection/models/model_checkpoint.h5" threshold=0.5
| stats avg(threat_score) as avg_risk, max(threat_score) as max_risk, count as events by user
| sort -avg_risk
```

### Temporal Analysis
```spl
index=security sourcetype=insider_threat 
| insider_threat_predict model_path="/opt/splunk/etc/apps/insider_threat_detection/models/model_checkpoint.h5" threshold=0.5
| where is_threat=true
| timechart span=1h count by risk_category
