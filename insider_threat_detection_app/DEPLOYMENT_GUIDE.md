# Insider Threat Detection App - Deployment Guide

## Pre-Deployment Checklist

### System Requirements
- [ ] Splunk Enterprise 8.0 or higher
- [ ] Python 3.7+ on Splunk servers
- [ ] 8GB+ RAM for ML operations
- [ ] 50GB+ disk space for models and data

### Python Dependencies
- [ ] TensorFlow 2.x
- [ ] pandas
- [ ] numpy
- [ ] scikit-learn
- [ ] splunk-sdk

## Step-by-Step Deployment

### 1. Prepare the Environment

```bash
# On Splunk server
cd $SPLUNK_HOME/etc/apps/

# Create Python virtual environment (optional but recommended)
python3 -m venv insider_threat_env
source insider_threat_env/bin/activate  # Linux/Mac
# or
insider_threat_env\Scripts\activate  # Windows
```

### 2. Install Python Dependencies

```bash
# Install required packages
pip install tensorflow==2.13.0
pip install pandas numpy scikit-learn
pip install splunk-sdk
```

### 3. Deploy the App

#### Option A: Via Splunk Web
1. Package the app: `tar -czf insider_threat_detection.tgz insider_threat_detection_app/`
2. In Splunk Web: Apps → Manage Apps → Install app from file
3. Upload the package and restart Splunk

#### Option B: Manual Deployment
```bash
# Copy app to Splunk
cp -r insider_threat_detection_app $SPLUNK_HOME/etc/apps/

# Set permissions
chown -R splunk:splunk $SPLUNK_HOME/etc/apps/insider_threat_detection_app
chmod -R 755 $SPLUNK_HOME/etc/apps/insider_threat_detection_app

# Restart Splunk
$SPLUNK_HOME/bin/splunk restart
```

### 4. Initial Configuration

#### Create Indexes
```spl
# In Splunk Web or via CLI
index=insider_threat
index=insider_threat_metrics
```

#### Configure Data Inputs
1. Navigate to Settings → Data Inputs
2. Configure your data sources to use `sourcetype=insider_threat`

### 5. Deploy Models

```bash
# Copy pre-trained models
mkdir -p $SPLUNK_HOME/etc/apps/insider_threat_detection_app/bin/models/
cp your_model.h5 $SPLUNK_HOME/etc/apps/insider_threat_detection_app/bin/models/
cp scaler.pkl $SPLUNK_HOME/etc/apps/insider_threat_detection_app/bin/models/
cp label_encoders.pkl $SPLUNK_HOME/etc/apps/insider_threat_detection_app/bin/models/
```

### 6. Verify Installation

Run test searches:
```spl
# Test command availability
| makeresults | eval user="test", action="login" | insider_threat_predict

# Check logs
index=_internal source=*insider_threat* | head 10
```

## Distributed Deployment

### Search Head Cluster
1. Deploy app to search head deployer
2. Push to cluster members:
```bash
$SPLUNK_HOME/bin/splunk apply shcluster-bundle -target https://sh1:8089
```

### Indexer Cluster
1. Deploy to cluster master
2. Push to indexers:
```bash
$SPLUNK_HOME/bin/splunk apply cluster-bundle
```

### Heavy Forwarders
Deploy full app for preprocessing capabilities

### Universal Forwarders
Only deploy inputs.conf and props.conf

## Post-Deployment Tasks

### 1. Configure Lookups
```spl
| inputlookup insider_threat_users.csv
| inputlookup activity_risk_mappings.csv
```

### 2. Enable Scheduled Searches
1. Navigate to Settings → Searches, reports, and alerts
2. Enable desired scheduled searches
3. Configure alert actions

### 3. Set Up Data Models
```spl
| datamodel Insider_Threat search
```

### 4. Configure KV Store (if needed)
```bash
# In transforms.conf
[insider_threat_kv]
external_type = kvstore
collection = insider_threat_profiles
fields_list = user,risk_score,last_updated
```

## Performance Tuning

### 1. Adjust Python Memory
In `commands.conf`:
```ini
[insider_threat_predict]
maxinputs = 50000
chunked = true
python.version = python3
```

### 2. Configure Parallel Processing
```ini
[parallelization]
max_threads = 4
batch_size = 1000
```

### 3. Optimize Model Loading
- Use model caching
- Implement lazy loading
- Consider model quantization

## Monitoring

### Health Checks
```spl
# App health
| rest /servicesNS/-/insider_threat_detection/apps/local/insider_threat_detection

# Command performance
index=_internal source=*metrics.log* group=searchcommands 
| search command=insider_threat_* 
| stats avg(elapsed_time) by command
```

### Key Metrics to Monitor
- Model prediction latency
- Memory usage
- Error rates
- Data quality scores

## Troubleshooting

### Common Issues

1. **Import Errors**
```bash
# Check Python path
$SPLUNK_HOME/bin/splunk cmd python -c "import sys; print(sys.path)"

# Verify packages
$SPLUNK_HOME/bin/splunk cmd python -c "import tensorflow; print(tensorflow.__version__)"
```

2. **Permission Errors**
```bash
# Fix permissions
chown -R splunk:splunk $SPLUNK_HOME/etc/apps/insider_threat_detection_app
chmod +x $SPLUNK_HOME/etc/apps/insider_threat_detection_app/bin/*.py
```

3. **Memory Issues**
- Increase Splunk memory limits
- Reduce batch sizes
- Enable model quantization

### Debug Mode
Add to `commands.conf`:
```ini
[insider_threat_predict]
enableheader = true
outputheader = true
requires_srinfo = true
stderr_dest = message
```

## Security Hardening

### 1. Restrict Command Access
In `default.meta`:
```ini
[commands/insider_threat_train]
access = read : [ admin ], write : [ admin ]
```

### 2. Secure Model Files
```bash
chmod 600 $SPLUNK_HOME/etc/apps/insider_threat_detection_app/bin/models/*
```

### 3. Enable Audit Logging
```spl
index=_audit action=search command=insider_threat_*
```

## Maintenance

### Regular Tasks
- **Daily**: Monitor error logs
- **Weekly**: Check model performance metrics
- **Monthly**: Retrain models with new data
- **Quarterly**: Review and update risk thresholds

### Backup Procedures
```bash
# Backup models and lookups
tar -czf insider_threat_backup_$(date +%Y%m%d).tgz \
  $SPLUNK_HOME/etc/apps/insider_threat_detection_app/bin/models/ \
  $SPLUNK_HOME/etc/apps/insider_threat_detection_app/lookups/
```

## Upgrade Process

1. Backup current app and models
2. Test new version in dev environment
3. Deploy during maintenance window
4. Verify all commands work
5. Monitor for 24 hours

## Support Contacts

- App Issues: security-team@example.com
- Splunk Support: support@splunk.com
- Documentation: See README.md
