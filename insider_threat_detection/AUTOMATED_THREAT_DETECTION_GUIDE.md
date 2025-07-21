# Automated Threat Detection Guide

## Current Status vs Full Automation

### What You Have Now:
- ✅ **Historical data** uploaded to Splunk (5 events with pre-calculated threat scores)
- ✅ Data is searchable in Splunk
- ❌ **NOT automated** - these are just static records

### What You Need for Automation:

## 1. Real-Time Data Ingestion
You need to continuously feed NEW events into Splunk as they happen:
- User login events
- File access events  
- Network activity
- Email activity

## 2. Automatic Threat Prediction
For each new event, the system should:
1. Receive the event
2. Run it through your ML model
3. Calculate threat score
4. Send results to Splunk

## 3. Setting Up Automation

### Option A: Use the Automated Script (Recommended)
```bash
# Run this to start automated monitoring
python automated_splunk_integration.py
```

This will:
- Load your trained model
- Monitor for new events
- Predict threat scores automatically
- Send results to Splunk in real-time

### Option B: Create Splunk Alerts
In Splunk, create alerts for high-risk events:

```spl
# Alert for high-risk threats
index=main sourcetype=insider_threat_realtime threat_score>0.8
| table _time user pc activity_type threat_score risk_category
```

### Option C: Integrate with Your Systems
Connect real data sources:
- Windows Event Logs
- Active Directory logs
- Proxy/Firewall logs
- Email server logs

## 4. Viewing Real-Time Threats

### Current Data (Historical):
```spl
# Your uploaded CSV data
index=main sourcetype=csv
```

### Future Real-Time Data:
```spl
# Automated predictions
index=main sourcetype=insider_threat_realtime

# High-risk users
index=main sourcetype=insider_threat_realtime threat_score>0.8
| stats max(threat_score) as max_risk by user

# Real-time dashboard
index=main sourcetype=insider_threat_realtime
| timechart avg(threat_score) by risk_category
```

## 5. Next Steps for Full Automation

1. **Fix HEC Configuration** (if you want real-time streaming)
   - Enable HEC in Splunk settings
   - Configure token with default index

2. **Connect Real Data Sources**
   - Install Splunk Universal Forwarders on your systems
   - Configure inputs for user activity logs

3. **Run Automated Monitoring**
   ```bash
   python monitor_threats.py
   ```

4. **Create Dashboards**
   - Real-time threat scores
   - User risk profiles
   - Activity patterns

## Summary

**Current State**: You have sample data in Splunk (not automated)

**To Make It Automated**: 
1. Connect real event sources
2. Run the automated prediction script
3. Configure alerts for high-risk events

The system is ready for automation - you just need to connect it to your real data sources!
