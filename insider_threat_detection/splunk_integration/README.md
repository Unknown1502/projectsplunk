# Insider Threat Detection - Splunk Enterprise Integration

This directory contains all the components needed to integrate the insider threat detection system with Splunk Enterprise.

## Integration Approaches

### 1. Direct MLTK Integration (Recommended)
- Convert existing model to work within Splunk's Machine Learning Toolkit
- Use MLTK's supported algorithms and SPL commands
- Native Splunk integration with built-in visualization

### 2. Custom Search Commands
- Wrap existing model as custom Splunk search command
- Keep existing code structure while making it Splunk-compatible
- Python script that Splunk calls directly

### 3. External Model Integration
- Use Splunk's REST API endpoints to send data to external model
- Return results back to Splunk for visualization and alerting
- Maintains model architecture intact

## Directory Structure

```
splunk_integration/
├── README.md                           # This file
├── splunk_app/                         # Complete Splunk app package
│   ├── default/                        # App configuration
│   ├── bin/                           # Custom search commands
│   ├── lookups/                       # Lookup tables
│   └── local/                         # Local configurations
├── cim_mapping/                       # CIM data model mappings
├── search_commands/                   # Custom search command implementations
├── dashboards/                        # Splunk dashboard definitions
├── alerts/                           # Alert configurations
└── deployment/                       # Deployment scripts and guides
```

## Quick Start

1. **Install the Splunk App**: Copy `splunk_app/` to your Splunk apps directory
2. **Configure CIM Mapping**: Use the provided CIM field mappings
3. **Deploy Custom Commands**: Install the custom search commands
4. **Import Dashboards**: Load the pre-built dashboards
5. **Set Up Alerts**: Configure real-time threat detection alerts

## Next Steps

See individual component README files for detailed implementation instructions.
