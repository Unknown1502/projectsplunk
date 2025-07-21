# Insider Threat Detection Project Structure

## Project File Format and Organization

```
insider_threat_detection/
│
├── 📁 config/                      # Configuration files
│   ├── __init__.py
│   ├── model_config.py            # Model hyperparameters
│   └── settings.py                # General settings
│
├── 📁 src/                        # Source code
│   ├── __init__.py
│   │
│   ├── 📁 data/                   # Data processing modules
│   │   ├── __init__.py
│   │   ├── loader.py              # Data loading utilities
│   │   ├── preprocessor.py       # Data preprocessing
│   │   └── feature_engineer.py   # Feature engineering
│   │
│   ├── 📁 models/                 # ML models
│   │   ├── __init__.py
│   │   ├── base_model.py         # Base model class
│   │   ├── lstm_model.py         # LSTM implementation
│   │   └── model_utils.py        # Model utilities
│   │
│   ├── 📁 training/               # Training modules
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training logic
│   │   └── callbacks.py          # Training callbacks
│   │
│   ├── 📁 evaluation/             # Model evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py          # Evaluation metrics
│   │   └── visualizer.py         # Visualization tools
│   │
│   └── 📁 utils/                  # Utility modules
│       ├── __init__.py
│       ├── logger.py             # Logging utilities
│       ├── gpu_setup.py          # GPU configuration
│       ├── console_utils.py      # Console utilities
│       ├── checkpoint_manager.py # Model checkpointing
│       └── explainability.py     # Model explainability
│
├── 📁 scripts/                    # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── predict.py                # Prediction script
│
├── 📁 splunk_integration/         # Splunk integration
│   ├── README.md
│   ├── SPLUNK_INTEGRATION_SUMMARY.md
│   │
│   ├── 📁 splunk_app/            # Splunk app structure
│   │   └── 📁 default/
│   │       ├── app.conf          # App configuration
│   │       └── commands.conf     # Custom commands
│   │
│   ├── 📁 alerts/                # Alert configurations
│   │   └── insider_threat_alerts.conf
│   │
│   ├── 📁 dashboards/            # Splunk dashboards
│   │   └── insider_threat_dashboard.xml
│   │
│   ├── 📁 search_commands/       # Custom search commands
│   │   └── insider_threat_predict.py
│   │
│   ├── 📁 cim_mapping/           # CIM field mappings
│   │   └── cim_field_mappings.conf
│   │
│   ├── 📁 deployment/            # Deployment guides
│   │   └── DEPLOYMENT_GUIDE.md
│   │
│   └── 📁 mltk/                  # ML Toolkit integration
│       └── mltk_searches.spl
│
├── 📁 tests/                      # Test files
│   ├── __init__.py
│   └── test_data_loader.py
│
├── 📁 notebooks/                  # Jupyter notebooks
│   └── data_exploration.ipynb
│
├── 📁 logs/                       # Log files (generated)
│
├── 📁 evaluation_results/         # Evaluation outputs (generated)
│
├── 📁 r1/                         # Data and checkpoints
│   ├── 📁 checkpoints/           # Model checkpoints
│   ├── device.csv                # Data files
│   ├── http.csv
│   ├── logon.csv
│   └── 📁 LDAP/
│
├── 📄 main.py                    # Main entry point
├── 📄 predict_realtime.py        # Real-time prediction
├── 📄 monitor_threats.py         # Threat monitoring
├── 📄 setup.py                   # Package setup
├── 📄 requirements.txt           # Python dependencies
├── 📄 .gitignore                # Git ignore file
├── 📄 README.md                  # Project documentation
│
├── 📄 splunk_config.py          # Splunk configuration
├── 📄 splunk_credentials.json   # Splunk credentials
│
├── 🔧 Splunk Integration Scripts
├── 📄 automated_splunk_integration.py
├── 📄 automated_splunk_integration_fixed.py
├── 📄 fix_hec_token.py
├── 📄 fix_splunk_integration_complete.py
├── 📄 quick_hec_fix.py
├── 📄 working_hec_sender.py
│
├── 📋 Batch Scripts
├── 📄 move_to_splunk.bat        # Move project to Splunk
├── 📄 run_complete_fix.bat      # Run all fixes
├── 📄 COMPLETE_FIX_STEPS.bat    # Step-by-step fix guide
├── 📄 run_project.bat           # Run project
├── 📄 run_project.ps1           # PowerShell runner
│
└── 📚 Documentation
    ├── 📄 MOVE_TO_SPLUNK_AND_FIX.md
    ├── 📄 AUTOMATED_THREAT_DETECTION_GUIDE.md
    ├── 📄 COMPLETE_EXECUTION_GUIDE.md
    ├── 📄 PROJECT_EXECUTION_GUIDE.md
    ├── 📄 SPLUNK_INTEGRATION_GUIDE.md
    ├── 📄 USAGE_GUIDE.md
    └── 📄 ENHANCEMENT_SUMMARY.md

## File Types and Purposes

### Python Files (.py)
- **Core modules**: In `src/` directory
- **Scripts**: Executable files in `scripts/`
- **Integration**: Splunk integration scripts
- **Tests**: Unit tests in `tests/`

### Configuration Files
- **.json**: `splunk_credentials.json` - Stores credentials
- **.conf**: Splunk configuration files
- **.xml**: Splunk dashboard definitions

### Documentation Files (.md)
- **README.md**: Main project documentation
- **Guide files**: Step-by-step instructions
- **Summary files**: Feature and integration summaries

### Batch Files (.bat)
- **Automation scripts**: For Windows execution
- **Setup scripts**: For deployment and configuration

### Data Files
- **.csv**: Training and test data in `r1/`
- **.h5**: Model checkpoints
- **.pkl**: Preprocessor objects (scalers, encoders)

## Splunk App Structure (After Moving)

When moved to Splunk using `move_to_splunk.bat`, the structure becomes:

```
C:\Program Files\Splunk\etc\apps\insider_threat_detection\
├── default/                      # Splunk app defaults
│   ├── app.conf                 # App configuration
│   └── inputs.conf              # Input configuration
├── local/                       # Local overrides
├── bin/                         # Executable scripts
├── lookups/                     # Lookup files
└── [all project files]          # Your project files
```

## Key Integration Points

1. **HEC Integration**: Port 8088 for event collection
2. **Management API**: Port 8089 for configuration
3. **Web Interface**: Port 8000 for UI access
4. **Model Path**: `models/` directory for ML models
5. **Data Path**: `r1/` directory for training data

## Environment Requirements

- Python 3.8+
- TensorFlow 2.x
- Splunk Enterprise 8.x
- Required Python packages in `requirements.txt`
