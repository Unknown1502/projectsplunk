# Insider Threat Detection System - Usage Guide

## 🚀 Quick Start

### 1. Installation
```bash
# Clone/create project directory
cd insider_threat_detection

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

### 2. Basic Usage

#### Train a Model
```bash
# Basic training (now uses LSTM by default)
python main.py train

# Advanced options
python main.py train --epochs 50 --batch-size 32

# Use GRU instead of default LSTM
python main.py train --use-gru --epochs 50
```

#### Evaluate Model
```bash
python main.py evaluate --model-path checkpoints/model_checkpoint.h5
```

#### Make Predictions
```bash
python main.py predict --model-path checkpoints/model_checkpoint.h5 --input-file new_data.csv
```

#### Run Demo
```bash
python main.py demo --quick
```

## 📁 File Structure Overview

```
insider_threat_detection/
├── 📋 main.py                    # Main entry point
├── 📋 requirements.txt           # Dependencies
├── 📋 README.md                  # Comprehensive documentation
├── 📋 setup.py                   # Package setup
├── 📋 .gitignore                 # Git ignore rules
├── 📋 USAGE_GUIDE.md            # This file
│
├── 📁 config/                    # Configuration
│   ├── settings.py              # Global settings
│   └── model_config.py          # Model architecture
│
├── 📁 src/                       # Source code
│   ├── 📁 data/                 # Data processing
│   │   ├── loader.py            # Data loading & merging
│   │   ├── preprocessor.py      # Data preprocessing
│   │   └── feature_engineer.py # Feature engineering
│   │
│   ├── 📁 models/               # Model definitions
│   │   ├── base_model.py        # Abstract base model
│   │   ├── lstm_model.py        # LSTM/GRU models
│   │   └── model_utils.py       # Model utilities
│   │
│   ├── 📁 training/             # Training pipeline
│   │   ├── trainer.py           # Main trainer
│   │   └── callbacks.py         # Custom callbacks
│   │
│   ├── 📁 evaluation/           # Evaluation & visualization
│   │   ├── evaluator.py         # Model evaluation
│   │   └── visualizer.py        # Result visualization
│   │
│   └── 📁 utils/                # Utilities
│       ├── gpu_setup.py         # GPU configuration
│       ├── logger.py            # Logging system
│       └── checkpoint_manager.py # Checkpoint management
│
├── 📁 scripts/                  # Standalone scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── predict.py               # Prediction script
│
├── 📁 tests/                    # Unit tests
│   └── test_data_loader.py      # Data loader tests
│
└── 📁 notebooks/                # Jupyter notebooks
    └── data_exploration.ipynb   # Interactive exploration
```

## 🔧 Configuration

### Data Configuration (`config/settings.py`)
```python
# Update these paths for your environment
DATA_PATH = r"C:\path\to\your\data"
CHECKPOINT_DIR = os.path.join(DATA_PATH, "checkpoints")

# Training parameters (Updated for better performance)
BATCH_SIZE = 16
MAX_EPOCHS = 30
LEARNING_RATE = 0.001              # Increased from 0.0005
LR_REDUCTION_PATIENCE = 5          # Increased from 3
LR_REDUCTION_FACTOR = 0.5          # Increased from 0.3
MIN_LEARNING_RATE = 0.00001        # Minimum learning rate
```

### Model Configuration (`config/model_config.py`)
```python
# Customize model architecture
MODEL_CONFIG = {
    'gru_layers': [
        {'units': 24, 'dropout': 0.4},
        {'units': 12, 'dropout': 0.5}
    ],
    'dense_layers': [
        {'units': 6, 'dropout': 0.6},
        {'units': 3, 'dropout': 0.5}
    ]
}
```

## 📊 Data Format

Your CSV files should have these columns:
- `id`: Unique identifier
- `date`: Timestamp (YYYY-MM-DD HH:MM:SS)
- `user`: User identifier
- `pc`: Computer identifier
- `activity_type`: Activity type (auto-added)
- `details`: Activity details (URL, file path, etc.)

## 🎯 Key Features

### 1. Modular Architecture
- **Separation of Concerns**: Each module has a specific responsibility
- **Easy Maintenance**: Update individual components without affecting others
- **Extensible**: Add new features or models easily

### 2. Advanced ML Pipeline
- **LSTM/GRU Models**: State-of-the-art sequence modeling (LSTM default)
- **Optimized Learning Rate**: Improved learning rate scheduling
- **Feature Engineering**: 18+ sophisticated features
- **Anomaly Detection**: Isolation Forest integration
- **Class Balancing**: Automatic handling of imbalanced data

### 3. Production Ready
- **Checkpointing**: Resume training from interruptions
- **Logging**: Comprehensive logging system
- **Error Handling**: Robust error handling and recovery
- **GPU Support**: Automatic GPU detection and optimization

### 4. Comprehensive Evaluation
- **Multiple Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualizations**: ROC curves, confusion matrices, training plots
- **Threshold Optimization**: Find optimal classification thresholds
- **Model Comparison**: Compare multiple models side-by-side

## 🛠️ Advanced Usage

### Custom Training Pipeline
```python
from src.training.trainer import InsiderThreatTrainer

trainer = InsiderThreatTrainer(data_path="path/to/data")
trainer.batch_size = 32
trainer.max_epochs = 50
results = trainer.run_complete_pipeline()
```

### Custom Model Architecture
```python
from src.models.lstm_model import InsiderThreatLSTM

model = InsiderThreatLSTM(use_gru=True)
model.create_bidirectional_model(input_shape=(8, 18))
```

### Batch Predictions
```python
python scripts/predict.py \
    --model-path model.h5 \
    --input-file large_dataset.csv \
    --batch-size 64 \
    --top-threats 100
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project directory
   cd insider_threat_detection
   python main.py train
   ```

2. **GPU Memory Issues**
   ```python
   # In config/settings.py
   ENABLE_GPU_MEMORY_GROWTH = True
   ```

3. **Data Loading Errors**
   ```bash
   # Check data paths in config/settings.py
   # Verify CSV file formats
   ```

4. **Training Interruption**
   ```bash
   # Resume from checkpoint
   python main.py train --resume
   ```

### Performance Optimization

1. **Increase Batch Size** (if you have enough GPU memory)
2. **Adjust Sequence Length** in config
3. **Tune Learning Rate** and other hyperparameters
4. **Use Mixed Precision** for faster training

## 📈 Performance Expectations

### Typical Results
- **Accuracy**: 85-95%
- **Precision**: 70-90%
- **Recall**: 60-85%
- **F1-Score**: 65-87%
- **ROC-AUC**: 0.80-0.95

### Training Time
- **Small Dataset** (< 10K records): 5-15 minutes
- **Medium Dataset** (10K-100K records): 15-60 minutes
- **Large Dataset** (> 100K records): 1-4 hours

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_data_loader.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 API Reference

### Core Classes
- `InsiderThreatTrainer`: Main training orchestrator
- `ModelEvaluator`: Comprehensive evaluation
- `ModelVisualizer`: Result visualization
- `DataLoader`: Data loading and merging
- `FeatureEngineer`: Advanced feature engineering

### Key Methods
```python
# Training
trainer.run_complete_pipeline()
trainer.prepare_data()
trainer.train_model()

# Evaluation
evaluator.evaluate_model(model, X_test, y_test)
evaluator.compare_models(models_dict)

# Visualization
visualizer.plot_training_history(history)
visualizer.create_comprehensive_report(results)
```

## 🎓 Learning Resources

1. **Start with**: `notebooks/data_exploration.ipynb`
2. **Read**: `README.md` for comprehensive documentation
3. **Try**: `python main.py demo` for quick overview
4. **Explore**: Individual modules in `src/` directory

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

## 📞 Support

1. Check logs in `logs/` directory
2. Review troubleshooting section
3. Run tests to verify installation
4. Check GitHub issues for known problems

---

**Happy Threat Hunting! 🔍🛡️**
