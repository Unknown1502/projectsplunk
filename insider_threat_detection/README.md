# Insider Threat Detection System

A comprehensive machine learning system for detecting insider threats using LSTM/GRU neural networks. This system analyzes user behavior patterns from multiple data sources to identify potential security threats.

## 🚀 Features

- **Advanced Neural Networks**: LSTM/GRU models with attention mechanisms
- **Multi-source Data Integration**: HTTP logs, device logs, authentication logs
- **Sophisticated Feature Engineering**: Time-based patterns, user behavior analysis, anomaly detection
- **Robust Training Pipeline**: Checkpointing, early stopping, learning rate scheduling
- **Comprehensive Evaluation**: ROC curves, precision-recall analysis, confusion matrices
- **Production-Ready**: Modular architecture, logging, error handling
- **GPU Support**: Automatic GPU detection and mixed precision training

## 📁 Project Structure

```
insider_threat_detection/
├── config/                     # Configuration files
│   ├── settings.py            # Global settings
│   └── model_config.py        # Model architecture config
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   │   ├── loader.py          # Data loading and merging
│   │   ├── preprocessor.py    # Data preprocessing
│   │   └── feature_engineer.py # Feature engineering
│   ├── models/                # Model definitions
│   │   ├── base_model.py      # Abstract base model
│   │   ├── lstm_model.py      # LSTM/GRU implementation
│   │   └── model_utils.py     # Model utilities
│   ├── training/              # Training pipeline
│   │   ├── trainer.py         # Main training orchestrator
│   │   └── callbacks.py       # Custom training callbacks
│   ├── evaluation/            # Evaluation and visualization
│   │   ├── evaluator.py       # Model evaluation
│   │   └── visualizer.py      # Result visualization
│   └── utils/                 # Utility modules
│       ├── gpu_setup.py       # GPU configuration
│       ├── logger.py          # Logging utilities
│       └── checkpoint_manager.py # Checkpoint management
├── scripts/                   # Standalone scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── predict.py            # Prediction script
├── tests/                     # Unit tests
├── notebooks/                 # Jupyter notebooks
├── main.py                   # Main entry point
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone or create the project directory:**
   ```bash
   mkdir insider_threat_detection
   cd insider_threat_detection
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

## 📊 Data Format

The system expects CSV files with the following structure:

### Required Columns
- `id`: Unique identifier
- `date`: Timestamp (YYYY-MM-DD HH:MM:SS)
- `user`: User identifier
- `pc`: Computer/device identifier
- `activity_type`: Type of activity (HTTP, DEVICE, LOGON)
- `details`: Activity details (URL, file path, etc.)

### Example Data Files
- `http.csv`: Web browsing logs
- `device.csv`: Device usage logs  
- `logon.csv`: Authentication logs

## 🚀 Quick Start

### 1. Training a Model

```bash
# Basic training
python main.py train

# Advanced training options
python main.py train --use-lstm --epochs 50 --batch-size 32

# Resume from checkpoint
python main.py train --resume
```

### 2. Evaluating a Model

```bash
# Basic evaluation
python main.py evaluate --model-path path/to/model.h5

# Detailed evaluation with plots
python scripts/evaluate.py --model-path path/to/model.h5 --create-plots --optimize-threshold
```

### 3. Making Predictions

```bash
# Predict on new data
python main.py predict --model-path path/to/model.h5 --input-file new_data.csv

# Get top threats only
python scripts/predict.py --model-path path/to/model.h5 --input-file data.csv --top-threats 10
```

### 4. Running Complete Demo

```bash
# Full demo pipeline
python main.py demo

# Quick demo (5 epochs)
python main.py demo --quick
```

## ⚙️ Configuration

### Model Configuration (`config/model_config.py`)

```python
MODEL_CONFIG = {
    'gru_layers': [
        {'units': 24, 'dropout': 0.4, 'recurrent_dropout': 0.4},
        {'units': 12, 'dropout': 0.5, 'recurrent_dropout': 0.5}
    ],
    'dense_layers': [
        {'units': 6, 'dropout': 0.6},
        {'units': 3, 'dropout': 0.5}
    ]
}
```

### Global Settings (`config/settings.py`)

```python
# Data paths
DATA_PATH = r"C:\path\to\your\data"

# Training parameters
BATCH_SIZE = 16
MAX_EPOCHS = 30
LEARNING_RATE = 0.0005

# Feature engineering
SEQUENCE_LENGTH = 8
ANOMALY_CONTAMINATION = 0.1
```

## 📈 Model Architecture

The system uses a sophisticated neural network architecture:

1. **Input Layer**: Sequences of user activities
2. **Recurrent Layers**: GRU/LSTM with dropout and regularization
3. **Batch Normalization**: For training stability
4. **Dense Layers**: Fully connected layers with dropout
5. **Output Layer**: Sigmoid activation for binary classification

### Key Features:
- **Sequence Length**: 8 timesteps per sequence
- **Feature Count**: 18+ engineered features
- **Regularization**: L1/L2 regularization, dropout, batch normalization
- **Class Balancing**: Automatic class weight calculation

## 🔧 Advanced Usage

### Custom Training Pipeline

```python
from src.training.trainer import InsiderThreatTrainer

# Initialize trainer
trainer = InsiderThreatTrainer(data_path="path/to/data", use_gru=True)

# Customize training parameters
trainer.batch_size = 32
trainer.max_epochs = 50

# Run training
results = trainer.run_complete_pipeline()
```

### Custom Model Architecture

```python
from src.models.lstm_model import InsiderThreatLSTM

# Create model with custom architecture
model = InsiderThreatLSTM(use_gru=True)

# Build bidirectional model
model.create_bidirectional_model(input_shape=(8, 18))

# Build attention model
model.create_attention_model(input_shape=(8, 18))
```

### Evaluation and Visualization

```python
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import ModelVisualizer

# Evaluate model
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test)

# Create visualizations
visualizer = ModelVisualizer()
visualizer.plot_training_history(training_history)
visualizer.plot_roc_curves(evaluation_results)
```

## 📊 Performance Metrics

The system provides comprehensive evaluation metrics:

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predictions
- **Recall**: True positive rate among actual threats
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve
- **PR AUC**: Area under Precision-Recall curve

### Advanced Analysis
- **Confusion Matrix**: Detailed classification breakdown
- **Threshold Analysis**: Performance across different thresholds
- **Prediction Confidence**: Distribution of prediction probabilities
- **Feature Importance**: Impact of different features

## 🔍 Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   ```python
   # Enable memory growth
   ENABLE_GPU_MEMORY_GROWTH = True
   ```

2. **Data Loading Errors**
   ```bash
   # Check data file paths and formats
   python -c "from src.data.loader import DataLoader; loader = DataLoader(); loader.load_and_merge_data()"
   ```

3. **Training Interruption**
   ```bash
   # Resume from checkpoint
   python main.py train --resume
   ```

4. **Low Performance**
   - Increase sequence length
   - Add more features
   - Tune hyperparameters
   - Collect more training data

### Logging

All components use comprehensive logging:

```bash
# Check logs directory
ls logs/

# View training logs
tail -f logs/training_*.log
```

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_data_loader.py -v
```

## 📚 API Reference

### Core Classes

- **`InsiderThreatTrainer`**: Main training orchestrator
- **`ModelEvaluator`**: Comprehensive model evaluation
- **`ModelVisualizer`**: Result visualization
- **`DataLoader`**: Data loading and merging
- **`FeatureEngineer`**: Advanced feature engineering
- **`CheckpointManager`**: Training state management

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

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Scikit-learn for machine learning utilities
- The cybersecurity research community for threat detection insights

## 📞 Support

For questions and support:

1. Check the troubleshooting section
2. Review the logs for error details
3. Open an issue with detailed error information
4. Provide sample data and configuration for reproduction

---

**Built with ❤️ for cybersecurity professionals**
