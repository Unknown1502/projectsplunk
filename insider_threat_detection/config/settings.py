"""Global settings and configuration for the insider threat detection system."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = r"C:\Program Files\Splunk\etc\apps\projectsplunk\r1"
CHECKPOINT_DIR = os.path.join(DATA_PATH, "checkpoints")
MODEL_SAVE_PATH = os.path.join(DATA_PATH, "best_model.h5")

# Data configuration
DATA_FILES = {
    'http': ['http_logs.csv', 'http.csv', 'web_logs.csv'],
    'device': ['device_logs.csv', 'device.csv', 'logon.csv'],
    'logon': ['logon_logs.csv', 'logon.csv', 'auth.csv']
}

COMMON_COLUMNS = ['id', 'date', 'user', 'pc', 'activity_type', 'details']

# Feature engineering configuration
SEQUENCE_LENGTH = 8
STRIDE = 2
ANOMALY_CONTAMINATION = 0.1
THREAT_PERCENTILE = 85

# Training configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25
BATCH_SIZE = 16
MAX_EPOCHS = 30
CHECKPOINT_FREQUENCY = 2

# Model configuration
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
LR_REDUCTION_PATIENCE = 5
LR_REDUCTION_FACTOR = 0.5
MIN_LEARNING_RATE = 0.00001

# GPU configuration
ENABLE_MIXED_PRECISION = True
ENABLE_GPU_MEMORY_GROWTH = True

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
