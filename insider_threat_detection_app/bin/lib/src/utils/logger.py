"""Logging utilities for the insider threat detection system."""

import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import LOG_LEVEL, LOG_FORMAT, BASE_DIR


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Set up a logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Avoid adding multiple handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = BASE_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{name}_{timestamp}.log"
    return setup_logger(name, log_file)


class TrainingLogger:
    """Specialized logger for training progress."""
    
    def __init__(self, name: str = "training"):
        self.logger = get_logger(name)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log the start of an epoch."""
        self.logger.info(f"Starting epoch {epoch + 1}/{total_epochs}")
    
    def log_epoch_end(self, epoch: int, metrics: dict):
        """Log the end of an epoch with metrics."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch + 1} completed - {metrics_str}")
    
    def log_checkpoint_save(self, epoch: int):
        """Log checkpoint saving."""
        self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")
    
    def log_training_complete(self, final_metrics: dict):
        """Log training completion."""
        self.logger.info("Training completed successfully!")
        for metric, value in final_metrics.items():
            # Handle both single values and nested dictionaries
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        self.logger.info(f"Final {sub_metric}: {sub_value:.4f}")
                    else:
                        self.logger.info(f"Final {sub_metric}: {sub_value}")
            elif isinstance(value, (int, float)):
                self.logger.info(f"Final {metric}: {value:.4f}")
            else:
                self.logger.info(f"Final {metric}: {value}")
    
    def log_error(self, error: Exception):
        """Log training errors."""
        self.logger.error(f"Training error: {str(error)}", exc_info=True)
