"""Checkpoint management utilities for model training."""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import load_model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import CHECKPOINT_DIR
from .logger import get_logger


class CheckpointManager:
    """Manages model checkpoints and training state."""
    
    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.logger = get_logger("checkpoint_manager")
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Define file paths
        self.model_path = os.path.join(self.checkpoint_dir, "model_checkpoint.h5")
        self.state_path = os.path.join(self.checkpoint_dir, "training_state.json")
        self.scaler_path = os.path.join(self.checkpoint_dir, "scaler.pkl")
        self.encoders_path = os.path.join(self.checkpoint_dir, "label_encoders.pkl")
        self.features_path = os.path.join(self.checkpoint_dir, "feature_columns.pkl")
    
    def save_checkpoint(self, 
                       model: tf.keras.Model,
                       epoch: int,
                       sequence_length: int,
                       feature_columns: list,
                       scaler: Optional[Any] = None,
                       label_encoders: Optional[Dict] = None,
                       additional_state: Optional[Dict] = None) -> bool:
        """Save complete training checkpoint."""
        try:
            # Save model
            model.save(self.model_path)
            self.logger.info(f"Model saved to {self.model_path}")
            
            # Save scaler if provided
            if scaler is not None:
                with open(self.scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
                self.logger.info("Scaler saved")
            
            # Save label encoders if provided
            if label_encoders is not None:
                with open(self.encoders_path, "wb") as f:
                    pickle.dump(label_encoders, f)
                self.logger.info("Label encoders saved")
            
            # Save feature columns
            with open(self.features_path, "wb") as f:
                pickle.dump(feature_columns, f)
            self.logger.info("Feature columns saved")
            
            # Save training state
            training_state = {
                "last_epoch": epoch,
                "sequence_length": sequence_length,
                "feature_columns": feature_columns,
                "timestamp": datetime.now().isoformat(),
                "model_path": self.model_path,
                "scaler_available": scaler is not None,
                "encoders_available": label_encoders is not None
            }
            
            # Add any additional state
            if additional_state:
                training_state.update(additional_state)
            
            with open(self.state_path, "w") as f:
                json.dump(training_state, f, indent=2)
            
            self.logger.info(f"[SUCCESS] Checkpoint saved successfully at epoch {epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error saving checkpoint: {e}")
            return False
    
    def load_checkpoint(self) -> Tuple[Optional[tf.keras.Model], int, Dict[str, Any]]:
        """Load complete training checkpoint."""
        model = None
        initial_epoch = 0
        state_info = {}
        
        if not self.checkpoint_exists():
            self.logger.info("No checkpoint found. Starting fresh training.")
            return model, initial_epoch, state_info
        
        try:
            self.logger.info("Loading checkpoint...")
            
            # Load training state
            with open(self.state_path, "r") as f:
                state = json.load(f)
                initial_epoch = state.get("last_epoch", 0)
                state_info = state
            
            # Load model
            if os.path.exists(self.model_path):
                model = load_model(self.model_path)
                self.logger.info("Model loaded successfully")
            
            self.logger.info(f"[SUCCESS] Checkpoint loaded - resuming from epoch {initial_epoch}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error loading checkpoint: {e}")
            self.logger.info("Starting fresh training...")
            model = None
            initial_epoch = 0
            state_info = {}
        
        return model, initial_epoch, state_info
    
    def load_scaler(self) -> Optional[Any]:
        """Load the saved scaler."""
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                self.logger.info("Scaler loaded successfully")
                return scaler
            except Exception as e:
                self.logger.error(f"Error loading scaler: {e}")
        return None
    
    def load_label_encoders(self) -> Optional[Dict]:
        """Load the saved label encoders."""
        if os.path.exists(self.encoders_path):
            try:
                with open(self.encoders_path, "rb") as f:
                    encoders = pickle.load(f)
                self.logger.info("Label encoders loaded successfully")
                return encoders
            except Exception as e:
                self.logger.error(f"Error loading label encoders: {e}")
        return None
    
    def load_feature_columns(self) -> Optional[list]:
        """Load the saved feature columns."""
        if os.path.exists(self.features_path):
            try:
                with open(self.features_path, "rb") as f:
                    features = pickle.load(f)
                self.logger.info("Feature columns loaded successfully")
                return features
            except Exception as e:
                self.logger.error(f"Error loading feature columns: {e}")
        return None
    
    def checkpoint_exists(self) -> bool:
        """Check if a checkpoint exists."""
        return (os.path.exists(self.model_path) and 
                os.path.exists(self.state_path))
    
    def get_checkpoint_info(self) -> Optional[Dict]:
        """Get information about the existing checkpoint."""
        if not os.path.exists(self.state_path):
            return None
        
        try:
            with open(self.state_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error reading checkpoint info: {e}")
            return None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Clean up old checkpoint files, keeping only the last N."""
        # This is a placeholder for more advanced checkpoint management
        # In a full implementation, you might want to keep multiple checkpoints
        pass
