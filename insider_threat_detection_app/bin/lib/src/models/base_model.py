"""Base model class for insider threat detection."""

from abc import ABC, abstractmethod
import tensorflow as tf
import sys
import os
from typing import Tuple, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger


class BaseModel(ABC):
    """Abstract base class for insider threat detection models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.logger = get_logger(f"model_{name}")
        self.is_compiled = False
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...]) -> tf.keras.Model:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration parameters."""
        pass
    
    def compile_model(self, optimizer: str = 'adam', 
                     loss: str = 'binary_crossentropy',
                     metrics: list = None) -> None:
        """Compile the model with specified parameters."""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall']
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.is_compiled = True
        self.logger.info(f"Model {self.name} compiled successfully")
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            return "Model not built yet"
        
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and non-trainable parameters."""
        if self.model is None:
            return {"trainable": 0, "non_trainable": 0, "total": 0}
        
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        
        return {
            "trainable": trainable_params,
            "non_trainable": non_trainable_params,
            "total": trainable_params + non_trainable_params
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save the model to specified path."""
        if self.model is None:
            self.logger.error("Cannot save model: model not built")
            return False
        
        try:
            self.model.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from specified path."""
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.is_compiled = True
            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, X: tf.Tensor, batch_size: int = 32) -> tf.Tensor:
        """Make predictions on input data."""
        if self.model is None:
            raise ValueError("Model must be built before making predictions")
        
        return self.model.predict(X, batch_size=batch_size)
    
    def evaluate(self, X: tf.Tensor, y: tf.Tensor, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError("Model must be built before evaluation")
        
        if not self.is_compiled:
            raise ValueError("Model must be compiled before evaluation")
        
        results = self.model.evaluate(X, y, batch_size=batch_size, verbose=0)
        
        # Map results to metric names
        metric_names = ['loss'] + [m.name if hasattr(m, 'name') else str(m) for m in self.model.metrics]
        
        return dict(zip(metric_names, results))
    
    def get_layer_info(self) -> list:
        """Get information about model layers."""
        if self.model is None:
            return []
        
        layer_info = []
        for i, layer in enumerate(self.model.layers):
            info = {
                "index": i,
                "name": layer.name,
                "type": type(layer).__name__,
                "output_shape": layer.output_shape,
                "param_count": layer.count_params()
            }
            
            # Add layer-specific information
            if hasattr(layer, 'units'):
                info["units"] = layer.units
            if hasattr(layer, 'activation'):
                info["activation"] = layer.activation.__name__ if callable(layer.activation) else str(layer.activation)
            if hasattr(layer, 'dropout'):
                info["dropout"] = layer.dropout
            if hasattr(layer, 'recurrent_dropout'):
                info["recurrent_dropout"] = layer.recurrent_dropout
            
            layer_info.append(info)
        
        return layer_info
    
    def validate_input_shape(self, input_shape: Tuple[int, ...]) -> bool:
        """Validate input shape compatibility."""
        if self.model is None:
            return True  # Will be validated when model is built
        
        expected_shape = self.model.input_shape[1:]  # Exclude batch dimension
        return input_shape == expected_shape
    
    def get_model_metrics(self) -> list:
        """Get list of model metrics."""
        if self.model is None or not self.is_compiled:
            return []
        
        return [m.name if hasattr(m, 'name') else str(m) for m in self.model.metrics]
