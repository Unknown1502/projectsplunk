"""Custom callbacks for training insider threat detection models."""

import tensorflow as tf
from typing import Dict, Any, Optional
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.logger import get_logger


class CustomCheckpointCallback(tf.keras.callbacks.Callback):
    """Custom callback for saving training checkpoints."""
    
    def __init__(self, 
                 checkpoint_manager: CheckpointManager,
                 scaler: Any,
                 label_encoders: Dict,
                 feature_columns: list,
                 save_frequency: int = 5):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.feature_columns = feature_columns
        self.save_frequency = save_frequency
        self.logger = get_logger("checkpoint_callback")
    
    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint at specified frequency."""
        if (epoch + 1) % self.save_frequency == 0:
            self.logger.info(f"Auto-saving checkpoint at epoch {epoch + 1}")
            
            success = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                epoch=epoch + 1,
                sequence_length=len(self.feature_columns),
                feature_columns=self.feature_columns,
                scaler=self.scaler,
                label_encoders=self.label_encoders,
                additional_state={
                    'current_metrics': logs,
                    'auto_save': True
                }
            )
            
            if success:
                self.logger.info(f"[PASS] Checkpoint saved successfully at epoch {epoch + 1}")
            else:
                self.logger.error(f"[FAIL] Failed to save checkpoint at epoch {epoch + 1}")


class MetricsLoggingCallback(tf.keras.callbacks.Callback):
    """Callback for detailed metrics logging."""
    
    def __init__(self, log_frequency: int = 1):
        super().__init__()
        self.log_frequency = log_frequency
        self.logger = get_logger("metrics_callback")
        self.epoch_metrics = []
    
    def on_epoch_begin(self, epoch, logs=None):
        """Log epoch start."""
        self.logger.info(f"Starting epoch {epoch + 1}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log detailed metrics at epoch end."""
        if logs and (epoch + 1) % self.log_frequency == 0:
            # Store metrics
            epoch_data = {
                'epoch': epoch + 1,
                'metrics': logs.copy()
            }
            self.epoch_metrics.append(epoch_data)
            
            # Log metrics
            metrics_str = []
            for key, value in logs.items():
                metrics_str.append(f"{key}: {value:.4f}")
            
            self.logger.info(f"Epoch {epoch + 1} - {', '.join(metrics_str)}")
            
            # Check for potential issues
            self._check_training_health(epoch, logs)
    
    def _check_training_health(self, epoch, logs):
        """Check for training issues and log warnings."""
        if len(self.epoch_metrics) < 2:
            return
        
        current_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        # Check for overfitting
        if val_loss > current_loss * 1.5:
            self.logger.warning(f"Potential overfitting detected at epoch {epoch + 1}: "
                              f"val_loss ({val_loss:.4f}) >> train_loss ({current_loss:.4f})")
        
        # Check for exploding gradients
        if current_loss > 10:
            self.logger.warning(f"High loss detected at epoch {epoch + 1}: {current_loss:.4f}")
        
        # Check for vanishing gradients (loss not decreasing)
        if len(self.epoch_metrics) >= 5:
            recent_losses = [m['metrics'].get('loss', 0) for m in self.epoch_metrics[-5:]]
            if all(abs(recent_losses[i] - recent_losses[i-1]) < 0.001 for i in range(1, len(recent_losses))):
                self.logger.warning(f"Loss plateau detected at epoch {epoch + 1} - consider adjusting learning rate")
    
    def get_metrics_history(self):
        """Get complete metrics history."""
        return self.epoch_metrics


class EarlyStoppingWithPatience(tf.keras.callbacks.Callback):
    """Enhanced early stopping with more sophisticated patience logic."""
    
    def __init__(self, 
                 monitor='val_loss',
                 patience=7,
                 min_delta=0.001,
                 restore_best_weights=True,
                 baseline=None):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        self.logger = get_logger("early_stopping")
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.best_epoch = 0
    
    def on_train_begin(self, logs=None):
        """Initialize early stopping."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf') if 'loss' in self.monitor else float('-inf')
        self.best_weights = None
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """Check early stopping condition."""
        current = logs.get(self.monitor)
        if current is None:
            self.logger.warning(f"Early stopping conditioned on metric `{self.monitor}` "
                              f"which is not available. Available metrics are: {list(logs.keys())}")
            return
        
        # Check if current is better than best
        if 'loss' in self.monitor:
            is_better = current < (self.best - self.min_delta)
        else:
            is_better = current > (self.best + self.min_delta)
        
        if is_better:
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            self.logger.info(f"New best {self.monitor}: {current:.4f} at epoch {epoch + 1}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                self.logger.info(f"Best {self.monitor}: {self.best:.4f} at epoch {self.best_epoch + 1}")
                
                if self.restore_best_weights and self.best_weights is not None:
                    self.logger.info("Restoring model weights from the end of the best epoch")
                    self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        """Log early stopping results."""
        if self.stopped_epoch > 0:
            self.logger.info(f"Training stopped early at epoch {self.stopped_epoch + 1}")
        else:
            self.logger.info("Training completed without early stopping")


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Custom learning rate scheduler with multiple strategies."""
    
    def __init__(self, 
                 strategy='reduce_on_plateau',
                 factor=0.5,
                 patience=3,
                 min_lr=1e-7,
                 monitor='val_loss'):
        super().__init__()
        self.strategy = strategy
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.monitor = monitor
        self.logger = get_logger("lr_scheduler")
        
        self.wait = 0
        self.best = float('inf') if 'loss' in monitor else float('-inf')
        self.initial_lr = None
    
    def on_train_begin(self, logs=None):
        """Initialize learning rate scheduler."""
        self.initial_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.logger.info(f"Initial learning rate: {self.initial_lr}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Adjust learning rate based on strategy."""
        current = logs.get(self.monitor)
        if current is None:
            return
        
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        if self.strategy == 'reduce_on_plateau':
            self._reduce_on_plateau(current, current_lr, epoch)
        elif self.strategy == 'exponential_decay':
            self._exponential_decay(current_lr, epoch)
        elif self.strategy == 'cosine_annealing':
            self._cosine_annealing(current_lr, epoch)
    
    def _reduce_on_plateau(self, current, current_lr, epoch):
        """Reduce learning rate on plateau."""
        if 'loss' in self.monitor:
            is_better = current < self.best
        else:
            is_better = current > self.best
        
        if is_better:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(current_lr * self.factor, self.min_lr)
                if new_lr < current_lr:
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                    self.logger.info(f"Reducing learning rate to {new_lr:.2e} at epoch {epoch + 1}")
                    self.wait = 0
    
    def _exponential_decay(self, current_lr, epoch):
        """Exponential decay of learning rate."""
        decay_rate = 0.96
        new_lr = max(self.initial_lr * (decay_rate ** epoch), self.min_lr)
        if new_lr != current_lr:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
    
    def _cosine_annealing(self, current_lr, epoch):
        """Cosine annealing learning rate schedule."""
        import math
        T_max = 50  # Maximum number of epochs for one cycle
        new_lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)


class ModelComplexityMonitor(tf.keras.callbacks.Callback):
    """Monitor model complexity and performance metrics."""
    
    def __init__(self, log_frequency=10):
        super().__init__()
        self.log_frequency = log_frequency
        self.logger = get_logger("complexity_monitor")
        self.complexity_history = []
    
    def on_train_begin(self, logs=None):
        """Log initial model complexity."""
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        self.logger.info(f"Model complexity - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Monitor training complexity metrics."""
        if (epoch + 1) % self.log_frequency == 0:
            # Calculate gradient norms (if available)
            try:
                gradients = []
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        grad = tf.keras.backend.gradients(self.model.total_loss, layer.kernel)[0]
                        if grad is not None:
                            grad_norm = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(grad)))
                            gradients.append(float(grad_norm))
                
                if gradients:
                    avg_grad_norm = sum(gradients) / len(gradients)
                    max_grad_norm = max(gradients)
                    
                    complexity_data = {
                        'epoch': epoch + 1,
                        'avg_gradient_norm': avg_grad_norm,
                        'max_gradient_norm': max_grad_norm,
                        'metrics': logs.copy() if logs else {}
                    }
                    
                    self.complexity_history.append(complexity_data)
                    
                    if max_grad_norm > 10:
                        self.logger.warning(f"High gradient norm detected: {max_grad_norm:.4f}")
                    elif max_grad_norm < 1e-6:
                        self.logger.warning(f"Very low gradient norm detected: {max_grad_norm:.2e}")
                        
            except Exception as e:
                self.logger.debug(f"Could not calculate gradient norms: {e}")
    
    def get_complexity_history(self):
        """Get complexity monitoring history."""
        return self.complexity_history
