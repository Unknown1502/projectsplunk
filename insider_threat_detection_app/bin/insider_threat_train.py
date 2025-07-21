#!/usr/bin/env python3
"""
Splunk Custom Search Command for Training Insider Threat Detection Models
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from splunklib.searchcommands import dispatch, GeneratingCommand, Configuration, Option, validators

# Add the lib directory to Python path
app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_dir = os.path.join(app_root, 'bin', 'lib')
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

# Import our modules
try:
    from src.training.trainer import InsiderThreatTrainer
    from src.utils.gpu_setup import initialize_gpu_environment
    from src.utils.logger import get_logger
    import tensorflow as tf
except ImportError as e:
    import traceback
    sys.stderr.write(f"Import error: {str(e)}\n")
    sys.stderr.write(traceback.format_exc())
    raise

@Configuration()
class InsiderThreatTrainCommand(GeneratingCommand):
    """
    Custom Splunk search command for training insider threat detection models.
    
    Usage:
    | insider_threat_train [data_source="lookup"] [model_name="auto"] [epochs=30] [batch_size=16] [use_lstm=false]
    
    Examples:
    | insider_threat_train
    | insider_threat_train data_source="index" model_name="model_v2" epochs=50
    | insider_threat_train use_lstm=true batch_size=32
    """
    
    data_source = Option(
        doc='Data source: "lookup" for CSV files or "index" for indexed data (default: lookup)',
        require=False,
        validate=validators.Match("data_source", r"^(lookup|index)$"),
        default="lookup"
    )
    
    model_name = Option(
        doc='Name for the trained model (default: auto-generated)',
        require=False,
        validate=validators.Match("model_name", r"^[\w\-\.]+$"),
        default="auto"
    )
    
    epochs = Option(
        doc='Number of training epochs (default: 30)',
        require=False,
        validate=validators.Integer(minimum=1, maximum=1000),
        default=30
    )
    
    batch_size = Option(
        doc='Batch size for training (default: 16)',
        require=False,
        validate=validators.Integer(minimum=1, maximum=512),
        default=16
    )
    
    use_lstm = Option(
        doc='Use LSTM instead of GRU (default: false)',
        require=False,
        validate=validators.Boolean(),
        default=False
    )
    
    use_gpu = Option(
        doc='Use GPU if available (default: true)',
        require=False,
        validate=validators.Boolean(),
        default=True
    )
    
    def generate(self):
        """Generate training results."""
        try:
            # Initialize GPU if requested
            if self.use_gpu:
                gpu_available = initialize_gpu_environment()
                self.logger.info(f"GPU initialization: {'Success' if gpu_available else 'Failed'}")
            
            # Determine data path
            if self.data_source == "lookup":
                data_path = os.path.join(self.service.app_path, 'lookups')
            else:
                # For indexed data, we'll need to export it first
                data_path = self._export_indexed_data()
            
            # Initialize trainer
            self.logger.info("Initializing trainer...")
            trainer = InsiderThreatTrainer(
                data_path=data_path,
                use_gru=not self.use_lstm
            )
            
            # Update training configuration
            trainer.batch_size = self.batch_size
            trainer.max_epochs = self.epochs
            
            # Generate model name if auto
            if self.model_name == "auto":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_type = "lstm" if self.use_lstm else "gru"
                self.model_name = f"insider_threat_{model_type}_{timestamp}"
            
            # Set model save path
            models_dir = os.path.join(self.service.app_path, 'bin', 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{self.model_name}.h5")
            
            # Start training
            self.logger.info(f"Starting training with configuration:")
            self.logger.info(f"  - Model type: {'LSTM' if self.use_lstm else 'GRU'}")
            self.logger.info(f"  - Batch size: {self.batch_size}")
            self.logger.info(f"  - Epochs: {self.epochs}")
            self.logger.info(f"  - Model name: {self.model_name}")
            
            # Run training pipeline
            results = trainer.run_complete_pipeline()
            
            # Save the model
            if results['overall_success'] and trainer.model and trainer.model.model:
                trainer.model.model.save(model_path)
                self.logger.info(f"Model saved to: {model_path}")
                
                # Save preprocessing artifacts
                self._save_preprocessing_artifacts(trainer, models_dir)
            
            # Generate output records
            yield self._create_summary_record(results, model_path)
            
            # Generate detailed metrics records
            if results.get('training_results', {}).get('final_metrics'):
                yield from self._create_metrics_records(results['training_results']['final_metrics'])
            
            # Generate training history records
            if results.get('training_results', {}).get('training_history'):
                yield from self._create_history_records(results['training_results']['training_history'])
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            yield {
                '_time': datetime.now().timestamp(),
                'status': 'failed',
                'error': str(e),
                'model_name': self.model_name
            }
    
    def _export_indexed_data(self):
        """Export indexed data to CSV for training."""
        # This is a placeholder - in a real implementation, you would:
        # 1. Run a search to get the training data
        # 2. Export it to CSV format
        # 3. Return the path to the exported data
        
        export_dir = os.path.join(self.service.app_path, 'lookups', 'training_data')
        os.makedirs(export_dir, exist_ok=True)
        
        # For now, return the default data path
        return os.path.join(self.service.app_path, 'lookups')
    
    def _save_preprocessing_artifacts(self, trainer, models_dir):
        """Save preprocessing artifacts alongside the model."""
        try:
            import pickle
            
            # Save scaler
            if hasattr(trainer, 'preprocessor') and hasattr(trainer.preprocessor, 'scaler'):
                scaler_path = os.path.join(models_dir, 'scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(trainer.preprocessor.scaler, f)
                self.logger.info("Saved scaler")
            
            # Save label encoders
            if hasattr(trainer, 'feature_engineer') and hasattr(trainer.feature_engineer, 'label_encoders'):
                encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
                with open(encoders_path, 'wb') as f:
                    pickle.dump(trainer.feature_engineer.label_encoders, f)
                self.logger.info("Saved label encoders")
            
            # Save feature columns
            if hasattr(trainer, 'feature_columns'):
                features_path = os.path.join(models_dir, 'feature_columns.pkl')
                with open(features_path, 'wb') as f:
                    pickle.dump(trainer.feature_columns, f)
                self.logger.info("Saved feature columns")
                
        except Exception as e:
            self.logger.warning(f"Failed to save preprocessing artifacts: {e}")
    
    def _create_summary_record(self, results, model_path):
        """Create a summary record of the training results."""
        record = {
            '_time': datetime.now().timestamp(),
            'event_type': 'training_summary',
            'status': 'success' if results['overall_success'] else 'failed',
            'model_name': self.model_name,
            'model_path': model_path,
            'model_type': 'LSTM' if self.use_lstm else 'GRU',
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        
        # Add data info if available
        if 'data_loading' in results and results['data_loading'].get('success'):
            data_info = results['data_loading'].get('data_info', {})
            record.update({
                'total_sequences': data_info.get('total_sequences', 0),
                'feature_count': data_info.get('feature_count', 0),
                'threat_ratio': round(data_info.get('threat_ratio', 0), 3)
            })
        
        # Add final metrics if available
        if results.get('training_results', {}).get('final_metrics', {}).get('basic_metrics'):
            metrics = results['training_results']['final_metrics']['basic_metrics']
            record.update({
                'final_accuracy': round(metrics.get('accuracy', 0), 4),
                'final_precision': round(metrics.get('precision', 0), 4),
                'final_recall': round(metrics.get('recall', 0), 4),
                'final_f1_score': round(metrics.get('f1_score', 0), 4)
            })
        
        return record
    
    def _create_metrics_records(self, final_metrics):
        """Create detailed metrics records."""
        timestamp = datetime.now().timestamp()
        
        # Basic metrics record
        if 'basic_metrics' in final_metrics:
            yield {
                '_time': timestamp,
                'event_type': 'training_metrics',
                'metric_type': 'basic',
                'model_name': self.model_name,
                **{k: round(v, 4) for k, v in final_metrics['basic_metrics'].items()}
            }
        
        # Advanced metrics record
        if 'advanced_metrics' in final_metrics:
            yield {
                '_time': timestamp,
                'event_type': 'training_metrics',
                'metric_type': 'advanced',
                'model_name': self.model_name,
                **{k: round(v, 4) if isinstance(v, (int, float)) else v 
                   for k, v in final_metrics['advanced_metrics'].items()}
            }
        
        # Per-class metrics
        if 'per_class_metrics' in final_metrics:
            for class_name, metrics in final_metrics['per_class_metrics'].items():
                yield {
                    '_time': timestamp,
                    'event_type': 'training_metrics',
                    'metric_type': 'per_class',
                    'model_name': self.model_name,
                    'class': class_name,
                    **{k: round(v, 4) for k, v in metrics.items()}
                }
    
    def _create_history_records(self, history):
        """Create training history records."""
        # Convert history to records
        epochs = len(history.get('loss', []))
        base_time = datetime.now().timestamp()
        
        for epoch in range(epochs):
            record = {
                '_time': base_time + (epoch * 60),  # Space out by 1 minute
                'event_type': 'training_history',
                'model_name': self.model_name,
                'epoch': epoch + 1
            }
            
            # Add metrics for this epoch
            for metric_name in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                if metric_name in history and epoch < len(history[metric_name]):
                    record[metric_name] = round(history[metric_name][epoch], 4)
            
            yield record

# Entry point for Splunk
if __name__ == "__main__":
    dispatch(InsiderThreatTrainCommand, sys.argv, sys.stdin, sys.stdout, __name__)
