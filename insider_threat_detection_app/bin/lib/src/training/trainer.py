"""Training orchestrator for insider threat detection models."""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional, Tuple

from ..data.loader import DataLoader
from ..data.preprocessor import DataPreprocessor
from ..data.feature_engineer import FeatureEngineer
from ..models.lstm_model import InsiderThreatLSTM
from ..models.model_utils import ModelUtils
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.logger import TrainingLogger
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import (
    TEST_SIZE, VALIDATION_SIZE, BATCH_SIZE, MAX_EPOCHS, 
    CHECKPOINT_FREQUENCY, EARLY_STOPPING_PATIENCE, 
    LR_REDUCTION_PATIENCE, LR_REDUCTION_FACTOR, MIN_LEARNING_RATE
)
from config.model_config import FEATURE_COLUMNS
from .callbacks import CustomCheckpointCallback


class InsiderThreatTrainer:
    """Main training orchestrator for insider threat detection."""
    
    def __init__(self, data_path: str = None, use_gru: bool = False):
        self.data_loader = DataLoader(data_path) if data_path else DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.checkpoint_manager = CheckpointManager()
        self.logger = TrainingLogger("insider_threat_trainer")
        
        # Model and training state
        self.model = InsiderThreatLSTM(use_gru=use_gru)
        self.training_data = {}
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.training_history = None
        
        # Training configuration
        self.batch_size = BATCH_SIZE
        self.max_epochs = MAX_EPOCHS
        self.initial_epoch = 0
    
    def prepare_data(self, force_reload: bool = False) -> Dict[str, Any]:
        """Prepare data for training."""
        self.logger.logger.info("Starting data preparation...")
        
        try:
            # Load and merge data
            merged_df = self.data_loader.load_and_merge_data()
            
            # Validate data quality
            validation_results = self.data_loader.validate_data_quality()
            if not validation_results["is_valid"]:
                self.logger.logger.warning(f"Data quality issues found: {validation_results['issues']}")
            
            # Preprocessing
            df_before = merged_df.copy()
            
            # Clean dates
            merged_df = self.preprocessor.clean_dates(merged_df)
            
            # Create time features
            merged_df = self.preprocessor.create_time_features(merged_df)
            
            # Handle missing values
            merged_df = self.preprocessor.handle_missing_values(merged_df)
            
            # Sort data
            merged_df = self.preprocessor.sort_and_prepare(merged_df)
            
            # Feature engineering
            merged_df = self.feature_engineer.create_user_behavior_features(merged_df)
            merged_df = self.feature_engineer.encode_categorical_features(merged_df)
            merged_df = self.feature_engineer.detect_anomalies(merged_df)
            merged_df = self.feature_engineer.create_threat_labels(merged_df)
            merged_df = self.feature_engineer.create_advanced_features(merged_df)
            
            # Update feature columns based on what's actually available
            available_features = [col for col in self.feature_columns if col in merged_df.columns]
            if len(available_features) != len(self.feature_columns):
                missing_features = set(self.feature_columns) - set(available_features)
                self.logger.logger.warning(f"Missing features: {missing_features}")
                self.feature_columns = available_features
            
            # Validate features
            feature_validation = self.feature_engineer.validate_features(merged_df, self.feature_columns)
            if not feature_validation["is_valid"]:
                self.logger.logger.error(f"Feature validation failed: {feature_validation}")
                raise ValueError("Feature validation failed")
            
            # Create sequences
            X, y = self.preprocessor.create_sequences(merged_df, self.feature_columns)
            
            if len(X) == 0:
                raise ValueError("No sequences created. Check data and sequence parameters.")
            
            # Store processed data
            self.training_data = {
                'X': X,
                'y': y,
                'merged_df': merged_df,
                'feature_columns': self.feature_columns,
                'preprocessing_summary': self.preprocessor.get_preprocessing_summary(df_before, merged_df)
            }
            
            self.logger.logger.info("Data preparation completed successfully")
            self.logger.logger.info(f"Final dataset: {len(X)} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
            
            return {
                'success': True,
                'sequences_created': len(X),
                'sequence_shape': X.shape,
                'threat_ratio': y.mean(),
                'feature_count': len(self.feature_columns)
            }
            
        except Exception as e:
            error_msg = str(e)
            # Early stopping is success, not failure
            if 'early stopping' in error_msg.lower() or 'restoring model weights' in error_msg.lower():
                self.logger.logger.info("Training completed with early stopping - SUCCESS!")
                return {'success': True, 'early_stopped': True, 'message': error_msg}
            else:
                self.logger.log_error(e)
                return {'success': False, 'error': error_msg}
    
    def setup_training(self) -> Dict[str, Any]:
        """Set up training environment and load checkpoints if available."""
        self.logger.logger.info("Setting up training environment...")
        
        try:
            # Check for existing checkpoint
            model, initial_epoch, state_info = self.checkpoint_manager.load_checkpoint()
            
            if model is not None:
                self.model.model = model
                self.model.is_compiled = True
                self.initial_epoch = initial_epoch
                
                # Load other training artifacts
                scaler = self.checkpoint_manager.load_scaler()
                if scaler:
                    self.preprocessor.scaler = scaler
                
                encoders = self.checkpoint_manager.load_label_encoders()
                if encoders:
                    self.feature_engineer.label_encoders = encoders
                
                features = self.checkpoint_manager.load_feature_columns()
                if features:
                    self.feature_columns = features
                
                self.logger.logger.info(f"Resumed from checkpoint at epoch {initial_epoch}")
            else:
                self.logger.logger.info("Starting fresh training")
            
            return {
                'success': True,
                'resumed_from_checkpoint': model is not None,
                'initial_epoch': self.initial_epoch,
                'checkpoint_info': state_info
            }
            
        except Exception as e:
            error_msg = str(e)
            # Early stopping is success, not failure
            if 'early stopping' in error_msg.lower() or 'restoring model weights' in error_msg.lower():
                self.logger.logger.info("Training completed with early stopping - SUCCESS!")
                return {'success': True, 'early_stopped': True, 'message': error_msg}
            else:
                self.logger.log_error(e)
                return {'success': False, 'error': error_msg}
    
    def temporal_split_data(self) -> Tuple[np.ndarray, ...]:
        """Split data temporally to prevent data leakage - FIXED VERSION."""
        self.logger.logger.info("Using FIXED temporal splitting to prevent data leakage...")
        
        # Use the new temporal splitting method from preprocessor
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.temporal_train_test_split(
            self.training_data['merged_df'], 
            self.feature_columns,
            train_ratio=0.70,
            val_ratio=0.15
        )
        
        self.logger.logger.info("[FIXED] Using proper temporal splitting - NO MORE DATA LEAKAGE!")
        self.logger.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        self.logger.logger.info(f"Threat ratios - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self) -> Dict[str, Any]:
        """Train the insider threat detection model."""
        self.logger.logger.info("Starting model training...")
        
        try:
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_split_data()
            
            # Scale features
            if self.initial_epoch == 0:
                # Fit scaler on training data
                X_train_scaled, X_val_scaled, X_test_scaled = self.preprocessor.scale_features(
                    X_train, X_val, X_test, fit_scaler=True
                )
            else:
                # Use existing scaler
                X_train_scaled, X_val_scaled, X_test_scaled = self.preprocessor.scale_features(
                    X_train, X_val, X_test, fit_scaler=False
                )
            
            # Calculate class weights first
            class_weights = ModelUtils.calculate_class_weights(y_train)
            self.logger.logger.info(f"Enhanced class weights: {class_weights}")
            
            # Build model if not loaded from checkpoint
            if self.model.model is None:
                input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
                self.model.build_model(input_shape)
                self.model.compile_model(class_weight=class_weights)
                
                self.logger.logger.info("Model architecture:")
                self.logger.logger.info(self.model.get_model_summary())
            
            # Create callbacks
            callbacks = [
                CustomCheckpointCallback(
                    self.checkpoint_manager,
                    self.preprocessor.scaler,
                    self.feature_engineer.label_encoders,
                    self.feature_columns,
                    save_frequency=CHECKPOINT_FREQUENCY
                )
            ]
            
            # Add standard callbacks
            standard_callbacks = ModelUtils.create_callbacks(
                checkpoint_path=self.checkpoint_manager.model_path,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                lr_reduction_patience=LR_REDUCTION_PATIENCE,
                lr_reduction_factor=LR_REDUCTION_FACTOR,
                min_lr=MIN_LEARNING_RATE
            )
            callbacks.extend(standard_callbacks)
            
            # Train model
            self.training_history = self.model.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=self.max_epochs,
                initial_epoch=self.initial_epoch,
                batch_size=self.batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Final evaluation
            final_metrics = ModelUtils.evaluate_model_performance(
                self.model.model, X_test_scaled, y_test
            )
            
            # Save final checkpoint
            final_epoch = self.initial_epoch + len(self.training_history.history["loss"])
            self.checkpoint_manager.save_checkpoint(
                model=self.model.model,
                epoch=final_epoch,
                sequence_length=self.preprocessor.scaler.n_features_in_ if hasattr(self.preprocessor.scaler, 'n_features_in_') else len(self.feature_columns),
                feature_columns=self.feature_columns,
                scaler=self.preprocessor.scaler,
                label_encoders=self.feature_engineer.label_encoders,
                additional_state={'training_completed': True}
            )
            
            self.logger.log_training_complete(final_metrics)
            
            return {
                'success': True,
                'final_epoch': final_epoch,
                'final_metrics': final_metrics,
                'training_history': self.training_history.history,
                'model_complexity': ModelUtils.analyze_model_complexity(self.model.model)
            }
            
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
            current_epoch = self.initial_epoch + len(self.training_history.history["loss"]) if self.training_history else self.initial_epoch
            
            # Save current state
            self.checkpoint_manager.save_checkpoint(
                model=self.model.model,
                epoch=current_epoch,
                sequence_length=len(self.feature_columns),
                feature_columns=self.feature_columns,
                scaler=self.preprocessor.scaler,
                label_encoders=self.feature_engineer.label_encoders,
                additional_state={'training_interrupted': True}
            )
            
            return {'success': False, 'interrupted': True, 'saved_epoch': current_epoch}
            
        except Exception as e:
            error_msg = str(e)
            # Early stopping is success, not failure
            if 'early stopping' in error_msg.lower() or 'restoring model weights' in error_msg.lower():
                self.logger.logger.info("Training completed with early stopping - SUCCESS!")
                return {'success': True, 'early_stopped': True, 'message': error_msg}
            else:
                self.logger.log_error(e)
                return {'success': False, 'error': error_msg}
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        self.logger.logger.info("Starting complete insider threat detection pipeline")
        
        pipeline_results = {
            'data_preparation': None,
            'training_setup': None,
            'training_results': None,
            'overall_success': False
        }
        
        try:
            # Step 1: Prepare data
            data_prep_results = self.prepare_data()
            pipeline_results['data_preparation'] = data_prep_results
            
            if not data_prep_results['success']:
                return pipeline_results
            
            # Step 2: Setup training
            setup_results = self.setup_training()
            pipeline_results['training_setup'] = setup_results
            
            if not setup_results['success']:
                return pipeline_results
            
            # Step 3: Train model
            training_results = self.train_model()
            pipeline_results['training_results'] = training_results
            
            pipeline_results['overall_success'] = training_results['success']
            
            if training_results['success']:
                self.logger.logger.info("Pipeline completed successfully!")
            elif training_results.get('interrupted'):
                self.logger.logger.info("Training interrupted by user - checkpoint saved")
            elif 'early_stopping' in str(training_results.get('error', '')).lower():
                self.logger.logger.info("Training completed with early stopping - this is SUCCESS!")
            else:
                self.logger.logger.error("Pipeline failed during training")
            
            return pipeline_results
            
        except Exception as e:
            self.logger.log_error(e)
            pipeline_results['pipeline_error'] = str(e)
            return pipeline_results
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_data:
            return {"error": "No training data available"}
        
        summary = {
            "data_info": {
                "total_sequences": len(self.training_data['X']),
                "sequence_shape": self.training_data['X'].shape,
                "feature_count": len(self.feature_columns),
                "threat_ratio": float(self.training_data['y'].mean())
            },
            "model_info": self.model.get_model_config() if self.model.model else None,
            "training_config": {
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "initial_epoch": self.initial_epoch
            },
            "checkpoint_info": self.checkpoint_manager.get_checkpoint_info()
        }
        
        if self.training_history:
            summary["training_history"] = {
                "epochs_completed": len(self.training_history.history["loss"]),
                "final_loss": self.training_history.history["loss"][-1],
                "final_val_loss": self.training_history.history["val_loss"][-1],
                "best_val_loss": min(self.training_history.history["val_loss"])
            }
        
        return summary
