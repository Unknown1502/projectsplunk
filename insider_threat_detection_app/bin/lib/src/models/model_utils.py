"""Utility functions for model operations."""

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Any
from ..utils.logger import get_logger


class ModelUtils:
    """Utility class for model-related operations."""
    
    def __init__(self):
        self.logger = get_logger("model_utils")
    
    @staticmethod
    def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced datasets with enhanced ratio."""
        classes = np.unique(y)
        
        # Count class occurrences
        class_counts = np.bincount(y.astype(int))
        
        # Calculate the ratio-based weights (55878/9703 ≈ 5.7)
        if len(class_counts) >= 2:
            normal_count = class_counts[0]  # Class 0 (normal)
            threat_count = class_counts[1]  # Class 1 (threats)
            
            if threat_count > 0:
                threat_weight = normal_count / threat_count
                class_weights = {0: 1.0, 1: threat_weight}
            else:
                class_weights = {0: 1.0, 1: 5.7}  # Fallback
        else:
            # Fallback to sklearn's balanced approach
            class_weights_array = compute_class_weight('balanced', classes=classes, y=y)
            class_weights = dict(enumerate(class_weights_array))
        
        return class_weights
    
    @staticmethod
    def create_callbacks(checkpoint_path: str, 
                        early_stopping_patience: int = 5,
                        lr_reduction_patience: int = 3,
                        lr_reduction_factor: float = 0.3,
                        min_lr: float = 0.00001) -> List[tf.keras.callbacks.Callback]:
        """Create standard callbacks for training."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_reduction_factor,
                patience=lr_reduction_patience,
                min_lr=min_lr,
                verbose=1
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    @staticmethod
    def evaluate_model_performance(model: tf.keras.Model, 
                                 X_test: np.ndarray, 
                                 y_test: np.ndarray,
                                 threshold: float = 0.3) -> Dict[str, Any]:
        """Comprehensive model evaluation with optimized threshold."""
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        logger = get_logger("model_evaluation")
        
        # Get predictions with optimized threshold
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > threshold).astype(int).flatten()  # Use 0.3 instead of 0.5
        y_test_flat = y_test.flatten()
        y_pred_proba_flat = y_pred_proba.flatten()
        
        # Calculate metrics
        results = {
            "test_loss": None,
            "test_accuracy": None,
            "classification_report": None,
            "confusion_matrix": None,
            "roc_auc": None,
            "prediction_distribution": None,
            "threshold_used": threshold,
            "threshold_comparison": None
        }
        
        try:
            # Model evaluation
            test_results = model.evaluate(X_test, y_test, verbose=0)
            results["test_loss"] = test_results[0]
            results["test_accuracy"] = test_results[1] if len(test_results) > 1 else None
            
            # Classification report
            results["classification_report"] = classification_report(
                y_test_flat, y_pred, output_dict=True
            )
            
            # Confusion matrix
            results["confusion_matrix"] = confusion_matrix(y_test_flat, y_pred).tolist()
            
            # ROC AUC
            if len(np.unique(y_test_flat)) > 1:
                results["roc_auc"] = roc_auc_score(y_test_flat, y_pred_proba_flat)
            
            # Prediction distribution
            results["prediction_distribution"] = {
                "mean": float(y_pred_proba_flat.mean()),
                "std": float(y_pred_proba_flat.std()),
                "min": float(y_pred_proba_flat.min()),
                "max": float(y_pred_proba_flat.max()),
                "percentiles": {
                    "25": float(np.percentile(y_pred_proba_flat, 25)),
                    "50": float(np.percentile(y_pred_proba_flat, 50)),
                    "75": float(np.percentile(y_pred_proba_flat, 75)),
                    "90": float(np.percentile(y_pred_proba_flat, 90)),
                    "95": float(np.percentile(y_pred_proba_flat, 95))
                }
            }
            
            # Compare different thresholds
            results["threshold_comparison"] = ModelUtils._compare_thresholds(
                y_test_flat, y_pred_proba_flat
            )
            
            logger.info("Model evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            results["error"] = str(e)
        
        return results
    
    @staticmethod
    def analyze_model_complexity(model: tf.keras.Model) -> Dict[str, Any]:
        """Analyze model complexity and architecture."""
        complexity_analysis = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "non_trainable_parameters": 0,
            "layer_count": len(model.layers),
            "layer_types": {},
            "memory_usage_mb": 0,
            "flops_estimate": 0
        }
        
        # Count parameters
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        complexity_analysis["trainable_parameters"] = trainable_params
        complexity_analysis["non_trainable_parameters"] = non_trainable_params
        complexity_analysis["total_parameters"] = trainable_params + non_trainable_params
        
        # Analyze layer types
        for layer in model.layers:
            layer_type = type(layer).__name__
            if layer_type in complexity_analysis["layer_types"]:
                complexity_analysis["layer_types"][layer_type] += 1
            else:
                complexity_analysis["layer_types"][layer_type] = 1
        
        # Estimate memory usage (rough approximation)
        # Assuming 32-bit floats (4 bytes per parameter)
        complexity_analysis["memory_usage_mb"] = (complexity_analysis["total_parameters"] * 4) / (1024 * 1024)
        
        return complexity_analysis
    
    @staticmethod
    def compare_models(models: Dict[str, tf.keras.Model], 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, Dict]:
        """Compare multiple models on the same test set."""
        logger = get_logger("model_comparison")
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                results = ModelUtils.evaluate_model_performance(model, X_test, y_test)
                complexity = ModelUtils.analyze_model_complexity(model)
                
                comparison_results[model_name] = {
                    "performance": results,
                    "complexity": complexity,
                    "efficiency_score": ModelUtils._calculate_efficiency_score(results, complexity)
                }
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                comparison_results[model_name] = {"error": str(e)}
        
        return comparison_results
    
    @staticmethod
    def _calculate_efficiency_score(performance: Dict, complexity: Dict) -> float:
        """Calculate efficiency score based on performance and complexity."""
        try:
            # Simple efficiency metric: accuracy per million parameters
            accuracy = performance.get("test_accuracy", 0)
            params_millions = complexity.get("total_parameters", 1) / 1_000_000
            
            if params_millions == 0:
                return 0
            
            return accuracy / params_millions
            
        except:
            return 0
    
    @staticmethod
    def get_model_predictions_analysis(model: tf.keras.Model,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     threshold: float = 0.5) -> Dict[str, Any]:
        """Analyze model predictions in detail."""
        predictions = model.predict(X)
        pred_binary = (predictions > threshold).astype(int).flatten()
        predictions_flat = predictions.flatten()
        y_flat = y.flatten()
        
        analysis = {
            "threshold_used": threshold,
            "total_samples": len(y_flat),
            "positive_predictions": int(pred_binary.sum()),
            "negative_predictions": int(len(pred_binary) - pred_binary.sum()),
            "true_positives": int(((pred_binary == 1) & (y_flat == 1)).sum()),
            "true_negatives": int(((pred_binary == 0) & (y_flat == 0)).sum()),
            "false_positives": int(((pred_binary == 1) & (y_flat == 0)).sum()),
            "false_negatives": int(((pred_binary == 0) & (y_flat == 1)).sum()),
            "confidence_distribution": {
                "high_confidence_positive": int((predictions_flat > 0.8).sum()),
                "medium_confidence_positive": int(((predictions_flat > 0.6) & (predictions_flat <= 0.8)).sum()),
                "low_confidence": int(((predictions_flat >= 0.4) & (predictions_flat <= 0.6)).sum()),
                "medium_confidence_negative": int(((predictions_flat >= 0.2) & (predictions_flat < 0.4)).sum()),
                "high_confidence_negative": int((predictions_flat < 0.2).sum())
            }
        }
        
        # Calculate derived metrics
        if analysis["true_positives"] + analysis["false_positives"] > 0:
            analysis["precision"] = analysis["true_positives"] / (analysis["true_positives"] + analysis["false_positives"])
        else:
            analysis["precision"] = 0
        
        if analysis["true_positives"] + analysis["false_negatives"] > 0:
            analysis["recall"] = analysis["true_positives"] / (analysis["true_positives"] + analysis["false_negatives"])
        else:
            analysis["recall"] = 0
        
        if analysis["precision"] + analysis["recall"] > 0:
            analysis["f1_score"] = 2 * (analysis["precision"] * analysis["recall"]) / (analysis["precision"] + analysis["recall"])
        else:
            analysis["f1_score"] = 0
        
        return analysis
    
    @staticmethod
    def optimize_threshold(model: tf.keras.Model,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, Dict]:
        """Find optimal threshold for binary classification."""
        from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
        
        predictions = model.predict(X_val).flatten()
        y_val_flat = y_val.flatten()
        
        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_val_flat, predictions)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        f1_scores = np.nan_to_num(f1_scores)
        
        best_threshold = 0.5
        best_score = 0
        
        if metric == 'f1':
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_score = f1_scores[best_idx]
        elif metric == 'precision':
            # Find threshold that maximizes precision while maintaining reasonable recall
            valid_indices = recall[:-1] >= 0.1  # At least 10% recall
            if valid_indices.any():
                valid_precision = precision[:-1][valid_indices]
                valid_thresholds = thresholds[valid_indices]
                best_idx = np.argmax(valid_precision)
                best_threshold = valid_thresholds[best_idx]
                best_score = valid_precision[best_idx]
        elif metric == 'recall':
            # Find threshold that maximizes recall while maintaining reasonable precision
            valid_indices = precision[:-1] >= 0.1  # At least 10% precision
            if valid_indices.any():
                valid_recall = recall[:-1][valid_indices]
                valid_thresholds = thresholds[valid_indices]
                best_idx = np.argmax(valid_recall)
                best_threshold = valid_thresholds[best_idx]
                best_score = valid_recall[best_idx]
        
        # Evaluate at best threshold
        y_pred_optimal = (predictions > best_threshold).astype(int)
        
        optimal_metrics = {
            'threshold': best_threshold,
            'f1_score': f1_score(y_val_flat, y_pred_optimal),
            'precision': precision_score(y_val_flat, y_pred_optimal),
            'recall': recall_score(y_val_flat, y_pred_optimal),
            'optimization_metric': metric,
            'optimization_score': best_score
        }
        
        return best_threshold, optimal_metrics
    
    @staticmethod
    def _compare_thresholds(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Dict]:
        """Compare performance at different thresholds."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        comparison = {}
        
        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            
            try:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                comparison[f"threshold_{thresh}"] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "predictions_positive": int(y_pred.sum()),
                    "false_negatives": int(((y_pred == 0) & (y_true == 1)).sum())
                }
            except:
                comparison[f"threshold_{thresh}"] = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "predictions_positive": 0,
                    "false_negatives": int(y_true.sum())
                }
        
        return comparison
