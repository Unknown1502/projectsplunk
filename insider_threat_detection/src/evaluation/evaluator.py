"""Model evaluation utilities for insider threat detection."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
from typing import Dict, List, Tuple, Any, Optional
import tensorflow as tf

from ..utils.logger import get_logger
from ..models.model_utils import ModelUtils


class ModelEvaluator:
    """Comprehensive model evaluation for insider threat detection."""
    
    def __init__(self):
        self.logger = get_logger("model_evaluator")
        self.evaluation_results = {}
    
    def evaluate_model(self, 
                      model: tf.keras.Model,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      threshold: float = 0.5,
                      model_name: str = "model") -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        self.logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Get predictions
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Flatten arrays for sklearn metrics
            y_test_flat = y_test.flatten()
            y_pred_flat = y_pred.flatten()
            y_pred_proba_flat = y_pred_proba.flatten()
            
            # Basic metrics
            basic_metrics = self._calculate_basic_metrics(y_test_flat, y_pred_flat, y_pred_proba_flat)
            
            # Advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(y_test_flat, y_pred_proba_flat)
            
            # Confusion matrix analysis
            confusion_analysis = self._analyze_confusion_matrix(y_test_flat, y_pred_flat)
            
            # Prediction distribution analysis
            prediction_analysis = self._analyze_predictions(y_pred_proba_flat, y_test_flat)
            
            # Threshold analysis
            threshold_analysis = self._analyze_thresholds(y_test_flat, y_pred_proba_flat)
            
            # Compile results
            evaluation_results = {
                "model_name": model_name,
                "threshold_used": threshold,
                "basic_metrics": basic_metrics,
                "advanced_metrics": advanced_metrics,
                "confusion_analysis": confusion_analysis,
                "prediction_analysis": prediction_analysis,
                "threshold_analysis": threshold_analysis,
                "evaluation_summary": self._create_evaluation_summary(basic_metrics, advanced_metrics)
            }
            
            self.evaluation_results[model_name] = evaluation_results
            self.logger.info(f"Evaluation completed for {model_name}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {e}")
            return {"error": str(e), "model_name": model_name}
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "specificity": self._calculate_specificity(y_true, y_pred),
            "balanced_accuracy": self._calculate_balanced_accuracy(y_true, y_pred)
        }
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced metrics."""
        advanced_metrics = {}
        
        try:
            # ROC AUC
            if len(np.unique(y_true)) > 1:
                advanced_metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
                
                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
                advanced_metrics["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist()
                }
            else:
                advanced_metrics["roc_auc"] = None
                advanced_metrics["roc_curve"] = None
            
            # Precision-Recall AUC
            advanced_metrics["pr_auc"] = float(average_precision_score(y_true, y_pred_proba))
            
            # Precision-Recall curve data
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
            advanced_metrics["pr_curve"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating advanced metrics: {e}")
            advanced_metrics["error"] = str(e)
        
        return advanced_metrics
    
    def _analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix in detail."""
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            analysis = {
                "confusion_matrix": cm.tolist(),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "total_samples": int(cm.sum()),
                "positive_samples": int(tp + fn),
                "negative_samples": int(tn + fp),
                "predicted_positive": int(tp + fp),
                "predicted_negative": int(tn + fn)
            }
            
            # Calculate rates
            if tp + fn > 0:
                analysis["true_positive_rate"] = float(tp / (tp + fn))
            else:
                analysis["true_positive_rate"] = 0.0
            
            if tn + fp > 0:
                analysis["true_negative_rate"] = float(tn / (tn + fp))
            else:
                analysis["true_negative_rate"] = 0.0
            
            if tp + fp > 0:
                analysis["positive_predictive_value"] = float(tp / (tp + fp))
            else:
                analysis["positive_predictive_value"] = 0.0
            
            if tn + fn > 0:
                analysis["negative_predictive_value"] = float(tn / (tn + fn))
            else:
                analysis["negative_predictive_value"] = 0.0
            
        else:
            analysis = {
                "confusion_matrix": cm.tolist(),
                "error": "Non-binary classification detected"
            }
        
        return analysis
    
    def _analyze_predictions(self, y_pred_proba: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction distributions."""
        analysis = {
            "prediction_statistics": {
                "mean": float(y_pred_proba.mean()),
                "std": float(y_pred_proba.std()),
                "min": float(y_pred_proba.min()),
                "max": float(y_pred_proba.max()),
                "median": float(np.median(y_pred_proba))
            },
            "percentiles": {
                "5th": float(np.percentile(y_pred_proba, 5)),
                "25th": float(np.percentile(y_pred_proba, 25)),
                "75th": float(np.percentile(y_pred_proba, 75)),
                "95th": float(np.percentile(y_pred_proba, 95))
            },
            "confidence_distribution": {
                "very_high_confidence": int((y_pred_proba > 0.9).sum()),
                "high_confidence": int(((y_pred_proba > 0.7) & (y_pred_proba <= 0.9)).sum()),
                "medium_confidence": int(((y_pred_proba > 0.3) & (y_pred_proba <= 0.7)).sum()),
                "low_confidence": int(((y_pred_proba > 0.1) & (y_pred_proba <= 0.3)).sum()),
                "very_low_confidence": int((y_pred_proba <= 0.1).sum())
            }
        }
        
        # Analyze predictions by true class
        if len(np.unique(y_true)) == 2:
            threat_predictions = y_pred_proba[y_true == 1]
            normal_predictions = y_pred_proba[y_true == 0]
            
            analysis["class_wise_predictions"] = {
                "threat_class": {
                    "count": len(threat_predictions),
                    "mean_prediction": float(threat_predictions.mean()) if len(threat_predictions) > 0 else 0,
                    "std_prediction": float(threat_predictions.std()) if len(threat_predictions) > 0 else 0
                },
                "normal_class": {
                    "count": len(normal_predictions),
                    "mean_prediction": float(normal_predictions.mean()) if len(normal_predictions) > 0 else 0,
                    "std_prediction": float(normal_predictions.std()) if len(normal_predictions) > 0 else 0
                }
            }
        
        return analysis
    
    def _analyze_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze performance across different thresholds."""
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_analysis = {
            "thresholds": thresholds.tolist(),
            "metrics_by_threshold": []
        }
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba > threshold).astype(int)
            
            metrics = {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred_thresh)),
                "precision": float(precision_score(y_true, y_pred_thresh, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred_thresh, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred_thresh, zero_division=0)),
                "predicted_positive_rate": float(y_pred_thresh.mean())
            }
            
            threshold_analysis["metrics_by_threshold"].append(metrics)
        
        # Find optimal thresholds
        f1_scores = [m["f1_score"] for m in threshold_analysis["metrics_by_threshold"]]
        best_f1_idx = np.argmax(f1_scores)
        
        threshold_analysis["optimal_thresholds"] = {
            "best_f1": {
                "threshold": threshold_analysis["metrics_by_threshold"][best_f1_idx]["threshold"],
                "f1_score": f1_scores[best_f1_idx]
            }
        }
        
        return threshold_analysis
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return 0.0
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return float((sensitivity + specificity) / 2)
    
    def _create_evaluation_summary(self, basic_metrics: Dict, advanced_metrics: Dict) -> Dict[str, Any]:
        """Create a summary of evaluation results."""
        summary = {
            "overall_performance": "Unknown",
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Determine overall performance
        f1 = basic_metrics.get("f1_score", 0)
        roc_auc = advanced_metrics.get("roc_auc", 0)
        
        if f1 > 0.8 and (roc_auc is None or roc_auc > 0.9):
            summary["overall_performance"] = "Excellent"
        elif f1 > 0.6 and (roc_auc is None or roc_auc > 0.8):
            summary["overall_performance"] = "Good"
        elif f1 > 0.4 and (roc_auc is None or roc_auc > 0.7):
            summary["overall_performance"] = "Fair"
        else:
            summary["overall_performance"] = "Poor"
        
        # Identify strengths and weaknesses
        precision = basic_metrics.get("precision", 0)
        recall = basic_metrics.get("recall", 0)
        
        if precision > 0.8:
            summary["strengths"].append("High precision - low false positive rate")
        elif precision < 0.5:
            summary["weaknesses"].append("Low precision - high false positive rate")
        
        if recall > 0.8:
            summary["strengths"].append("High recall - good threat detection")
        elif recall < 0.5:
            summary["weaknesses"].append("Low recall - missing many threats")
        
        if roc_auc and roc_auc > 0.9:
            summary["strengths"].append("Excellent discrimination ability")
        elif roc_auc and roc_auc < 0.7:
            summary["weaknesses"].append("Poor discrimination ability")
        
        # Generate recommendations
        if precision < 0.6:
            summary["recommendations"].append("Consider adjusting threshold to reduce false positives")
        
        if recall < 0.6:
            summary["recommendations"].append("Model may need more training data or feature engineering")
        
        if f1 < 0.5:
            summary["recommendations"].append("Consider model architecture changes or hyperparameter tuning")
        
        return summary
    
    def compare_models(self, evaluation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare multiple model evaluation results."""
        if len(evaluation_results) < 2:
            return {"error": "Need at least 2 models to compare"}
        
        comparison = {
            "model_count": len(evaluation_results),
            "models": list(evaluation_results.keys()),
            "metric_comparison": {},
            "ranking": {},
            "best_model": {}
        }
        
        # Extract metrics for comparison
        metrics_to_compare = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        
        for metric in metrics_to_compare:
            comparison["metric_comparison"][metric] = {}
            
            for model_name, results in evaluation_results.items():
                if metric == "roc_auc":
                    value = results.get("advanced_metrics", {}).get(metric)
                else:
                    value = results.get("basic_metrics", {}).get(metric)
                
                comparison["metric_comparison"][metric][model_name] = value
        
        # Rank models by F1 score
        f1_scores = comparison["metric_comparison"]["f1_score"]
        sorted_models = sorted(f1_scores.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
        
        comparison["ranking"]["by_f1_score"] = [{"model": model, "f1_score": score} for model, score in sorted_models]
        comparison["best_model"] = {
            "name": sorted_models[0][0],
            "f1_score": sorted_models[0][1]
        }
        
        return comparison
    
    def generate_evaluation_report(self, model_name: str) -> str:
        """Generate a text report of evaluation results."""
        if model_name not in self.evaluation_results:
            return f"No evaluation results found for model: {model_name}"
        
        results = self.evaluation_results[model_name]
        
        report = f"""
INSIDER THREAT DETECTION MODEL EVALUATION REPORT
===============================================

Model: {model_name}
Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

BASIC METRICS
-------------
Accuracy:  {results['basic_metrics']['accuracy']:.4f}
Precision: {results['basic_metrics']['precision']:.4f}
Recall:    {results['basic_metrics']['recall']:.4f}
F1-Score:  {results['basic_metrics']['f1_score']:.4f}
Specificity: {results['basic_metrics']['specificity']:.4f}

ADVANCED METRICS
----------------
ROC AUC:   {results['advanced_metrics'].get('roc_auc', 'N/A')}
PR AUC:    {results['advanced_metrics'].get('pr_auc', 'N/A')}

CONFUSION MATRIX ANALYSIS
-------------------------
True Positives:  {results['confusion_analysis']['true_positives']}
False Positives: {results['confusion_analysis']['false_positives']}
True Negatives:  {results['confusion_analysis']['true_negatives']}
False Negatives: {results['confusion_analysis']['false_negatives']}

OVERALL ASSESSMENT
------------------
Performance Level: {results['evaluation_summary']['overall_performance']}

Strengths:
{chr(10).join(['- ' + strength for strength in results['evaluation_summary']['strengths']])}

Weaknesses:
{chr(10).join(['- ' + weakness for weakness in results['evaluation_summary']['weaknesses']])}

Recommendations:
{chr(10).join(['- ' + rec for rec in results['evaluation_summary']['recommendations']])}
"""
        
        return report
