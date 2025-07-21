"""Visualization utilities for model evaluation and training results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import get_logger


class ModelVisualizer:
    """Comprehensive visualization for insider threat detection models."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        self.logger = get_logger("model_visualizer")
        self.style = style
        self.figsize = figsize
        
        # Set plotting style
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')
            self.logger.warning(f"Style '{self.style}' not available, using default")
        
        # Set color palette
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
        sns.set_palette("husl")
    
    def plot_training_history(self, history: Dict[str, List], save_path: Optional[str] = None) -> plt.Figure:
        """Plot comprehensive training history."""
        metrics = list(history.keys())
        val_metrics = [m for m in metrics if m.startswith('val_')]
        train_metrics = [m for m in metrics if not m.startswith('val_')]
        
        # Determine subplot layout
        n_metrics = len(train_metrics)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(train_metrics):
            ax = axes[i] if i < len(axes) else plt.subplot(rows, cols, i+1)
            
            # Plot training metric
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}', linewidth=2)
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(epochs, history[val_metric], 'r-', label=f'Validation {metric}', linewidth=2)
            
            ax.set_title(f'{metric.capitalize()} Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add trend analysis
            if len(history[metric]) > 5:
                self._add_trend_annotation(ax, history[metric], metric)
        
        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training history plot saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, evaluation_results: Dict[str, Dict], save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curves for multiple models."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            roc_data = results.get('advanced_metrics', {}).get('roc_curve')
            roc_auc = results.get('advanced_metrics', {}).get('roc_auc')
            
            if roc_data and roc_auc:
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
                
                ax.plot(fpr, tpr, color=self.colors[i % len(self.colors)], 
                       linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot random classifier line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curves plot saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curves(self, evaluation_results: Dict[str, Dict], save_path: Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curves for multiple models."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            pr_data = results.get('advanced_metrics', {}).get('pr_curve')
            pr_auc = results.get('advanced_metrics', {}).get('pr_auc')
            
            if pr_data and pr_auc:
                precision = pr_data['precision']
                recall = pr_data['recall']
                
                ax.plot(recall, precision, color=self.colors[i % len(self.colors)], 
                       linewidth=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=16, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"PR curves plot saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self, evaluation_results: Dict[str, Dict], save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrices for multiple models."""
        n_models = len(evaluation_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            ax = axes[i] if i < len(axes) else plt.subplot(rows, cols, i+1)
            
            cm_data = results.get('confusion_analysis', {}).get('confusion_matrix')
            if cm_data:
                cm = np.array(cm_data)
                
                # Create heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Normal', 'Threat'],
                           yticklabels=['Normal', 'Threat'])
                
                ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
        
        # Remove empty subplots
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrices plot saved to {save_path}")
        
        return fig
    
    def plot_prediction_distributions(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                    model_name: str = "Model", save_path: Optional[str] = None) -> plt.Figure:
        """Plot prediction probability distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall distribution
        axes[0, 0].hist(y_pred_proba, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Overall Prediction Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Prediction Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Distribution by true class
        if len(np.unique(y_true)) == 2:
            threat_preds = y_pred_proba[y_true == 1]
            normal_preds = y_pred_proba[y_true == 0]
            
            axes[0, 1].hist(normal_preds, bins=30, alpha=0.7, label='Normal', color='green', density=True)
            axes[0, 1].hist(threat_preds, bins=30, alpha=0.7, label='Threat', color='red', density=True)
            axes[0, 1].set_title('Prediction Distribution by True Class', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Prediction Probability')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot by class
        if len(np.unique(y_true)) == 2:
            df_plot = pd.DataFrame({
                'Prediction': y_pred_proba.flatten(),
                'True_Class': ['Threat' if x == 1 else 'Normal' for x in y_true.flatten()]
            })
            
            sns.boxplot(data=df_plot, x='True_Class', y='Prediction', ax=axes[1, 0])
            axes[1, 0].set_title('Prediction Distribution Box Plot', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Confidence levels
        confidence_levels = ['Very Low\n(0-0.2)', 'Low\n(0.2-0.4)', 'Medium\n(0.4-0.6)', 
                           'High\n(0.6-0.8)', 'Very High\n(0.8-1.0)']
        confidence_counts = [
            np.sum(y_pred_proba <= 0.2),
            np.sum((y_pred_proba > 0.2) & (y_pred_proba <= 0.4)),
            np.sum((y_pred_proba > 0.4) & (y_pred_proba <= 0.6)),
            np.sum((y_pred_proba > 0.6) & (y_pred_proba <= 0.8)),
            np.sum(y_pred_proba > 0.8)
        ]
        
        axes[1, 1].bar(confidence_levels, confidence_counts, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Prediction Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Prediction distributions plot saved to {save_path}")
        
        return fig
    
    def plot_threshold_analysis(self, evaluation_results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Plot threshold analysis results."""
        threshold_data = evaluation_results.get('threshold_analysis', {})
        if not threshold_data:
            self.logger.warning("No threshold analysis data available")
            return None
        
        metrics_by_threshold = threshold_data['metrics_by_threshold']
        df = pd.DataFrame(metrics_by_threshold)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Precision, Recall, F1 vs Threshold
        axes[0, 0].plot(df['threshold'], df['precision'], 'b-', label='Precision', linewidth=2)
        axes[0, 0].plot(df['threshold'], df['recall'], 'r-', label='Recall', linewidth=2)
        axes[0, 0].plot(df['threshold'], df['f1_score'], 'g-', label='F1-Score', linewidth=2)
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Metrics vs Threshold', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy vs Threshold
        axes[0, 1].plot(df['threshold'], df['accuracy'], 'purple', linewidth=2)
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs Threshold', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predicted Positive Rate vs Threshold
        axes[1, 0].plot(df['threshold'], df['predicted_positive_rate'], 'orange', linewidth=2)
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Predicted Positive Rate')
        axes[1, 0].set_title('Predicted Positive Rate vs Threshold', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Optimal threshold visualization
        optimal_f1 = threshold_data.get('optimal_thresholds', {}).get('best_f1', {})
        if optimal_f1:
            opt_threshold = optimal_f1['threshold']
            opt_f1_score = optimal_f1['f1_score']
            
            axes[1, 1].bar(['Current\n(0.5)', f'Optimal\n({opt_threshold:.2f})'], 
                          [df[df['threshold'] == 0.5]['f1_score'].iloc[0] if not df[df['threshold'] == 0.5].empty else 0, 
                           opt_f1_score],
                          color=['lightblue', 'lightgreen'], alpha=0.7)
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].set_title('Threshold Comparison', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Threshold analysis plot saved to {save_path}")
        
        return fig
    
    def plot_model_comparison(self, comparison_results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Plot comprehensive model comparison."""
        if 'metric_comparison' not in comparison_results:
            self.logger.warning("No metric comparison data available")
            return None
        
        metrics_data = comparison_results['metric_comparison']
        models = comparison_results['models']
        
        # Prepare data for plotting
        metrics_df = pd.DataFrame(metrics_data).T
        metrics_df = metrics_df.fillna(0)  # Fill NaN values with 0
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar plot of all metrics
        metrics_df.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Radar chart (if we have multiple metrics)
        if len(metrics_df.columns) > 2:
            self._create_radar_chart(metrics_df, axes[0, 1])
        
        # F1-Score ranking
        f1_data = comparison_results.get('ranking', {}).get('by_f1_score', [])
        if f1_data:
            models_ranked = [item['model'] for item in f1_data]
            f1_scores = [item['f1_score'] for item in f1_data]
            
            bars = axes[1, 0].barh(models_ranked, f1_scores, color='lightcoral', alpha=0.7)
            axes[1, 0].set_xlabel('F1-Score')
            axes[1, 0].set_title('Model Ranking by F1-Score', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                axes[1, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{score:.3f}', va='center', fontsize=10)
        
        # Performance summary table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        # Create summary table
        summary_data = []
        for model in models:
            row = [model]
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                value = metrics_data.get(metric, {}).get(model, 0)
                row.append(f'{value:.3f}' if value else 'N/A')
            summary_data.append(row)
        
        table = axes[1, 1].table(cellText=summary_data,
                                colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Performance Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def _add_trend_annotation(self, ax, values: List[float], metric: str):
        """Add trend annotation to training plots."""
        if len(values) < 5:
            return
        
        # Calculate trend over last 5 epochs
        recent_values = values[-5:]
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        if abs(trend) > 0.001:  # Only annotate significant trends
            trend_text = "[UP] Improving" if trend > 0 and 'loss' not in metric else "[DOWN] Improving"
            if (trend > 0 and 'loss' in metric) or (trend < 0 and 'loss' not in metric):
                trend_text = "[UP] Worsening" if trend > 0 else "[DOWN] Worsening"
            
            ax.annotate(trend_text, xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=10, ha='left', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    def _create_radar_chart(self, metrics_df: pd.DataFrame, ax):
        """Create radar chart for model comparison."""
        try:
            from math import pi
            
            # Number of metrics
            categories = list(metrics_df.index)
            N = len(categories)
            
            # Compute angle for each axis
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            
            # Plot each model
            for i, model in enumerate(metrics_df.columns):
                values = metrics_df[model].tolist()
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=model, color=self.colors[i % len(self.colors)])
                ax.fill(angles, values, alpha=0.25, color=self.colors[i % len(self.colors)])
            
            ax.set_ylim(0, 1)
            ax.set_title('Performance Radar Chart', fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Could not create radar chart: {e}")
            ax.text(0.5, 0.5, 'Radar Chart\nNot Available', ha='center', va='center', transform=ax.transAxes)
    
    def create_comprehensive_report(self, evaluation_results: Dict[str, Dict], 
                                  training_history: Optional[Dict] = None,
                                  save_dir: str = "evaluation_plots") -> Dict[str, str]:
        """Create comprehensive visualization report."""
        import os
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        saved_plots = {}
        
        try:
            # Training history plot
            if training_history:
                fig = self.plot_training_history(training_history)
                path = os.path.join(save_dir, "training_history.png")
                fig.savefig(path, dpi=300, bbox_inches='tight')
                saved_plots["training_history"] = path
                plt.close(fig)
            
            # ROC curves
            fig = self.plot_roc_curves(evaluation_results)
            path = os.path.join(save_dir, "roc_curves.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            saved_plots["roc_curves"] = path
            plt.close(fig)
            
            # PR curves
            fig = self.plot_precision_recall_curves(evaluation_results)
            path = os.path.join(save_dir, "pr_curves.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            saved_plots["pr_curves"] = path
            plt.close(fig)
            
            # Confusion matrices
            fig = self.plot_confusion_matrices(evaluation_results)
            path = os.path.join(save_dir, "confusion_matrices.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            saved_plots["confusion_matrices"] = path
            plt.close(fig)
            
            self.logger.info(f"Comprehensive visualization report saved to {save_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive report: {e}")
        
        return saved_plots
