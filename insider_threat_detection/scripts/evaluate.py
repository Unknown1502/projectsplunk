"""Evaluation script for insider threat detection model."""

import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Lazy imports to speed up help text display
def lazy_imports():
    """Import heavy modules only when needed."""
    global ModelEvaluator, ModelVisualizer, DataLoader, DataPreprocessor
    global FeatureEngineer, CheckpointManager, get_logger, InsiderThreatLSTM
    global train_test_split
    
    from src.evaluation.evaluator import ModelEvaluator
    from src.evaluation.visualizer import ModelVisualizer
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.data.feature_engineer import FeatureEngineer
    from src.utils.checkpoint_manager import CheckpointManager
    from src.utils.logger import get_logger
    from src.models.lstm_model import InsiderThreatLSTM
    from sklearn.model_selection import train_test_split


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Insider Threat Detection Model')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved model file')
    
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory (default: from config)')
    
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results (default: evaluation_results)')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    
    parser.add_argument('--optimize-threshold', action='store_true', default=False,
                       help='Find optimal threshold (default: False)')
    
    parser.add_argument('--create-plots', action='store_true', default=True,
                       help='Create visualization plots (default: True)')
    
    parser.add_argument('--save-predictions', action='store_true', default=False,
                       help='Save predictions to file (default: False)')
    
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose logging (default: False)')
    
    return parser.parse_args()


def load_model_and_artifacts(model_path: str, checkpoint_manager):
    """Load model and associated artifacts."""
    logger = get_logger("model_loader")
    
    try:
        # Load model
        model = InsiderThreatLSTM()
        success = model.load_model(model_path)
        
        if not success:
            raise ValueError(f"Failed to load model from {model_path}")
        
        # Load associated artifacts
        scaler = checkpoint_manager.load_scaler()
        label_encoders = checkpoint_manager.load_label_encoders()
        feature_columns = checkpoint_manager.load_feature_columns()
        
        logger.info("Model and artifacts loaded successfully")
        
        return model, scaler, label_encoders, feature_columns
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def prepare_test_data(data_path: str, scaler, label_encoders, feature_columns):
    """Prepare test data for evaluation."""
    logger = get_logger("data_preparation")
    
    try:
        # Load and process data
        data_loader = DataLoader(data_path)
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        
        # Set loaded artifacts
        if scaler:
            preprocessor.scaler = scaler
        if label_encoders:
            feature_engineer.label_encoders = label_encoders
        
        # Load data
        merged_df = data_loader.load_and_merge_data()
        
        # Preprocessing
        merged_df = preprocessor.clean_dates(merged_df)
        merged_df = preprocessor.create_time_features(merged_df)
        merged_df = preprocessor.handle_missing_values(merged_df)
        merged_df = preprocessor.sort_and_prepare(merged_df)
        
        # Feature engineering
        merged_df = feature_engineer.create_user_behavior_features(merged_df)
        merged_df = feature_engineer.encode_categorical_features(merged_df)
        merged_df = feature_engineer.detect_anomalies(merged_df)
        merged_df = feature_engineer.create_threat_labels(merged_df)
        merged_df = feature_engineer.create_advanced_features(merged_df)
        
        # Create sequences
        X, y = preprocessor.create_sequences(merged_df, feature_columns)
        
        # Split data (use same split as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_test_scaled = preprocessor.scale_features(X_test, fit_scaler=False)
        
        logger.info(f"Test data prepared: {len(X_test_scaled)} samples")
        
        return X_test_scaled, y_test
        
    except Exception as e:
        logger.error(f"Error preparing test data: {e}")
        raise


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Import heavy modules only when actually running (not for --help)
    lazy_imports()
    
    # Setup logging
    logger = get_logger("evaluate_script")
    
    if args.verbose:
        logger.info("Verbose logging enabled")
        logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize components
        checkpoint_manager = CheckpointManager()
        evaluator = ModelEvaluator()
        
        if args.create_plots:
            visualizer = ModelVisualizer()
        
        # Load model and artifacts
        logger.info("Loading model and artifacts...")
        model, scaler, label_encoders, feature_columns = load_model_and_artifacts(
            args.model_path, checkpoint_manager
        )
        
        # Prepare test data
        logger.info("Preparing test data...")
        X_test, y_test = prepare_test_data(
            args.data_path, scaler, label_encoders, feature_columns
        )
        
        # Optimize threshold if requested
        threshold = args.threshold
        if args.optimize_threshold:
            logger.info("Optimizing classification threshold...")
            from src.models.model_utils import ModelUtils
            
            # Use a portion of test data for threshold optimization
            X_val = X_test[:len(X_test)//2]
            y_val = y_test[:len(y_test)//2]
            X_test = X_test[len(X_test)//2:]
            y_test = y_test[len(y_test)//2:]
            
            optimal_threshold, threshold_metrics = ModelUtils.optimize_threshold(
                model.model, X_val, y_val, metric='f1'
            )
            
            threshold = optimal_threshold
            logger.info(f"Optimal threshold found: {threshold:.3f}")
            logger.info(f"Threshold optimization metrics: {threshold_metrics}")
        
        # Evaluate model
        logger.info(f"Evaluating model with threshold {threshold:.3f}...")
        evaluation_results = evaluator.evaluate_model(
            model.model, X_test, y_test, threshold=threshold, model_name="InsiderThreatModel"
        )
        
        # Save evaluation results
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info(f"Evaluation results saved to {results_file}")
        
        # Generate text report
        report = evaluator.generate_evaluation_report("InsiderThreatModel")
        report_file = os.path.join(args.output_dir, "evaluation_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to {report_file}")
        
        # Create visualizations
        if args.create_plots:
            logger.info("Creating visualization plots...")
            
            # Prediction distributions
            y_pred_proba = model.model.predict(X_test)
            fig = visualizer.plot_prediction_distributions(
                y_test, y_pred_proba, model_name="InsiderThreatModel"
            )
            plot_file = os.path.join(args.output_dir, "prediction_distributions.png")
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction distributions plot saved to {plot_file}")
            
            # ROC and PR curves
            eval_dict = {"InsiderThreatModel": evaluation_results}
            
            fig = visualizer.plot_roc_curves(eval_dict)
            roc_file = os.path.join(args.output_dir, "roc_curve.png")
            fig.savefig(roc_file, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {roc_file}")
            
            fig = visualizer.plot_precision_recall_curves(eval_dict)
            pr_file = os.path.join(args.output_dir, "pr_curve.png")
            fig.savefig(pr_file, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {pr_file}")
            
            # Confusion matrix
            fig = visualizer.plot_confusion_matrices(eval_dict)
            cm_file = os.path.join(args.output_dir, "confusion_matrix.png")
            fig.savefig(cm_file, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {cm_file}")
            
            # Threshold analysis
            if 'threshold_analysis' in evaluation_results:
                fig = visualizer.plot_threshold_analysis(evaluation_results)
                if fig:
                    threshold_file = os.path.join(args.output_dir, "threshold_analysis.png")
                    fig.savefig(threshold_file, dpi=300, bbox_inches='tight')
                    logger.info(f"Threshold analysis saved to {threshold_file}")
        
        # Save predictions if requested
        if args.save_predictions:
            logger.info("Saving predictions...")
            y_pred_proba = model.model.predict(X_test)
            y_pred = (y_pred_proba > threshold).astype(int)
            
            predictions_df = {
                'true_labels': y_test.flatten().tolist(),
                'predicted_probabilities': y_pred_proba.flatten().tolist(),
                'predicted_labels': y_pred.flatten().tolist(),
                'threshold_used': threshold
            }
            
            pred_file = os.path.join(args.output_dir, "predictions.json")
            with open(pred_file, 'w') as f:
                json.dump(predictions_df, f, indent=2)
            logger.info(f"Predictions saved to {pred_file}")
        
        # Print summary
        logger.info("[SUCCESS] Evaluation completed successfully!")
        logger.info("Summary:")
        
        basic_metrics = evaluation_results.get('basic_metrics', {})
        for metric, value in basic_metrics.items():
            logger.info(f"  - {metric.capitalize()}: {value:.4f}")
        
        advanced_metrics = evaluation_results.get('advanced_metrics', {})
        if 'roc_auc' in advanced_metrics and advanced_metrics['roc_auc']:
            logger.info(f"  - ROC AUC: {advanced_metrics['roc_auc']:.4f}")
        if 'pr_auc' in advanced_metrics:
            logger.info(f"  - PR AUC: {advanced_metrics['pr_auc']:.4f}")
        
        # Performance assessment
        summary = evaluation_results.get('evaluation_summary', {})
        logger.info(f"  - Overall Performance: {summary.get('overall_performance', 'Unknown')}")
        
        logger.info(f"All results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
