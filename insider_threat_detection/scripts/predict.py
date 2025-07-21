"""Prediction script for insider threat detection model."""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineer import FeatureEngineer
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.logger import get_logger
from src.models.lstm_model import InsiderThreatLSTM


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make Predictions with Insider Threat Detection Model')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved model file')
    
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory (default: from config)')
    
    parser.add_argument('--input-file', type=str, default=None,
                       help='Path to specific input CSV file')
    
    parser.add_argument('--output-file', type=str, default='predictions.csv',
                       help='Output file for predictions (default: predictions.csv)')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    
    parser.add_argument('--include-probabilities', action='store_true', default=True,
                       help='Include prediction probabilities in output (default: True)')
    
    parser.add_argument('--include-features', action='store_true', default=False,
                       help='Include feature values in output (default: False)')
    
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for prediction (default: 32)')
    
    parser.add_argument('--top-threats', type=int, default=None,
                       help='Show only top N threat predictions (default: None)')
    
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose logging (default: False)')
    
    return parser.parse_args()


def load_model_and_artifacts(model_path: str, checkpoint_manager: CheckpointManager):
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
        
        if not scaler:
            logger.warning("No scaler found - predictions may be inaccurate")
        if not feature_columns:
            logger.warning("No feature columns found - using default")
        
        logger.info("Model and artifacts loaded successfully")
        
        return model, scaler, label_encoders, feature_columns
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def prepare_prediction_data(data_path: str, input_file: str, scaler, label_encoders, feature_columns):
    """Prepare data for prediction."""
    logger = get_logger("data_preparation")
    
    try:
        # Initialize processors
        data_loader = DataLoader(data_path)
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        
        # Set loaded artifacts
        if scaler:
            preprocessor.scaler = scaler
        if label_encoders:
            feature_engineer.label_encoders = label_encoders
        
        # Load data
        if input_file:
            # Load specific file
            logger.info(f"Loading data from {input_file}")
            merged_df = pd.read_csv(input_file)
            # Add activity_type if not present
            if 'activity_type' not in merged_df.columns:
                merged_df['activity_type'] = 'UNKNOWN'
        else:
            # Load and merge all data
            logger.info("Loading and merging all data files")
            merged_df = data_loader.load_and_merge_data()
        
        # Store original data for output
        original_df = merged_df.copy()
        
        # Preprocessing
        merged_df = preprocessor.clean_dates(merged_df)
        merged_df = preprocessor.create_time_features(merged_df)
        merged_df = preprocessor.handle_missing_values(merged_df)
        merged_df = preprocessor.sort_and_prepare(merged_df)
        
        # Feature engineering
        merged_df = feature_engineer.create_user_behavior_features(merged_df)
        merged_df = feature_engineer.encode_categorical_features(merged_df)
        merged_df = feature_engineer.detect_anomalies(merged_df)
        merged_df = feature_engineer.create_advanced_features(merged_df)
        
        # Create sequences
        X, _ = preprocessor.create_sequences(merged_df, feature_columns)
        
        # Scale features
        X_scaled = preprocessor.scale_features(X, fit_scaler=False)
        
        logger.info(f"Prediction data prepared: {len(X_scaled)} sequences")
        
        return X_scaled, merged_df, original_df
        
    except Exception as e:
        logger.error(f"Error preparing prediction data: {e}")
        raise


def make_predictions(model, X, threshold: float, batch_size: int):
    """Make predictions using the model."""
    logger = get_logger("prediction")
    
    try:
        logger.info(f"Making predictions for {len(X)} sequences...")
        
        # Get prediction probabilities
        y_pred_proba = model.model.predict(X, batch_size=batch_size, verbose=1)
        
        # Convert to binary predictions
        y_pred = (y_pred_proba > threshold).astype(int)
        
        logger.info(f"Predictions completed")
        logger.info(f"Threat predictions: {y_pred.sum()} out of {len(y_pred)} ({y_pred.mean()*100:.1f}%)")
        
        return y_pred_proba.flatten(), y_pred.flatten()
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def create_prediction_output(original_df, merged_df, y_pred_proba, y_pred, 
                           feature_columns, threshold, include_probabilities, 
                           include_features, top_threats):
    """Create prediction output dataframe."""
    logger = get_logger("output_creation")
    
    try:
        # Create base output dataframe
        # Note: We need to handle the sequence-to-record mapping
        # For simplicity, we'll use the last record of each sequence
        
        # Get unique users and their last records
        user_last_records = merged_df.groupby('user').tail(1).reset_index(drop=True)
        
        # Ensure we have the right number of predictions
        n_predictions = min(len(y_pred_proba), len(user_last_records))
        
        output_df = user_last_records.head(n_predictions).copy()
        
        # Add predictions
        output_df['threat_prediction'] = y_pred[:n_predictions]
        
        if include_probabilities:
            output_df['threat_probability'] = y_pred_proba[:n_predictions]
            output_df['confidence_level'] = pd.cut(
                y_pred_proba[:n_predictions],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # Add risk assessment
        output_df['risk_level'] = 'Low'
        output_df.loc[y_pred_proba[:n_predictions] > 0.7, 'risk_level'] = 'High'
        output_df.loc[(y_pred_proba[:n_predictions] > 0.4) & (y_pred_proba[:n_predictions] <= 0.7), 'risk_level'] = 'Medium'
        
        # Add feature values if requested
        if include_features and feature_columns:
            available_features = [col for col in feature_columns if col in merged_df.columns]
            for feature in available_features:
                if feature in user_last_records.columns:
                    output_df[f'feature_{feature}'] = user_last_records[feature].head(n_predictions)
        
        # Add metadata
        output_df['prediction_threshold'] = threshold
        output_df['prediction_timestamp'] = pd.Timestamp.now()
        
        # Sort by threat probability (highest first)
        if include_probabilities:
            output_df = output_df.sort_values('threat_probability', ascending=False)
        
        # Filter top threats if requested
        if top_threats and top_threats > 0:
            output_df = output_df.head(top_threats)
            logger.info(f"Showing top {top_threats} threat predictions")
        
        logger.info(f"Output dataframe created with {len(output_df)} records")
        
        return output_df
        
    except Exception as e:
        logger.error(f"Error creating prediction output: {e}")
        raise


def generate_prediction_summary(output_df, y_pred_proba, y_pred, threshold):
    """Generate prediction summary statistics."""
    summary = {
        "prediction_summary": {
            "total_predictions": len(y_pred),
            "threat_predictions": int(y_pred.sum()),
            "normal_predictions": int(len(y_pred) - y_pred.sum()),
            "threat_percentage": float(y_pred.mean() * 100),
            "threshold_used": threshold
        },
        "probability_distribution": {
            "mean": float(y_pred_proba.mean()),
            "std": float(y_pred_proba.std()),
            "min": float(y_pred_proba.min()),
            "max": float(y_pred_proba.max()),
            "median": float(np.median(y_pred_proba))
        },
        "confidence_distribution": {
            "very_high_confidence": int((y_pred_proba > 0.8).sum()),
            "high_confidence": int(((y_pred_proba > 0.6) & (y_pred_proba <= 0.8)).sum()),
            "medium_confidence": int(((y_pred_proba > 0.4) & (y_pred_proba <= 0.6)).sum()),
            "low_confidence": int(((y_pred_proba > 0.2) & (y_pred_proba <= 0.4)).sum()),
            "very_low_confidence": int((y_pred_proba <= 0.2).sum())
        },
        "risk_distribution": {
            "high_risk": int((output_df['risk_level'] == 'High').sum()),
            "medium_risk": int((output_df['risk_level'] == 'Medium').sum()),
            "low_risk": int((output_df['risk_level'] == 'Low').sum())
        }
    }
    
    return summary


def main():
    """Main prediction function."""
    args = parse_arguments()
    
    # Setup logging
    logger = get_logger("predict_script")
    
    if args.verbose:
        logger.info("Verbose logging enabled")
        logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Initialize components
        checkpoint_manager = CheckpointManager()
        
        # Load model and artifacts
        logger.info("Loading model and artifacts...")
        model, scaler, label_encoders, feature_columns = load_model_and_artifacts(
            args.model_path, checkpoint_manager
        )
        
        # Prepare prediction data
        logger.info("Preparing prediction data...")
        X, merged_df, original_df = prepare_prediction_data(
            args.data_path, args.input_file, scaler, label_encoders, feature_columns
        )
        
        # Make predictions
        logger.info(f"Making predictions with threshold {args.threshold}...")
        y_pred_proba, y_pred = make_predictions(
            model, X, args.threshold, args.batch_size
        )
        
        # Create output
        logger.info("Creating prediction output...")
        output_df = create_prediction_output(
            original_df, merged_df, y_pred_proba, y_pred, feature_columns,
            args.threshold, args.include_probabilities, args.include_features,
            args.top_threats
        )
        
        # Save predictions
        output_df.to_csv(args.output_file, index=False)
        logger.info(f"Predictions saved to {args.output_file}")
        
        # Generate and save summary
        summary = generate_prediction_summary(output_df, y_pred_proba, y_pred, args.threshold)
        
        summary_file = args.output_file.replace('.csv', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Prediction summary saved to {summary_file}")
        
        # Print summary
        logger.info("[TARGET] Prediction completed successfully!")
        logger.info("Summary:")
        logger.info(f"  - Total predictions: {summary['prediction_summary']['total_predictions']}")
        logger.info(f"  - Threat predictions: {summary['prediction_summary']['threat_predictions']}")
        logger.info(f"  - Threat percentage: {summary['prediction_summary']['threat_percentage']:.1f}%")
        logger.info(f"  - Average probability: {summary['probability_distribution']['mean']:.3f}")
        
        # Show top threats if any
        if args.include_probabilities and len(output_df) > 0:
            logger.info("Top threat predictions:")
            top_5 = output_df.head(5)
            for idx, row in top_5.iterrows():
                logger.info(f"  - User: {row['user']}, Probability: {row['threat_probability']:.3f}, Risk: {row['risk_level']}")
        
        logger.info(f"Results saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
