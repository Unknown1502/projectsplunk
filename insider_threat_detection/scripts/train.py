"""Training script for insider threat detection model."""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import InsiderThreatTrainer
from src.utils.gpu_setup import initialize_gpu_environment
from src.utils.logger import get_logger
from src.utils.console_utils import setup_console_encoding, safe_print, print_success, print_error


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Insider Threat Detection Model')
    
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory (default: from config)')
    
    parser.add_argument('--use-lstm', action='store_true', default=False,
                       help='Use LSTM instead of GRU (default: False)')
    
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    
    parser.add_argument('--epochs', type=int, default=30,
                       help='Maximum number of epochs (default: 30)')
    
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume training from checkpoint (default: False)')
    
    parser.add_argument('--force-reload', action='store_true', default=False,
                       help='Force reload data even if checkpoint exists (default: False)')
    
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU if available (default: True)')
    
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose logging (default: False)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Setup console encoding for Unicode support
    setup_console_encoding()
    
    args = parse_arguments()
    
    # Setup logging
    logger = get_logger("train_script")
    
    if args.verbose:
        logger.info("Verbose logging enabled")
        logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Initialize GPU environment
        if args.gpu:
            gpu_available = initialize_gpu_environment()
            logger.info(f"GPU initialization: {'Success' if gpu_available else 'Failed'}")
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = InsiderThreatTrainer(
            data_path=args.data_path,
            use_gru=not args.use_lstm
        )
        
        # Update training configuration
        trainer.batch_size = args.batch_size
        trainer.max_epochs = args.epochs
        
        logger.info(f"Training configuration:")
        logger.info(f"  - Model type: {'GRU' if args.use_lstm else 'LSTM'}")
        logger.info(f"  - Batch size: {args.batch_size}")
        logger.info(f"  - Max epochs: {args.epochs}")
        logger.info(f"  - Resume training: {args.resume}")
        
        # Run training pipeline
        logger.info("Starting training pipeline...")
        results = trainer.run_complete_pipeline()
        
        # Print results
        if results['overall_success']:
            print_success("Training completed successfully!")
            
            # Print final metrics
            if 'training_results' in results and 'final_metrics' in results['training_results']:
                final_metrics = results['training_results']['final_metrics']
                logger.info("Final Test Metrics:")
                for metric, value in final_metrics.get('basic_metrics', {}).items():
                    logger.info(f"  - {metric.capitalize()}: {value:.4f}")
                
                if 'roc_auc' in final_metrics.get('advanced_metrics', {}):
                    roc_auc = final_metrics['advanced_metrics']['roc_auc']
                    logger.info(f"  - ROC AUC: {roc_auc:.4f}")
            
            # Print training summary
            summary = trainer.get_training_summary()
            logger.info("Training Summary:")
            logger.info(f"  - Total sequences: {summary['data_info']['total_sequences']}")
            logger.info(f"  - Feature count: {summary['data_info']['feature_count']}")
            logger.info(f"  - Threat ratio: {summary['data_info']['threat_ratio']:.3f}")
            
            if 'training_history' in summary:
                history = summary['training_history']
                logger.info(f"  - Epochs completed: {history['epochs_completed']}")
                logger.info(f"  - Final loss: {history['final_loss']:.4f}")
                logger.info(f"  - Best val loss: {history['best_val_loss']:.4f}")
        
        else:
            print_error("Training failed!")
            
            # Print error details
            for stage, stage_results in results.items():
                if isinstance(stage_results, dict) and not stage_results.get('success', True):
                    logger.error(f"  - {stage}: {stage_results.get('error', 'Unknown error')}")
            
            return 1
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("Checkpoint should be saved automatically")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
