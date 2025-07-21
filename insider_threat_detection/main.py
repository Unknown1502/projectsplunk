"""
Main entry point for the Insider Threat Detection System.
This is the refactored version of final.py with modular architecture.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.trainer import InsiderThreatTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import ModelVisualizer
from src.utils.gpu_setup import initialize_gpu_environment
from src.utils.logger import get_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Insider Threat Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    # Train with default settings
  python main.py train --use-lstm         # Train with LSTM instead of GRU
  python main.py train --epochs 50        # Train for 50 epochs
  python main.py evaluate --model-path path/to/model.h5
  python main.py predict --model-path path/to/model.h5 --input-file data.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-path', type=str, default=None,
                             help='Path to data directory')
    train_parser.add_argument('--use-lstm', action='store_true', default=False,
                             help='Use LSTM instead of GRU')
    train_parser.add_argument('--batch-size', type=int, default=16,
                             help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=30,
                             help='Maximum number of epochs')
    train_parser.add_argument('--resume', action='store_true', default=False,
                             help='Resume training from checkpoint')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to saved model file')
    eval_parser.add_argument('--data-path', type=str, default=None,
                            help='Path to data directory')
    eval_parser.add_argument('--output-dir', type=str, default='evaluation_results',
                            help='Directory to save evaluation results')
    eval_parser.add_argument('--threshold', type=float, default=0.5,
                            help='Classification threshold')
    
    # Prediction command
    pred_parser = subparsers.add_parser('predict', help='Make predictions')
    pred_parser.add_argument('--model-path', type=str, required=True,
                            help='Path to saved model file')
    pred_parser.add_argument('--input-file', type=str, required=True,
                            help='Path to input CSV file')
    pred_parser.add_argument('--output-file', type=str, default='predictions.csv',
                            help='Output file for predictions')
    pred_parser.add_argument('--threshold', type=float, default=0.5,
                            help='Classification threshold')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run complete demo pipeline')
    demo_parser.add_argument('--data-path', type=str, default=None,
                            help='Path to data directory')
    demo_parser.add_argument('--quick', action='store_true', default=False,
                            help='Run quick demo with reduced epochs')
    
    # Global arguments
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Enable verbose logging')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                       help='Disable GPU usage')
    
    return parser.parse_args()


def run_training(args):
    """Run training pipeline."""
    logger = get_logger("main_train")
    
    try:
        # Initialize trainer
        trainer = InsiderThreatTrainer(
            data_path=args.data_path,
            use_gru=not args.use_lstm
        )
        
        # Update configuration
        trainer.batch_size = args.batch_size
        trainer.max_epochs = args.epochs
        
        logger.info("Starting training pipeline...")
        results = trainer.run_complete_pipeline()
        
        if results['overall_success']:
            logger.info("[SUCCESS] Training completed successfully!")
            return 0
        else:
            logger.error("[FAIL] Training failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        return 1


def run_evaluation(args):
    """Run evaluation pipeline."""
    logger = get_logger("main_eval")
    
    try:
        # Import and run evaluation script
        from scripts.evaluate import main as eval_main
        
        # Prepare arguments for evaluation script
        sys.argv = [
            'evaluate.py',
            '--model-path', args.model_path,
            '--output-dir', args.output_dir,
            '--threshold', str(args.threshold)
        ]
        
        if args.data_path:
            sys.argv.extend(['--data-path', args.data_path])
        
        if args.verbose:
            sys.argv.append('--verbose')
        
        return eval_main()
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return 1


def run_prediction(args):
    """Run prediction pipeline."""
    logger = get_logger("main_pred")
    
    try:
        # Import and run prediction script
        from scripts.predict import main as pred_main
        
        # Prepare arguments for prediction script
        sys.argv = [
            'predict.py',
            '--model-path', args.model_path,
            '--input-file', args.input_file,
            '--output-file', args.output_file,
            '--threshold', str(args.threshold)
        ]
        
        if args.verbose:
            sys.argv.append('--verbose')
        
        return pred_main()
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 1


def run_demo(args):
    """Run complete demo pipeline."""
    logger = get_logger("main_demo")
    
    try:
        logger.info("[START] Starting Insider Threat Detection Demo")
        logger.info("=" * 60)
        
        # Step 1: Training
        logger.info("Step 1: Training Model")
        logger.info("-" * 30)
        
        trainer = InsiderThreatTrainer(data_path=args.data_path, use_gru=False)
        
        if args.quick:
            trainer.max_epochs = 5
            logger.info("Quick demo mode: Training for 5 epochs only")
        
        training_results = trainer.run_complete_pipeline()
        
        if not training_results['overall_success']:
            logger.error("Demo failed during training")
            return 1
        
        # Step 2: Evaluation
        logger.info("\nStep 2: Model Evaluation")
        logger.info("-" * 30)
        
        # Get the trained model
        model = trainer.model.model
        X_test = training_results.get('test_data', {}).get('X')
        y_test = training_results.get('test_data', {}).get('y')
        
        if X_test is not None and y_test is not None:
            evaluator = ModelEvaluator()
            eval_results = evaluator.evaluate_model(model, X_test, y_test, model_name="DemoModel")
            
            # Print evaluation summary
            logger.info("Evaluation Results:")
            basic_metrics = eval_results.get('basic_metrics', {})
            for metric, value in basic_metrics.items():
                logger.info(f"  - {metric.capitalize()}: {value:.4f}")
        
        # Step 3: Visualization
        logger.info("\nStep 3: Creating Visualizations")
        logger.info("-" * 30)
        
        if training_results.get('training_results', {}).get('training_history'):
            visualizer = ModelVisualizer()
            history = training_results['training_results']['training_history']
            
            # Create training history plot
            fig = visualizer.plot_training_history(history)
            fig.savefig('demo_training_history.png', dpi=300, bbox_inches='tight')
            logger.info("Training history plot saved to: demo_training_history.png")
        
        logger.info("\n[SUCCESS] Demo completed successfully!")
        logger.info("=" * 60)
        logger.info("Demo Results Summary:")
        logger.info(f"  - Model trained successfully")
        logger.info(f"  - Evaluation completed")
        logger.info(f"  - Visualizations created")
        logger.info("Check the generated files for detailed results.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    logger = get_logger("main")
    
    if args.verbose:
        logger.info("Verbose logging enabled")
        logger.info(f"Command: {args.command}")
    
    # Initialize GPU environment
    if not args.no_gpu:
        gpu_available = initialize_gpu_environment()
        logger.info(f"GPU initialization: {'Success' if gpu_available else 'Failed'}")
    
    # Route to appropriate function
    if args.command == 'train':
        return run_training(args)
    elif args.command == 'evaluate':
        return run_evaluation(args)
    elif args.command == 'predict':
        return run_prediction(args)
    elif args.command == 'demo':
        return run_demo(args)
    else:
        logger.error("No command specified. Use --help for usage information.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
