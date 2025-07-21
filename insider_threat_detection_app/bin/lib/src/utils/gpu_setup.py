"""GPU setup and configuration utilities."""

import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import RANDOM_SEED, ENABLE_MIXED_PRECISION, ENABLE_GPU_MEMORY_GROWTH
from src.utils.console_utils import safe_print, print_success, print_error, print_warning


def setup_reproducibility():
    """Set random seeds for reproducibility."""
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def check_gpu_availability():
    """Check GPU availability and provide detailed information."""
    try:
        # Check physical devices
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print_error("No GPU devices found by TensorFlow")
            safe_print("Possible solutions:")
            safe_print("1. Install CUDA-enabled TensorFlow: pip install tensorflow[and-cuda]")
            safe_print("2. Check NVIDIA drivers are installed")
            safe_print("3. Verify CUDA and cuDNN installation")
            return False, []
        
        # Get detailed GPU information
        gpu_details = []
        for i, gpu in enumerate(gpus):
            try:
                # Get GPU name and memory info
                gpu_name = gpu.name if hasattr(gpu, 'name') else f"GPU {i}"
                gpu_details.append({
                    'index': i,
                    'name': gpu_name,
                    'device': gpu
                })
                
                # Try to get memory info
                try:
                    memory_info = tf.config.experimental.get_memory_info(gpu)
                    gpu_details[i]['memory'] = memory_info
                except:
                    gpu_details[i]['memory'] = None
                    
            except Exception as e:
                safe_print(f"Warning: Could not get details for GPU {i}: {e}")
                gpu_details.append({
                    'index': i,
                    'name': f"GPU {i}",
                    'device': gpu,
                    'memory': None
                })
        
        return True, gpu_details
        
    except Exception as e:
        print_error(f"Error checking GPU availability: {e}")
        return False, []


def setup_gpu():
    """Configure GPU memory growth to prevent memory issues."""
    try:
        gpu_available, gpu_details = check_gpu_availability()
        
        if not gpu_available:
            print_error("No GPU found - Training will use CPU")
            return False
        
        # Configure each GPU
        configured_gpus = 0
        for gpu_info in gpu_details:
            gpu = gpu_info['device']
            gpu_name = gpu_info['name']
            
            try:
                if ENABLE_GPU_MEMORY_GROWTH:
                    # Enable memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    safe_print(f"   Memory growth enabled for {gpu_name}")
                
                # Set memory limit if needed (optional)
                # tf.config.experimental.set_memory_limit(gpu, 1024)  # 1GB limit
                
                configured_gpus += 1
                
            except Exception as e:
                print_warning(f"Could not configure {gpu_name}: {e}")
        
        if configured_gpus > 0:
            print_success(f"GPU ready: {configured_gpus} GPU(s) configured")
            
            # Print detailed GPU information
            for gpu_info in gpu_details:
                gpu_name = gpu_info['name']
                safe_print(f"   {gpu_name}")
                
                if gpu_info['memory']:
                    try:
                        current = gpu_info['memory']['current'] / (1024**3)  # Convert to GB
                        peak = gpu_info['memory']['peak'] / (1024**3)
                        safe_print(f"     Memory - Current: {current:.2f}GB, Peak: {peak:.2f}GB")
                    except:
                        pass
            
            return True
        else:
            print_error("No GPUs could be configured - Training will use CPU")
            return False
            
    except Exception as e:
        print_error(f"GPU setup failed: {e}")
        safe_print("   Training will continue on CPU")
        return False


def setup_mixed_precision():
    """Set up mixed precision for better performance on compatible GPUs."""
    if not ENABLE_MIXED_PRECISION:
        return False
        
    try:
        # Check if GPU supports mixed precision
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print_warning("Mixed precision disabled - no GPU available")
            return False
        
        # Set mixed precision policy
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        print_success("Mixed precision enabled for better performance")
        safe_print(f"   Compute dtype: {policy.compute_dtype}")
        safe_print(f"   Variable dtype: {policy.variable_dtype}")
        
        return True
        
    except Exception as e:
        print_warning(f"Mixed precision not available: {e}")
        safe_print("   Continuing with float32")
        return False


def test_gpu_computation():
    """Test basic GPU computation to verify setup."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return False
        
        # Test basic computation on GPU
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            
        safe_print(f"   GPU computation test: {c.shape} tensor created successfully")
        return True
        
    except Exception as e:
        print_warning(f"GPU computation test failed: {e}")
        return False


def initialize_gpu_environment():
    """Initialize complete GPU environment."""
    safe_print("Initializing GPU environment...")
    
    # Set up reproducibility
    setup_reproducibility()
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    if gpu_available:
        # Setup mixed precision if GPU is available
        mixed_precision_enabled = setup_mixed_precision()
        
        # Test GPU computation
        gpu_test_passed = test_gpu_computation()
        
        if not gpu_test_passed:
            print_warning("GPU test failed, but GPU setup completed")
    else:
        safe_print("Continuing with CPU-only configuration")
    
    return gpu_available
