import time
import numpy as np
import tensorflow as tf
import os

def benchmark_model(model_path, input_shapes):
    """
    Measures computational performance report:
    - Model Size (MB)
    - Number of Parameters
    - Hardware used
    - Inference Latency (ms)
    """
    print("\n" + "="*50)
    print(" COMPUTATIONAL PERFORMANCE REPORT")
    print("="*50)
    
    # 1. Hardware
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"Hardware        : {len(physical_devices)} GPUs Detected")
    
    # 2. Model Size & Params
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}. Please train first.")
        return
        
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    model = tf.keras.models.load_model(model_path, compile=False)
    params = model.count_params()
    
    print(f"Model Size      : {model_size_mb:.2f} MB")
    print(f"Total Parameters: {params:,}")
    
    # 3. Inference Latency (ms)
    # Generate dummy data
    dummy_inputs = []
    for shape in input_shapes:
        # shape usually includes batch size dimension, e.g., (None, 224, 224, 3)
        actual_shape = (1,) + shape[1:]
        dummy_inputs.append(np.random.rand(*actual_shape).astype(np.float32))
        
    # Warmup
    for _ in range(10):
        model.predict(dummy_inputs, verbose=0)
        
    # Benchmark
    iterations = 100
    start_time = time.time()
    for _ in range(iterations):
        model.predict(dummy_inputs, verbose=0)
    end_time = time.time()
    
    avg_latency = ((end_time - start_time) / iterations) * 1000 # to ms
    print(f"Avg Latency/Img : {avg_latency:.2f} ms")
    print("="*50 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to best_hybrid_model.h5")
    args = parser.parse_args()
    
    benchmark_model(args.model_path, input_shapes=[(None, 224, 224, 3), (None, 20), (None, 2)])
