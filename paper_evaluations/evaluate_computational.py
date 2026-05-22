import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_hybrid_model
from src.config import IMG_SIZE

def get_model_size_mb(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='../results/all_datasets')
    parser.add_argument('--gpu', type=str, default=None)
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        
    print("=== Computational Performance Evaluation (Minor 3) ===")
    
    num_classes = 4
    model = build_hybrid_model((IMG_SIZE[0], IMG_SIZE[1], 3), (20,), num_classes)
    
    model_path = os.path.join(args.model_dir, 'best_hybrid_model.h5')
    if os.path.exists(model_path):
        model.load_weights(model_path)
    else:
        print(f"Warning: {model_path} not found. Using untrained weights for parameter count.")
        
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_count = trainable_count + non_trainable_count
    
    model_size = get_model_size_mb(model_path)
    
    # Measure Latency
    print("Running 1000 warm-up inferences...")
    dummy_img = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
    dummy_feat = np.random.rand(1, 20).astype(np.float32)
    
    for _ in range(100):
        _ = model.predict([dummy_img, dummy_feat], verbose=0)
        
    print("Measuring inference latency over 1000 runs...")
    times = []
    for _ in range(1000):
        start = time.time()
        _ = model.predict([dummy_img, dummy_feat], verbose=0)
        times.append(time.time() - start)
        
    avg_latency_ms = np.mean(times) * 1000
    std_latency_ms = np.std(times) * 1000
    
    print("\n" + "="*50)
    print(" COMPUTATIONAL PERFORMANCE REPORT ")
    print("="*50)
    print(f"Model Architecture    : Colonomind Hybrid Mod-SE(2)")
    print(f"Total Parameters      : {total_count:,}")
    print(f"Trainable Params      : {trainable_count:,}")
    print(f"Non-Trainable Params  : {non_trainable_count:,}")
    print(f"Model File Size       : {model_size:.2f} MB")
    print(f"Hardware Inference    : GPU {args.gpu if args.gpu else 'CPU/Default'}")
    print(f"Avg Inference Latency : {avg_latency_ms:.2f} ± {std_latency_ms:.2f} ms per frame")
    print(f"Equivalent FPS        : {1000 / avg_latency_ms:.1f} FPS")
    print("="*50)
    
    with open('../paper_results/computational_performance.txt', 'w') as f:
        f.write(f"Model Architecture: Colonomind Hybrid Mod-SE(2)\n")
        f.write(f"Total Parameters: {total_count:,}\n")
        f.write(f"Model File Size: {model_size:.2f} MB\n")
        f.write(f"Avg Inference Latency: {avg_latency_ms:.2f} ms per frame\n")
