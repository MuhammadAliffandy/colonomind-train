#!/bin/bash
# ==============================================================================
# COLONOMIND — RESTORE CUDA 12 DRIVER COMPATIBILITY IN colono_env
# ==============================================================================
set -e

echo "======================================================================"
echo "🔧 STEP 1: PURGING CUDA 13 & INSTALLING CUDA 12 LIBRARIES"
echo "======================================================================"

pip list 2>/dev/null | grep -E "cu13|13\." | awk '{print $1}' | xargs -r pip uninstall -y || true
pip uninstall -y torch torchvision 2>/dev/null || true

pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nvjitlink-cu12

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -d "$SITE_PACKAGES/nvidia" ]; then
    for sub in cuda_runtime cudnn cublas cufft curand cusolver cusparse nvjitlink; do
        if [ -d "$SITE_PACKAGES/nvidia/$sub/lib" ]; then
            export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/$sub/lib:$LD_LIBRARY_PATH"
        fi
    done
fi
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=3

echo ""
echo "======================================================================"
echo "🧪 STEP 2: VERIFYING TENSORFLOW KERNEL ON GPU 3 (NVIDIA H100)"
echo "======================================================================"
python -c "
import tensorflow as tf
print('TF Version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('Visible GPUs:', gpus)
assert len(gpus) > 0, 'No GPU detected by TensorFlow!'

for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.matmul(a, a)
    print('✅ SUCCESS: GPU H100 MatMul verified! Result:\n', b.numpy())
"

echo "======================================================================"
echo "🎉 CUDA ENVIRONMENT 100% HEALTHY & READY!"
echo "======================================================================"
