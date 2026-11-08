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

echo ""
echo "======================================================================"
echo "🧪 STEP 2: VERIFYING TENSORFLOW OP:CAST & MATMUL ON GPU 3 (NVIDIA H100)"
echo "======================================================================"
python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import site, ctypes, glob
for sp in site.getsitepackages():
    nvidia_dir = os.path.join(sp, 'nvidia')
    if os.path.exists(nvidia_dir):
        for so in sorted(glob.glob(os.path.join(nvidia_dir, '*', 'lib', '*.so*'))):
            try: ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except Exception: pass

import tensorflow as tf
print('TF Version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('Visible GPUs:', gpus)
assert len(gpus) > 0, 'No GPU detected by TensorFlow!'
print('✅ GPU verified successfully:', gpus[0].name)

with tf.device('/GPU:0'):
    a = tf.random.normal([2000, 2000])
    b = tf.matmul(a, a)
    c = tf.cast(b, tf.float16)
    print('✅ SUCCESS: GPU MatMul & Op:Cast executed flawlessly! Output shape:', c.shape)
"

echo "======================================================================"
echo "🎉 CUDA ENVIRONMENT 100% HEALTHY & READY!"
echo "======================================================================"
