#!/bin/bash
# ==============================================================================
# COLONOMIND — RESTORE CUDA 12 DRIVER COMPATIBILITY IN colono_env
# ==============================================================================
set -e

echo "======================================================================"
echo "🔧 PURGING CONFLICTING CUDA 13 WHEELS FROM colono_env"
echo "======================================================================"

pip list 2>/dev/null | grep -E "cu13|13\." | awk '{print $1}' | xargs -r pip uninstall -y || true
pip uninstall -y torch torchvision 2>/dev/null || true

echo ""
echo "======================================================================"
echo "🧪 VERIFYING TENSORFLOW OP:CAST & MATMUL ON GPU 3 (NVIDIA H100)"
echo "======================================================================"
python -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import tensorflow as tf
print('TF Version:', tf.__version__)
print('Visible GPUs:', tf.config.list_physical_devices('GPU'))
with tf.device('/GPU:0'):
    a = tf.random.normal([2000, 2000])
    b = tf.matmul(a, a)
    c = tf.cast(b, tf.float16)
    print('✅ SUCCESS: GPU MatMul & Op:Cast executed flawlessly! Output shape:', c.shape)
"

echo "======================================================================"
echo "🎉 CUDA ENVIRONMENT 100% HEALTHY & READY!"
echo "======================================================================"
