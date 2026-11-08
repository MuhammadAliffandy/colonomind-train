#!/bin/bash
# ==============================================================================
# COLONOMIND v19 FOCUSED — AUTO-HEALING & TRAINING LAUNCHER
# ==============================================================================
set -e

# Change to repo directory
cd "$(dirname "$0")"

echo "======================================================================"
echo "🔧 STEP 1: RESTORING CUDA 12 DRIVER COMPATIBILITY IN colono_env"
echo "======================================================================"

# 1. Purge conflicting CUDA 13 packages if any exist
CU13_PACKAGES=$(pip list 2>/dev/null | grep -E "cu13|13\." | awk '{print $1}' || true)
if [ -n "$CU13_PACKAGES" ]; then
    echo "🧹 Removing conflicting CUDA 13 packages..."
    pip list 2>/dev/null | grep -E "cu13|13\." | awk '{print $1}' | xargs -r pip uninstall -y || true
    pip uninstall -y torch torchvision 2>/dev/null || true
fi

# 2. Ensure official CUDA 12 packages for TensorFlow 2.21 are installed
echo "📦 Installing CUDA 12 runtime & cuDNN libraries for driver 550.90 (CUDA 12.4)..."
pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nvjitlink-cu12

# 3. Configure LD_LIBRARY_PATH to include nvidia pip libraries
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -d "$SITE_PACKAGES/nvidia" ]; then
    for sub in cuda_runtime cudnn cublas cufft curand cusolver cusparse nvjitlink; do
        if [ -d "$SITE_PACKAGES/nvidia/$sub/lib" ]; then
            export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/$sub/lib:$LD_LIBRARY_PATH"
        fi
    done
fi
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

echo "✅ CUDA 12 environment successfully configured!"

# 4. Clean up any stale background processes
pkill -f "train_colonomind_v19_noise_robust.py" 2>/dev/null || true
sleep 1

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

echo ""
echo "======================================================================"
echo "🧪 STEP 2: VERIFYING TENSORFLOW ON NVIDIA H100 (GPU 3)"
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
    a = tf.random.normal([1000, 1000])
    b = tf.matmul(a, a)
    c = tf.cast(b, tf.float16)
    print('✅ SUCCESS: Op:Cast & MatMul executed on GPU H100 flawlessly!')
"

SAVE_DIR="../Result/ColonoMind_v19_NoiseRobust"
LOG_FILE="colonomind_v19.log"

echo ""
echo "======================================================================"
echo "🚀 STEP 3: LAUNCHING COLONOMIND v19 IN BACKGROUND"
echo "======================================================================"
echo "Architecture : 3-Branch Mod-SE(2) + Knowledge Distillation (EndoViT Teacher)"
echo "Distillation : Soft Teacher Targets from 700k-image Foundation Model"
echo "Feedback     : Active Error Injection Loop (LightGBM feedback_agent.txt)"
echo "Epochs       : 40 Epochs (CDW-CE Loss + Optimizer EMA)"
echo "Save Root    : $SAVE_DIR"
echo "Log File     : $LOG_FILE"
echo "======================================================================"

# Launch fully detached using setsid + nohup + disown
setsid nohup python -u src/train_colonomind_v19_noise_robust.py \
    --save_dir "$SAVE_DIR" \
    --base_dir ../Result/ColonoMind_v18_Audit \
    --teacher_probs ../Result/ColonoMind_v19_teacher/teacher_probs_endovit.npz \
    --epochs 40 \
    < /dev/null > "$LOG_FILE" 2>&1 &

PID=$!
disown -h $PID 2>/dev/null || true

echo "✅ Pipeline successfully launched in background (PID: $PID)"
echo "🔒 100% IMMUNE to terminal closure, SSH disconnects, and Ctrl+C!"
echo ""
echo "📊 How to monitor execution live:"
echo "   tail -f $LOG_FILE"
echo ""
echo "💡 TIP: You can safely press Ctrl+C inside 'tail -f' to exit the log."
echo "   The background training process will KEEP RUNNING undisturbed!"
echo ""
echo "🛑 How to stop if needed:"
echo "   kill -9 $PID"
echo "======================================================================"
