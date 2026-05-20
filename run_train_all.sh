#!/bin/bash
# ============================================================
# Colonomind - Run train_all.py in Background (Safe from SSH disconnects)
# ============================================================

# 1. Create logs directory if it doesn't exist
mkdir -p logs

# 2. Get local pip-installed Nvidia libraries and add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn, nvidia.cublas, nvidia.cuda_runtime, os; print(':'.join([os.path.dirname(m.__file__) + '/lib' for m in (nvidia.cudnn, nvidia.cublas, nvidia.cuda_runtime)])):$LD_LIBRARY_PATH")

# 3. Launch training on GPU 6 in the background
CUDA_VISIBLE_DEVICES=6 nohup python src/train_all.py --output_dir ./results/all_datasets > logs/train_all.log 2>&1 &

# 4. Show success message
PID=$!
echo "=========================================================="
echo "🚀 Training started in background on GPU 6!"
echo "   Process ID (PID): $PID"
echo "   Output log: logs/train_all.log"
echo "=========================================================="
echo "You can safely disconnect from SSH now. The training will continue."
echo "To monitor training, run: tail -f logs/train_all.log"
echo "=========================================================="
