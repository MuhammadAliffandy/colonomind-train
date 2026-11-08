#!/bin/bash
# ==============================================================================
# COLONOMIND v19 — AUTOMATED PACKAGING & XXDRIVE TRANSFER SCRIPT
# ==============================================================================
set -e

cd "$(dirname "$0")"

echo "======================================================================"
echo "📦 PACKAGING COLONOMIND v19 FOR XXDRIVE & WEB INTEGRATION"
echo "======================================================================"

TARGET_DEST="$HOME/xxdrive"
if [ ! -d "$TARGET_DEST" ]; then
    echo "⚠️ Target $TARGET_DEST does not exist. Checking alternative mounts..."
    if [ -d "/home/D13K48009/xxdrive" ]; then
        TARGET_DEST="/home/D13K48009/xxdrive"
    else
        echo "ℹ️ Using local export directory: ./ColonoMind_V19_Export"
        TARGET_DEST="./ColonoMind_V19_Export"
    fi
fi

# If feedback_agent.txt or v19_evaluation.json is missing, generate them instantly (~30s)
if [ ! -f "../Result/ColonoMind_v19_NoiseRobust/feedback_agent.txt" ] || [ ! -f "../Result/ColonoMind_v19_NoiseRobust/v19_evaluation.json" ]; then
    echo "⚡ feedback_agent.txt or v19_evaluation.json not found yet."
    echo "🚀 Running fast zero-hang calibration & feedback generation (~30s)..."
    python src/train_colonomind_v19_noise_robust.py
fi

python src/package_v19_to_xxdrive.py --dest "$TARGET_DEST"

echo ""
echo "======================================================================"
echo "✅ PACKAGING COMPLETED!"
echo "======================================================================"
echo "Output files ready in:"
echo "  1. Directory : $TARGET_DEST/ColonoMind_V19_Package"
echo "  2. ZIP File  : $TARGET_DEST/colonomind_v19_package.zip"
echo "======================================================================"

