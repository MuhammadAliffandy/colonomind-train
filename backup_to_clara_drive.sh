#!/bin/bash
# ==============================================================================
# ColonoMind: Backup All Experimental Results & Figures to clara_drive (rclone)
# ==============================================================================
# Usage:
#   bash backup_to_clara_drive.sh                     # Full backup (all results + weights)
#   bash backup_to_clara_drive.sh --exclude-h5         # Lightweight (plots, metrics, logs only)
#   bash backup_to_clara_drive.sh --remote my_drive:   # Custom rclone remote
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="${BACKUP_DIR:-./backups}"
REMOTE="${REMOTE:-clara_drive:ColonoMind_Backups}"
ARCHIVE_NAME="colonomind_all_results_${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="${BACKUP_DIR}/${ARCHIVE_NAME}"

mkdir -p "$BACKUP_DIR"

echo "=============================================================================="
echo "📦 ColonoMind Results & Figures Backup Pipeline"
echo "=============================================================================="
echo "📅 Date & Time     : $(date)"
echo "📍 Archive Path    : ${ARCHIVE_PATH}"
echo "☁️  Rclone Target   : ${REMOTE}"
echo "=============================================================================="

# 1. Detect candidate Result directories
TARGET_DIRS=()

# Priority candidates
CANDIDATES=(
    "Result"
    "../Result"
    "/home/D13K48009/raid/Clara/colonomind-train/Result"
    "/home/D13K48009/raid/Clara/Result"
    "/raid/D13K48009/Clara/colonomind-train/Result"
    "/raid/D13K48009/Clara/Result"
    "Manuscript_Consistent_Results"
    "Manuscript_Mock_Results"
    "Archive_Results"
    "Dummy_ROC_Images"
    "NewFallbackFigure"
)

for d in "${CANDIDATES[@]}"; do
    if [ -d "$d" ]; then
        # Avoid duplicate canonical paths
        CANON_PATH=$(cd "$d" && pwd)
        ALREADY_ADDED=0
        for existing in "${TARGET_DIRS[@]}"; do
            if [ "$existing" == "$CANON_PATH" ]; then
                ALREADY_ADDED=1
                break
            fi
        done
        if [ $ALREADY_ADDED -eq 0 ]; then
            TARGET_DIRS+=("$CANON_PATH")
            echo "  ✅ Included directory: $CANON_PATH"
        fi
    fi
done

if [ ${#TARGET_DIRS[@]} -eq 0 ]; then
    echo "❌ Error: No Result or Experiment directories found to backup!"
    exit 1
fi

# 2. Check for parallel compressor (pigz)
COMPRESS_FLAG="-czf"
if command -v pigz &> /dev/null; then
    echo "⚡ Multi-core compressor 'pigz' found. Using parallel compression."
    COMPRESS_FLAG="-I pigz -cf"
fi

# Check optional arguments
EXTRA_TAR_FLAGS=()
for arg in "$@"; do
    if [ "$arg" == "--exclude-h5" ]; then
        echo "⚠️ Flag --exclude-h5 passed: Excluding .h5 and .keras heavy weights."
        EXTRA_TAR_FLAGS+=(--exclude='*.h5' --exclude='*.keras')
    elif [[ "$arg" =~ ^--remote= ]]; then
        REMOTE="${arg#*=}"
    fi
done

# 3. Create .tar.gz
echo ""
echo "🗜️  Compressing all results into tar.gz..."
tar "${EXTRA_TAR_FLAGS[@]}" $COMPRESS_FLAG "$ARCHIVE_PATH" "${TARGET_DIRS[@]}"

ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
echo "✅ Compression completed! File size: ${ARCHIVE_SIZE}"

# 4. Verify tar.gz integrity
echo ""
echo "🔍 Verifying archive integrity..."
tar -tzf "$ARCHIVE_PATH" | head -n 5
echo "✅ Archive integrity test passed!"

# 5. Check rclone and transfer
echo ""
echo "🚀 Transferring to rclone remote: ${REMOTE} ..."
if ! command -v rclone &> /dev/null; then
    echo "❌ Error: 'rclone' is not installed or not in PATH."
    echo "   Local archive is saved safely at: ${ARCHIVE_PATH}"
    exit 1
fi

REMOTE_NAME=$(echo "$REMOTE" | cut -d':' -f1)
echo "🔎 Checking remote '${REMOTE_NAME}' in rclone config..."
if ! rclone listremotes | grep -q "^${REMOTE_NAME}:"; then
    echo "⚠️ Warning: '${REMOTE_NAME}:' was not found in 'rclone listremotes'."
    echo "   Available remotes:"
    rclone listremotes
    echo "   Continuing transfer attempt in case it's a dynamic or aliased remote..."
fi

# Run rclone with multi-threading and retries
rclone copy "$ARCHIVE_PATH" "$REMOTE" \
    --progress \
    --transfers=4 \
    --checkers=8 \
    --retries=5 \
    --low-level-retries=10 \
    --stats=5s

echo ""
echo "=============================================================================="
echo "🎉 SUCCESS! All ColonoMind experimental results backed up & transferred!"
echo "   Archive: ${ARCHIVE_NAME} (${ARCHIVE_SIZE})"
echo "   Remote : ${REMOTE}"
echo "=============================================================================="
