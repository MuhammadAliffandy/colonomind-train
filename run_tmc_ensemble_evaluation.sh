#!/bin/bash

# Script to run TMC-UCM Ensemble Voting Evaluation in the background (nohup)

LOG_FILE="ensemble_evaluation.log"

echo "================================================================="
echo "🚀 Starting Ensemble Voting Evaluation in the background..."
echo "📂 Output will be logged to: $LOG_FILE"
echo "================================================================="

# Run the python script in the background using nohup and redirect output to log file (Force using GPU 3)
CUDA_VISIBLE_DEVICES=3 nohup python src/evaluate_tmc_ensemble_voting.py > "$LOG_FILE" 2>&1 &

# Capture the process ID (PID)
PID=$!
echo "✅ Process started with PID: $PID"
echo "👀 To monitor the progress in real-time, run: tail -f $LOG_FILE"
echo "🛑 To stop the process, run: kill $PID"
