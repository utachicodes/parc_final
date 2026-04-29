#!/bin/bash
# PARC ROBOTICS - Start Vision Server
# Loads LFM2.5 Vision model on port 8081

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"

LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
VISION_MODEL="$MODEL_DIR/vision/LFM2.5-VL-450M-Q4_0.gguf"

echo "Starting LFM2.5 Vision server..."

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    exit 1
fi

if [ ! -f "$VISION_MODEL" ]; then
    echo "ERROR: Vision model not found at $VISION_MODEL"
    exit 1
fi

# Kill existing if running
if [ -f "$LOG_DIR/llama-vision.pid" ]; then
    kill $(cat "$LOG_DIR/llama-vision.pid") 2>/dev/null || true
fi

# Start server
nohup "$LLAMA_SERVER" \
    -m "$VISION_MODEL" \
    -c 2048 \
    -ngl 99 \
    --host 0.0.0.0 \
    --port 8081 \
    > "$LOG_DIR/llama-vision.log" 2>&1 &

PID=$!
echo $PID > "$LOG_DIR/llama-vision.pid"
echo "LFM2.5 Vision started (PID: $PID) on http://localhost:8081"
