#!/bin/bash
# PARC ROBOTICS - Start AI Chat Server
# Loads LFM2.5 Thinking model on port 8080

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"

LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
AI_MODEL="$MODEL_DIR/general/lfm2.5-1.2b-thinking-q4_k_m.gguf"

echo "Starting LFM2.5 AI Chat server..."

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    exit 1
fi

if [ ! -f "$AI_MODEL" ]; then
    echo "ERROR: AI model not found at $AI_MODEL"
    exit 1
fi

# Kill existing if running
if [ -f "$LOG_DIR/llama-ai.pid" ]; then
    kill $(cat "$LOG_DIR/llama-ai.pid") 2>/dev/null || true
fi

# Start server
nohup "$LLAMA_SERVER" \
    -m "$AI_MODEL" \
    -c 2048 \
    -ngl 99 \
    --host 0.0.0.0 \
    --port 8080 \
    > "$LOG_DIR/llama-ai.log" 2>&1 &

PID=$!
echo $PID > "$LOG_DIR/llama-ai.pid"
echo "LFM2.5 AI started (PID: $PID) on http://localhost:8080"
