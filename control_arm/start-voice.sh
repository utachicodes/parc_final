#!/bin/bash
# PARC ROBOTICS - Start Voice Services
# Starts Whisper STT (8084) and Piper TTS (8085)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"

WHISPER_BIN="$HOME/whisper.cpp/build/bin/whisper-server"
WHISPER_MODEL="$MODEL_DIR/whisper/ggml-base.bin"
PIPER_BIN="$HOME/miniconda3/bin/piper"
PIPER_VOICE="$MODEL_DIR/voices/en_US-lessac-medium.onnx"

echo "Starting voice services..."

# Kill existing if running
[ -f "$LOG_DIR/whisper.pid" ] && kill $(cat "$LOG_DIR/whisper.pid") 2>/dev/null || true
[ -f "$LOG_DIR/piper.pid" ] && kill $(cat "$LOG_DIR/piper.pid") 2>/dev/null || true

# Start Whisper
if [ -f "$WHISPER_BIN" ] && [ -f "$WHISPER_MODEL" ]; then
    nohup "$WHISPER_BIN" \
        -m "$WHISPER_MODEL" \
        --host 0.0.0.0 \
        --port 8084 \
        > "$LOG_DIR/whisper.log" 2>&1 &
    PID=$!
    echo $PID > "$LOG_DIR/whisper.pid"
    echo "Whisper STT started (PID: $PID) on http://localhost:8084"
else
    echo "Whisper not available (binary or model missing)"
fi

# Start Piper
if [ -f "$PIPER_BIN" ] && [ -f "$PIPER_VOICE" ]; then
    nohup "$PIPER_BIN" \
        --model "$PIPER_VOICE" \
        --port 8085 \
        --log_file "$LOG_DIR/piper.log" \
        > "$LOG_DIR/piper.log" 2>&1 &
    PID=$!
    echo $PID > "$LOG_DIR/piper.pid"
    echo "Piper TTS started (PID: $PID) on http://localhost:8085"
else
    echo "Piper not available (binary or voice model missing)"
fi

echo "Voice services started"
