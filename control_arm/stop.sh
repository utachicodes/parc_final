#!/bin/bash
# PARC ROBOTICS - Stop All Services
# ================================

LOG_DIR="$(dirname "$0")/logs"

if [ -f "$LOG_DIR/.services" ]; then
    source "$LOG_DIR/.services"
fi

echo "Stopping PARC ROBOTICS services..."

# Kill Flask
if [ -n "$FLASK_PID" ] && kill -0 $FLASK_PID 2>/dev/null; then
    kill $FLASK_PID 2>/dev/null && echo "Flask stopped" || true
fi

# Kill Llama AI
if [ -n "$LLAMA_AI_PID" ] && kill -0 $LLAMA_AI_PID 2>/dev/null; then
    kill $LLAMA_AI_PID 2>/dev/null && echo "LFM2.5 AI stopped" || true
fi

# Kill Llama Vision
if [ -n "$LLAMA_VISION_PID" ] && kill -0 $LLAMA_VISION_PID 2>/dev/null; then
    kill $LLAMA_VISION_PID 2>/dev/null && echo "LFM2.5 Vision stopped" || true
fi

# Kill Whisper
if [ -n "$WHISPER_PID" ] && kill -0 $WHISPER_PID 2>/dev/null; then
    kill $WHISPER_PID 2>/dev/null && echo "Whisper STT stopped" || true
fi

# Kill Piper
if [ -n "$PIPER_PID" ] && kill -0 $PIPER_PID 2>/dev/null; then
    kill $PIPER_PID 2>/dev/null && echo "Piper TTS stopped" || true
fi

# Also try fuser for ports
fuser -k 5000/tcp 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
fuser -k 8081/tcp 2>/dev/null || true
fuser -k 8084/tcp 2>/dev/null || true
fuser -k 8085/tcp 2>/dev/null || true

echo "All services stopped."
