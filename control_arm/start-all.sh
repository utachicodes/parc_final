#!/bin/bash
# PARC ROBOTICS - Start All Services
# Starts Flask + AI + Vision + Voice

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stop any existing
$SCRIPT_DIR/stop.sh 2>/dev/null || true
sleep 1

# Start all
echo "Starting all services..."
echo ""

# Start AI
if [ -f "$SCRIPT_DIR/start-ai.sh" ]; then
    bash "$SCRIPT_DIR/start-ai.sh"
fi

# Start Vision
if [ -f "$SCRIPT_DIR/start-vision.sh" ]; then
    bash "$SCRIPT_DIR/start-vision.sh"
fi

# Start Voice
if [ -f "$SCRIPT_DIR/start-voice.sh" ]; then
    bash "$SCRIPT_DIR/start-voice.sh"
fi

# Start Flask
echo ""
echo "Starting Flask..."
cd "$SCRIPT_DIR"
LOG_DIR="$SCRIPT_DIR/logs"
DOCKER_COMPOSE_FILE="/home/parc/simulation/simulation/compose.yaml"
source "$SCRIPT_DIR/.venv/bin/activate"
nohup .venv/bin/python app.py --host 0.0.0.0 --port 5000 > "$LOG_DIR/flask.log" 2>&1 &
FLASK_PID=$!
echo $FLASK_PID > "$LOG_DIR/flask.pid"
echo "Flask started (PID: $FLASK_PID)"

# Update .services
cat > "$LOG_DIR/.services" << EOF
FLASK_PID=$FLASK_PID
LLAMA_AI_PID=$(cat "$LOG_DIR/llama-ai.pid" 2>/dev/null || echo "")
LLAMA_VISION_PID=$(cat "$LOG_DIR/llama-vision.pid" 2>/dev/null || echo "")
WHISPER_PID=$(cat "$LOG_DIR/whisper.pid" 2>/dev/null || echo "")
PIPER_PID=$(cat "$LOG_DIR/piper.pid" 2>/dev/null || echo "")
EOF
docker rm -f $(docker ps -aq)
docker network prune -f

echo "ENTER PASSWORD:"
sudo systemctl restart docker.service

docker compose -f /home/parc/simulation/compose.yaml up simulation bridge -d 
echo ""
echo "============================================"
echo "   All Services Started"
echo "============================================"
echo ""
echo "  - AI Chat:     http://localhost:8080"
echo "  - Vision:      http://localhost:8081"
echo "  - Whisper:     http://localhost:8084"
echo "  - Piper:       http://localhost:8085"
echo "  - Flask App:   http://localhost:5000"
echo ""
echo "Ready!"
