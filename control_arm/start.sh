#!/bin/bash
# PARC ROBOTICS - Minimal Starter
# ================================
# Starts Flask only. AI services auto-start when needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

echo ""
echo "============================================"
echo "   PARC ROBOTICS - SO-101"
echo "============================================"
echo ""
echo -e "${BLUE}[PARC]${NC} Flask only - AI services start on demand"
echo ""

# Stop existing
$SCRIPT_DIR/stop.sh 2>/dev/null || true
sleep 1

# Start Flask
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"
nohup .venv/bin/python app.py --host 0.0.0.0 --port 5000 > "$LOG_DIR/flask.log" 2>&1 &
FLASK_PID=$!
echo $FLASK_PID > "$LOG_DIR/flask.pid"

echo -e "${GREEN}[OK]${NC} Flask started (PID: $FLASK_PID)"
echo ""
echo "  Web UI:    http://localhost:5000"
echo "  AI:        auto-starts on first chat"
echo "  Vision:    auto-starts on first analyze"
echo "  Voice:     auto-starts on first voice command"
echo ""
echo "  ./stop.sh  - Stop all"
echo ""
echo "Ready! Access the app at: http://localhost:5000"
