#!/bin/bash
# PARC ROBOTICS - Stop All Services
# Stops Flask + AI + Vision + Voice services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
SERVICES_FILE="$LOG_DIR/.services"

echo "============================================"
echo "   Stopping All Services..."
echo "============================================"
echo ""

# Function to stop a process by PID
stop_process() {
    local pid=$1
    local name=$2
    
    if [ -n "$pid" ] && [ "$pid" != "" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping $name (PID: $pid)..."
        kill -TERM "$pid" 2>/dev/null
        
        # Wait up to 10 seconds for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "  ✓ $name stopped gracefully"
                return 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        echo "  ⚠ $name did not stop gracefully, forcing..."
        kill -9 "$pid" 2>/dev/null
        sleep 1
        echo "  ✓ $name force stopped"
    else
        echo "  ○ $name not running (no valid PID)"
    fi
}

# Function to stop via individual PID file
stop_by_pid_file() {
    local pid_file=$1
    local name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            stop_process "$pid" "$name"
        else
            echo "  ○ $name not running (stale PID file)"
        fi
        rm -f "$pid_file"
    fi
}

# Load PIDs from .services file if it exists
if [ -f "$SERVICES_FILE" ]; then
    echo "Loading service PIDs from .services..."
    source "$SERVICES_FILE"
    echo ""
fi

# Stop Flask (main app)
echo "→ Stopping Flask App..."
if [ -n "$FLASK_PID" ] && kill -0 "$FLASK_PID" 2>/dev/null; then
    stop_process "$FLASK_PID" "Flask"
else
    stop_by_pid_file "$LOG_DIR/flask.pid" "Flask"
fi
rm -f "$LOG_DIR/flask.pid"
echo ""

# Stop AI Services
echo "→ Stopping AI Services..."
stop_by_pid_file "$LOG_DIR/llama-ai.pid" "LLAMA-AI"
echo ""

# Stop Vision Services
echo "→ Stopping Vision Services..."
stop_by_pid_file "$LOG_DIR/llama-vision.pid" "LLAMA-VISION"
echo ""

# Stop Voice Services (Whisper + Piper)
echo "→ Stopping Voice Services..."
stop_by_pid_file "$LOG_DIR/whisper.pid" "Whisper"
stop_by_pid_file "$LOG_DIR/piper.pid" "Piper"
echo ""

# Alternative: Try to stop via sub-scripts if they have stop functionality
echo "→ Running individual stop scripts (if available)..."
for script in stop-ai.sh stop-vision.sh stop-voice.sh; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        echo "  Executing $script..."
        bash "$SCRIPT_DIR/$script" 2>/dev/null || true
    fi
done
echo ""

# Clean up port conflicts (optional - kill processes on known ports)
echo "→ Cleaning up any remaining processes on service ports..."
for port in 5000 8080 8081 8084 8085; do
    pid=$(lsof -t -i:$port 2>/dev/null || netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "  Killing process on port $port (PID: $pid)"
        kill -9 "$pid" 2>/dev/null || true
    fi
done
echo ""

# Remove services file
rm -f "$SERVICES_FILE"

echo "============================================"
echo "   ✓ All Services Stopped"
echo "============================================"
echo ""
echo "Services stopped:"
echo "  - Flask App:   http://localhost:5000"
echo "  - AI Chat:     http://localhost:8080"
echo "  - Vision:      http://localhost:8081"
echo "  - Whisper:     http://localhost:8084"
echo "  - Piper:       http://localhost:8085"
echo ""
echo "Ready for restart!"