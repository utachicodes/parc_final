#!/usr/bin/env python3
"""
PARC ROBOTICS - Startup Script
==============================
Launches all AI servers and web application.

Usage:
    python3 start_parc.py [--mode dev|prod] [--skip-ai]

Modes:
    dev   - Start with debugging, local model paths
    prod  - Production mode, expect models in ~/models/
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = Path.home() / "models"

# Server configurations
SERVERS = {
    "ai_thinking": {
        "port": 8080,
        "model": MODEL_DIR / "general" / "lfm2.5-1.2b-thinking-q4_k_m.gguf",
        "cmd": ["/home/parc/bin/llama-server", "-m", str(MODEL_DIR / "general" / "lfm2.5-1.2b-thinking-q4_k_m.gguf"), "-c", "2048", "--port", "8080"],
        "timeout": 60,
        "health": "http://localhost:8080/v1/models"
    },
    "ai_vision": {
        "port": 8081,
        "model": MODEL_DIR / "vision" / "LFM2.5-VL-450M-Q4_0.gguf",
        "cmd": ["/home/parc/bin/llama-server", "-m", str(MODEL_DIR / "vision" / "LFM2.5-VL-450M-Q4_0.gguf"), "-c", "2048", "--port", "8081"],
        "timeout": 60,
        "health": "http://localhost:8081/v1/models"
    },
    "whisper_stt": {
        "port": 8084,
        "model": MODEL_DIR / "whisper" / "ggml-base.bin",
        "script": SCRIPT_DIR / "llama_integration" / "whisper_server.py",
        "cmd": ["python3", str(SCRIPT_DIR / "llama_integration" / "whisper_server.py"), "--port", "8084"],
        "timeout": 20,
        "health": "http://localhost:8084/health"
    },
    "piper_tts": {
        "port": 8085,
        "model": MODEL_DIR / "voices" / "en_US-lessac-medium.onnx",
        "script": SCRIPT_DIR / "llama_integration" / "piper_server.py",
        "cmd": ["python3", str(SCRIPT_DIR / "llama_integration" / "piper_server.py"), "--port", "8085"],
        "timeout": 20,
        "health": "http://localhost:8085/health"
    },
    "yoloe_vision": {
        "port": 8083,
        "model": MODEL_DIR / "yoloe" / "yoloe-11s-seg.pt",
        "script": SCRIPT_DIR / "llama_integration" / "vision_server.py",
        "cmd": ["python3", str(SCRIPT_DIR / "llama_integration" / "vision_server.py"), "--port", "8083"],
        "timeout": 30,
        "health": None  # YOLOE is called via Flask, not standalone
    }
}


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗  ██████╗ ██████╗ ████████╗ ██████╗  ██████╗ ██████╗   ║
║   ██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔═══██╗██╔══██╗  ║
║   ██████╔╝██║   ██║██████╔╝   ██║   ██║   ██║██║   ██║██████╔╝  ║
║   ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██║   ██║██║   ██║██╔══██╗  ║
║   ██║     ╚██████╔╝██║  ██║   ██║   ╚██████╔╝╚██████╔╝██║  ██║  ║
║   ╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝  ╚═════╝ ╚═╝  ╚═╝  ║
║                                                              ║
║   {Colors.ENDC}ROBOTICS - SO-101 Robot Arm Controller{Colors.CYAN}                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{Colors.ENDC}""")


def print_status(msg, status="INFO"):
    colors = {
        "INFO": Colors.BLUE,
        "OK": Colors.GREEN,
        "WARN": Colors.YELLOW,
        "ERROR": Colors.RED
    }
    color = colors.get(status, Colors.ENDC)
    print(f"{color}[{status}]{Colors.ENDC} {msg}")


def check_port(port):
    """Check if port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(('localhost', port))
            return True
        except:
            return False


def check_health(url, timeout=5):
    """Check if a service is healthy."""
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except:
        return False


def wait_for_service(port, timeout=30):
    """Wait for a service to become available."""
    start = time.time()
    while time.time() - start < timeout:
        if check_port(port):
            time.sleep(0.5)  # Give it a moment to fully initialize
            return True
        time.sleep(0.5)
    return False


def start_server(name, config, skip_ai=False):
    """Start a single server."""
    if skip_ai and name in ["ai_thinking", "ai_vision", "whisper_stt", "piper_tts"]:
        print_status(f"Skipping {name} (--skip-ai flag)", "WARN")
        return None

    port = config["port"]
    
    # Check if already running
    if check_port(port):
        print_status(f"{name} already running on port {port}", "OK")
        return None

    # Check if model exists
    if "model" in config:
        model_path = Path(config["model"])
        if not model_path.exists():
            print_status(f"Model not found: {model_path}", "WARN")
            print_status(f"  Download from HuggingFace to enable {name}", "WARN")
            return None

    # Start the server
    print_status(f"Starting {name} on port {port}...")
    
    try:
        if "script" in config:
            # Python-based server
            proc = subprocess.Popen(
                config["cmd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
        else:
            # llama-server
            proc = subprocess.Popen(
                config["cmd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
        
        # Wait for it to be ready
        if wait_for_service(port, config.get("timeout", 30)):
            print_status(f"{name} ready on port {port}", "OK")
            return proc
        else:
            print_status(f"{name} failed to start (timeout)", "ERROR")
            proc.kill()
            return None
            
    except FileNotFoundError as e:
        print_status(f"{name} executable not found: {e}", "ERROR")
        return None
    except Exception as e:
        print_status(f"{name} error: {e}", "ERROR")
        return None


def start_flask():
    """Start the Flask web application."""
    print_status("Starting Flask web application on port 5000...")
    
    env = os.environ.copy()
    env["FLASK_ENV"] = "development" if args.mode == "dev" else "production"
    
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT_DIR / "app.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True
    )
    
    if wait_for_service(5000, 15):
        print_status("Flask web UI ready at http://localhost:5000", "OK")
        return proc
    else:
        print_status("Flask failed to start", "ERROR")
        return None


def print_summary(processes):
    """Print startup summary."""
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗
║                    SYSTEM STATUS                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║""")
    
    services = [
        ("LFM2.5 Thinking", 8080, processes.get("ai_thinking")),
        ("LFM2.5 Vision", 8081, processes.get("ai_vision")),
        ("Whisper STT", 8084, processes.get("whisper_stt")),
        ("Piper TTS", 8085, processes.get("piper_tts")),
        ("YOLOE Vision", 8083, None),  # Via Flask
        ("Flask Web UI", 5000, processes.get("flask")),
    ]
    
    all_ok = True
    for name, port, proc in services:
        if proc or (port == 8083):
            status = f"{Colors.GREEN}RUNNING{Colors.ENDC}"
        elif port == 8083:
            status = f"{Colors.YELLOW}VIA FLASK{Colors.ENDC}"
        else:
            status = f"{Colors.RED}NOT STARTED{Colors.ENDC}"
            all_ok = False
        
        print(f"║   {name:20s} : {port:5d}  [{status}]")
    
    print(f"""║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║""")
    
    if all_ok:
        print(f"║   {Colors.GREEN}Web UI: http://localhost:5000{Colors.ENDC}                      ║")
    else:
        print(f"║   {Colors.YELLOW}Some services not available - check model paths{Colors.ENDC}        ║")
    
    print(f"""║                                                              ║
║   Press Ctrl+C to stop all services                         ║
╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")


def cleanup(processes):
    """Stop all processes."""
    print_status("Shutting down all services...", "WARN")
    for name, proc in processes.items():
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                proc.kill()
    print_status("All services stopped", "OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PARC ROBOTICS Startup Script")
    parser.add_argument("--mode", choices=["dev", "prod"], default="dev", help="Run mode")
    parser.add_argument("--skip-ai", action="store_true", help="Skip AI servers (for testing UI)")
    parser.add_argument("--flask-only", action="store_true", help="Start only Flask, no AI servers")
    args = parser.parse_args()

    print_banner()
    
    processes = {}
    
    if not args.flask_only:
        # Start AI servers
        for name, config in SERVERS.items():
            proc = start_server(name, config, args.skip_ai)
            if proc:
                processes[name] = proc
        
        # Small delay between servers
        time.sleep(1)
    
    # Start Flask
    proc = start_flask()
    if proc:
        processes["flask"] = proc
    
    print_summary(processes)
    
    # Wait for Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(processes)