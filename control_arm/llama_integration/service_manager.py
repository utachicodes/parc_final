#!/usr/bin/env python3
"""
PARC Service Manager - Auto-starts AI services when needed
Lazy loading: services start only when first API call is made
"""

import subprocess
import os
import time
import logging

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_DIR = os.path.expanduser("~/models")

LLAMA_SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
WHISPER_BIN = os.path.expanduser("~/whisper.cpp/build/bin/whisper-server")
PIPER_BIN = os.path.expanduser("~/miniconda3/bin/piper")

SERVICES = {
    "ai": {
        "port": 8080,
        "model": os.path.join(MODEL_DIR, "general/lfm2.5-1.2b-thinking-q4_k_m.gguf"),
        "log": os.path.join(LOG_DIR, "llama-ai.log"),
        "pid_file": os.path.join(LOG_DIR, "llama-ai.pid"),
    },
    "vision": {
        "port": 8081,
        "model": os.path.join(MODEL_DIR, "vision/LFM2.5-VL-450M-Q4_0.gguf"),
        "log": os.path.join(LOG_DIR, "llama-vision.log"),
        "pid_file": os.path.join(LOG_DIR, "llama-vision.pid"),
    },
    "whisper": {
        "port": 8084,
        "model": os.path.join(MODEL_DIR, "whisper/ggml-base.bin"),
        "log": os.path.join(LOG_DIR, "whisper.log"),
        "pid_file": os.path.join(LOG_DIR, "whisper.pid"),
        "binary": WHISPER_BIN,
    },
    "piper": {
        "port": 8085,
        "model": os.path.join(MODEL_DIR, "voices/en_US-lessac-medium.onnx"),
        "log": os.path.join(LOG_DIR, "piper.log"),
        "pid_file": os.path.join(LOG_DIR, "piper.pid"),
        "binary": PIPER_BIN,
    },
}


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("127.0.0.1", port))
            return result == 0
    except Exception:
        return False


def is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_pid(pid_file: str) -> int | None:
    """Get PID from file if process is running."""
    try:
        if os.path.exists(pid_file):
            with open(pid_file) as f:
                pid = int(f.read().strip())
            if is_process_running(pid):
                return pid
    except (ValueError, OSError):
        pass
    return None


def start_service(name: str) -> bool:
    """Start a service if not already running."""
    svc = SERVICES.get(name)
    if not svc:
        logger.error(f"Unknown service: {name}")
        return False

    # Check if already running
    port = svc["port"]
    if is_port_in_use(port):
        logger.debug(f"Service {name} already running on port {port}")
        return True

    # Check PID file
    pid = get_pid(svc["pid_file"])
    if pid and is_process_running(pid):
        logger.debug(f"Service {name} already running (PID: {pid})")
        return True

    # Check binary exists
    binary = svc.get("binary", LLAMA_SERVER)
    if not os.path.exists(binary):
        logger.warning(f"Binary not found for {name}: {binary}")
        return False

    # Check model exists
    model = svc.get("model")
    if model and not os.path.exists(model):
        logger.warning(f"Model not found for {name}: {model}")
        return False

    # Start the service
    logger.info(f"Starting {name} service...")

    try:
        if name in ("ai", "vision"):
            # Llama server
            cmd = [
                binary,
                "-m", model,
                "-c", "2048",
                "-ngl", "99",
                "--host", "0.0.0.0",
                "--port", str(port),
            ]
        elif name == "whisper":
            cmd = [
                binary,
                "-m", model,
                "--host", "0.0.0.0",
                "--port", str(port),
            ]
        elif name == "piper":
            cmd = [
                binary,
                "--model", model,
                "--port", str(port),
                "--log_file", svc["log"],
            ]

        with open(svc["log"], "w") as f:
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

        # Wait a bit for startup
        time.sleep(2)

        # Check if started
        if is_port_in_use(port):
            logger.info(f"{name} service started on port {port}")
            # Save PID
            import subprocess as sp
            result = sp.run(["pgrep", "-f", f"--port {port}"], capture_output=True, text=True)
            pids = result.stdout.strip().split("\n")
            if pids and pids[0]:
                with open(svc["pid_file"], "w") as f:
                    f.write(pids[0])
            return True
        else:
            logger.error(f"Failed to start {name} service")
            return False

    except Exception as e:
        logger.error(f"Error starting {name}: {e}")
        return False


def ensure_service(name: str) -> bool:
    """Ensure service is running, start if not."""
    if not is_port_in_use(SERVICES[name]["port"]):
        return start_service(name)
    return True


def stop_service(name: str) -> bool:
    """Stop a service."""
    svc = SERVICES.get(name)
    if not svc:
        return False

    pid = get_pid(svc["pid_file"])
    if pid:
        try:
            os.kill(pid, 9)
            logger.info(f"Stopped {name} service (PID: {pid})")
        except OSError:
            pass

    # Also try by port
    try:
        import subprocess as sp
        sp.run(["fuser", "-k", f"{svc['port']}/tcp"], capture_output=True)
    except Exception:
        pass

    return True


def stop_all():
    """Stop all services."""
    for name in SERVICES:
        stop_service(name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test - start all services
    for name in SERVICES:
        start_service(name)
