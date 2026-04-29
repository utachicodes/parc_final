#!/usr/bin/env python3
"""
PARC ROBOTICS - Connectivity Checker
====================================
Verify all AI servers are running and responding.

Usage:
    python3 check_connectivity.py
"""

import json
import sys
import urllib.request
import urllib.error

PORTS = {
    5000: ("Flask Web UI", "http://localhost:5000/"),
    8080: ("LFM2.5 Thinking", "http://localhost:8080/v1/models"),
    8081: ("LFM2.5 Vision", "http://localhost:8081/v1/models"),
    8083: ("YOLOE Vision", None),  # Checked via Flask
    8084: ("Whisper STT", "http://localhost:8084/health"),
    8085: ("Piper TTS", "http://localhost:8085/health"),
}

def check_port(port):
    """Check if port is open."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(('localhost', port))
            return True
        except:
            return False

def check_url(url, timeout=5):
    """Check if URL returns 200."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except:
        return False

def main():
    print("=" * 60)
    print("PARC ROBOTICS - Connectivity Check")
    print("=" * 60)
    print()
    
    all_ok = True
    
    for port, (name, url) in PORTS.items():
        port_open = check_port(port)
        
        if url:
            url_ok = check_url(url)
        else:
            url_ok = port_open  # YOLOE is via Flask, just check port
        
        if port_open and url_ok:
            status = "OK"
            color = "\033[92m"
        elif port_open:
            status = "PORT OPEN"
            color = "\033[93m"
            all_ok = False
        else:
            status = "NOT RUNNING"
            color = "\033[91m"
            all_ok = False
        
        reset = "\033[0m"
        print(f"{color}[{status}]{reset} {name:20s} : {port}  {url or '(via Flask)'}")
    
    print()
    
    # Check AI status via Flask
    try:
        req = urllib.request.Request("http://localhost:5000/api/ai/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        
        print("AI Server Status:")
        for key, val in data.items():
            if isinstance(val, dict):
                connected = val.get("connected", val.get("available", False))
                model = val.get("model", val.get("voice", ""))
                port = val.get("port", "")
                status = "ON" if connected else "OFF"
                color = "\033[92m" if connected else "\033[91m"
                reset = "\033[0m"
                print(f"  {color}[{status}]{reset} {key:10s} port:{port} model:{model}")
    except Exception as e:
        print(f"Could not fetch AI status: {e}")
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("All systems operational!")
        return 0
    else:
        print("Some services not available - see above")
        return 1

if __name__ == "__main__":
    sys.exit(main())