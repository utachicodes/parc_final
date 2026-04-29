# Llama.cpp Build Guide for Jetson Orin Nano
============================================

## System Requirements

- **Jetson Orin Nano Super Developer Kit** (8GB LPDDR5)
- **CPU**: ARM Cortex-A78AE (64-bit) @ 1.4 GHz
- **GPU**: NVIDIA Ampere (2048 CUDA cores) @ 0.3 GHz
- **OS**: JetPack 5.x or 6.x (Ubuntu 20.04/22.04 based)

## Quick Build

Run on your Jetson Orin Nano:

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# CPU + GPU (CUDA) build
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_NATIVE=OFF \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

# Install
sudo cp build/bin/llama-server /usr/local/bin/
sudo cp build/bin/llama-cli /usr/local/bin/
```

## Download Models

### 1. Vision Model (Port 8081) - Camera Analysis
For object detection, scene understanding, pick-and-place assistance.

```bash
mkdir -p ~/models/vision
cd ~/models/vision

# LFM2.5-VL-1.6B (3.2GB) - Full vision model
wget https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/lfm2.5-vl-1.6b-f16.gguf

# LFM2.5-VL-450M (900MB) - Fast vision model
wget https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/lfm2.5-vl-450m-f16.gguf
```

### 2. Audio Model (Port 8082) - Voice Commands
For speech recognition and voice-controlled operations.

```bash
mkdir -p ~/models/audio
cd ~/models/audio

# LFM2.5-Audio-1.5B (3.0GB)
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/resolve/main/lfm2.5-audio-1.5b-f16.gguf
```

### 3. Thinking Model (Port 8080) - Reasoning
For complex reasoning, planning, and RAG chatbot.

```bash
mkdir -p ~/models/general
cd ~/models/general

# LFM2.5-1.2B-Thinking (800MB Q4_K_M) - Recommended
wget https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/resolve/main/lfm2.5-1.2b-thinking-q4_k_m.gguf
```

## Start Servers

```bash
# Terminal 1: General AI (Port 8080)
llama-server \
    -m ~/models/general/lfm2.5-1.2b-thinking-q4_k_m.gguf \
    -c 2048 \
    -ngl 99 \
    --host 0.0.0.0 \
    --port 8080

# Terminal 2: Vision AI (Port 8081)
llama-server \
    -m ~/models/vision/lfm2.5-vl-1.6b-f16.gguf \
    -c 2048 \
    -ngl 99 \
    --host 0.0.0.0 \
    --port 8081

# Note: Audio now uses Whisper.cpp + Piper instead of LFM2.5-Audio
# See sections below for audio setup (ports 8084 and 8085)
```

## Whisper.cpp - Multilingual STT

**Latest version: v1.8.4** (March 2026)

Whisper.cpp provides speech-to-text with 100+ language support, replacing LFM2.5-Audio (English-only).

### Build Whisper.cpp for Jetson (CUDA + cuBLAS)

```bash
cd ~
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp

# Build with CUDA support for GPU acceleration
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

# Install server
sudo cp build/bin/whisper-server /usr/local/bin/
sudo cp build/bin/whisper-cli /usr/local/bin/
```

### GPU Memory Note

For Jetson Orin Nano 8GB, use `tiny` or `base` model:
- `tiny.bin` - 75 MB, ~273 MB RAM
- `base.bin` - 142 MB, ~388 MB RAM  
- `small.bin` - 466 MB, ~852 MB RAM (may OOM on 8GB)

### Run Whisper Server

```bash
# Start server on port 8084
whisper-server \
    -m ~/models/whisper/ggml-base.bin \
    -t 4 \
    --port 8084 \
    --host 0.0.0.0

# With GPU acceleration (if CUDA properly configured)
whisper-server \
    -m ~/models/whisper/ggml-base.bin \
    -t 4 \
    -ngl 99 \
    --port 8084 \
    --host 0.0.0.0
```

### Download Whisper Models

```bash
mkdir -p ~/models/whisper
cd ~/models/whisper

# tiny (75MB) - Fastest, lower accuracy
wget https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin -O ggml-tiny.bin

# base (142MB) - Good balance (RECOMMENDED)
wget https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.bin -O ggml-base.bin

# base.en (142MB) - English only, better for English
wget https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -O ggml-base.en.bin

# small (466MB) - Better accuracy, needs more RAM
wget https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-small.bin -O ggml-small.bin
```

## Piper TTS - Multilingual Speech Synthesis

**Latest version: onnxruntime + piper-tts**

Piper provides text-to-speech with low latency and natural voices.

### Install Piper

```bash
# Install piper-tts
pip install piper-tts numpy

# Download voices
mkdir -p ~/models/voices
cd ~/models/voices

# English (US) - medium quality
wget https://github.com/rhasspy/piper/raw/master/voices/en_US-lessac-medium.onnx
wget https://github.com/rhasspy/piper/raw/master/voices/en_US-lessac-medium.onnx.json

# English (UK)
wget https://github.com/rhasspy/piper/raw/master/voices/en_GB-alan-medium.onnx
wget https://github.com/rhasspy/piper/raw/master/voices/en_GB-alan-medium.onnx.json

# French
wget https://github.com/rhasspy/piper/raw/master/voices/fr_FR-siwis-medium.onnx
wget https://github.com/rhasspy/piper/raw/master/voices/fr_FR-siwis-medium.onnx.json

# German
wget https://github.com/rhasspy/piper/raw/master/voices/de_DE-karlsson-medium.onnx
wget https://github.com/rhasspy/piper/raw/master/voices/de_DE-karlsson-medium.onnx.json

# Spanish
wget https://github.com/rhasspy/piper/raw/master/voices/es_ES-sharvard-medium.onnx
wget https://github.com/rhasspy/piper/raw/master/voices/es_ES-sharvard-medium.onnx.json
```

### Run Piper Server

```bash
# Start server on port 8085
python3 ~/PARC-Remote-Lab/llama_integration/piper_server.py \
    --voice ~/models/voices/en_US-lessac-medium.onnx \
    --port 8085
```

### Available Voices

| Voice ID | Language | Style |
|----------|----------|-------|
| en_US-lessac-medium | English (US) | Medium |
| en_US-lessac-low | English (US) | Low quality |
| en_GB-alan-medium | English (UK) | Medium |
| fr_FR-siwis-medium | French | Medium |
| de_DE-karlsson-medium | German | Medium |
| es_ES-sharvard-medium | Spanish | Medium |
| it_IT-palia-medium | Italian | Medium |

Full list: https://github.com/rhasspy/piper/blob/master/VOICES.md

### Start Audio Servers

```bash
# Terminal 4: Whisper STT (Port 8084)
whisper-server \
    -m ~/models/whisper/ggml-base.bin \
    -t 4 \
    --port 8084

# Terminal 5: Piper TTS (Port 8085)
# Use piper-tts Python API or simple HTTP server
python3 -c "
from piper import PiperVoice
import http.server
import socketserver

voice = PiperVoice('~/models/voices/en_US-lessac-medium.onnx')

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        import json
        length = int(self.headers.get('content-length', 0))
        data = json.loads(self.rfile.read(length))
        wav = voice.synthesize(data['text'])
        self.send_response(200)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Content-Length', len(wav))
        self.end_headers()
        self.wfile.write(wav)

with socketserver.TCPServer(('', 8085), Handler) as httpd:
    httpd.serve_forever()
"
```

### Supported Languages

Whisper.cpp supports 100+ languages including:
- English, French, Spanish, German, Italian, Portuguese
- Chinese, Japanese, Korean, Arabic, Russian
- Many more...

Piper voices available: en_US, en_GB, fr_FR, de_DE, es_ES, it_IT, etc.

## Verify Servers

```bash
# Check all models
curl http://localhost:8080/v1/models
curl http://localhost:8081/v1/models
curl http://localhost:8084/health
curl http://localhost:8085/voices

# Test chat
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "lfm2.5-1.2b-thinking-q4_k_m",
        "messages": [{"role": "user", "content": "How do I pick up an object?"}]
    }'
```

## YOLOE Object Detection (Port 8083)

For precise pick-and-place operations, use YOLOE for real-time object detection + segmentation.

### Install Ultralytics

```bash
pip install ultralytics
```

### Download YOLOE Model

```bash
mkdir -p ~/models/yoloe
cd ~/models/yoloe

# YOLOE-11S-SEG (7.3MB) - Recommended for Jetson
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg.pt

# YOLOE-26S-SEG (26MB) - Higher accuracy but slower
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt
```

### Export to TensorRT (Optional but recommended)

For best performance on Jetson, export to TensorRT:

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-11s-seg.pt")
model.export(format="engine")  # Creates yoloe-11s-seg.engine
```

### Run YOLOE Detection Server

```bash
# Start as standalone server on port 8083
python3 ~/PARC-Remote-Lab/llama_integration/vision_server.py
```

Or use the Flask integrated endpoint at `/api/vision/yoloe/detect`.

### YOLOE vs LFM2.5-VL

| Feature | YOLOE | LFM2.5-VL |
|---------|-------|-----------|
| Speed | 161 FPS | 5-10 FPS |
| Detection | Bounding boxes + masks | Text description only |
| Use case | Real-time pick-and-place | Scene understanding, reasoning |
| Custom objects | Text prompts | Not supported |

**Recommendation**: Use YOLOE for "where is the object" (real-time detection) and LFM2.5-VL for "what should I do with it" (reasoning).

## Systemd Service (Recommended)

Create `/etc/systemd/system/llama-parc.service`:

```ini
[Unit]
Description=PARC ROBOTICS AI Servers
After=network.target

[Service]
Type=simple
User=parc
WorkingDirectory=/home/parc

# General AI
ExecStartPre=/bin/sleep 2
ExecStart=/usr/local/bin/llama-server \
    -m /home/parc/models/general/lfm2.5-1.2b-thinking-q4_k_m.gguf \
    -c 2048 -ngl 99 --port 8080

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable llama-parc
sudo systemctl start llama-parc
```

## Performance Tips

| Model | RAM Usage | GPU Layers | Context |
|-------|-----------|------------|---------|
| VL-1.6B | ~4GB | 99 | 2048 |
| VL-450M | ~1.5GB | 99 | 2048 |
| Thinking-1.2B | ~1GB | 99 | 2048 |
| Whisper (base) | ~1GB | N/A (CPU) | N/A |

For 8GB Jetson, run one server at a time or use Q4_K_M quantization.

## Troubleshooting

**Out of Memory**:
```bash
# Reduce GPU layers
-nngl 33

# Or reduce context
-c 1024
```

**Slow Inference**:
```bash
# Force FP32 compute
GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1 llama-server ...
```

**Check GPU**:
```bash
nvidia-smi dmon
```

## PARC ROBOTICS Integration

The Flask app connects to:
- Port 8080: General AI chat and RAG (LFM2.5-Thinking)
- Port 8081: Vision camera analysis (LFM2.5-VL)
- Port 8084: Speech-to-text (Whisper.cpp) - multilingual
- Port 8085: Text-to-speech (Piper) - multilingual
- Port 8083: Object detection via YOLOE (via Flask /api/vision/yoloe/detect)

### Quick Test

```bash
# Test Whisper STT
curl -X POST http://localhost:8084/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d '{"model": "base", "audio_url": "data:audio/wav;base64,..."}'

# Test YOLOE detection
curl -X POST http://localhost:5000/api/vision/yoloe/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'

# Test TTS
curl -X POST http://localhost:8085/speak \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from PARC ROBOTICS", "voice": "en_US-lessac-medium"}'
```

## Hybrid Vision + Audio Architecture

```
Camera Feed
     │
     ├─────────────► YOLOE ──────────────────────────────────┐
     │                     │                                   │
     │                     ▼                                   │
     │              Object positions                          │
     │                     │                                   │
     │                     └──────────────► LFM2.5-VL ─────────┤
     │                                               │         │
     │                                               ▼         │
     │                                    "Pick up the red     │
     │                                         cube"           │
     │                                                           │
Microphone ──────────► Whisper.cpp (8084) ──► LFM2.5-Thinking ─┘
     │                     │                      │
     │                     ▼                      ▼
     │              "Pick up red"          Parse command
     │                                    │
     └────────────────────────────────────┤
                                          ▼
                                   Robot execution
                                          
Piper TTS (8085) ◄──── Response generation
     │
     ▼
Speech output
```

See `llama_integration/__init__.py` for API details.