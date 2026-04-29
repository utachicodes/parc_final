#!/usr/bin/env python3
"""
Audio Server for PARC ROBOTICS
===============================
Combined Whisper.cpp (STT) + Piper (TTS) server

Usage:
    python3 audio_server.py [--stt-model ggml-base.bin] [--tts-voice en_US-lessac-medium]

Ports:
    - STT: 8084
    - TTS: 8085
"""

import argparse
import base64
import json
import logging
import os
import signal
import sys
import threading
import wave
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhisperSTT:
    """Whisper.cpp-based speech-to-text."""

    def __init__(self, model_path: str = "ggml-base.bin", language: str = "auto"):
        self.model_path = model_path
        self.language = language
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Load Whisper model."""
        if not os.path.exists(self.model_path):
            logger.error(f"STT model not found: {self.model_path}")
            return False

        try:
            from whispercpp import Whisper
            self._model = Whisper(self.model_path)
            logger.info(f"Whisper STT loaded: {self.model_path}")
            return True
        except ImportError:
            logger.warning("whispercpp not installed. Install with: pip install whispercpp")
            return False
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            return False

    def is_available(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio to text."""
        if not self._model:
            return {"error": "Model not loaded", "text": ""}

        with self._lock:
            try:
                import numpy as np

                # Decode audio
                if audio_data[:4] == b'RIFF':
                    with wave.open(BytesIO(audio_data)) as wf:
                        frames = wf.readframes(wf.getnframes())
                        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe
                lang = self.language if self.language != "auto" else None
                text = self._model.transcribe(audio_np, language=lang)

                return {"text": text.strip(), "language": self.language or "auto"}
            except Exception as e:
                logger.error(f"Transcribe error: {e}")
                return {"error": str(e), "text": ""}


class PiperTTS:
    """Piper text-to-speech."""

    def __init__(self, voice_path: str = "en_US-lessac-medium.onnx"):
        self.voice_path = voice_path
        self._voice = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Load Piper voice."""
        if not os.path.exists(self.voice_path):
            logger.error(f"TTS voice not found: {self.voice_path}")
            return False

        try:
            from piper import PiperVoice
            self._voice = PiperVoice(self.voice_path)
            logger.info(f"Piper TTS loaded: {self.voice_path}")
            return True
        except ImportError:
            logger.warning("piper-tts not installed. Install with: pip install piper-tts")
            return False
        except Exception as e:
            logger.error(f"Failed to load Piper: {e}")
            return False

    def is_available(self) -> bool:
        return self._voice is not None

    def speak(self, text: str) -> bytes:
        """Synthesize text to speech."""
        if not self._voice:
            return b"TTS model not loaded"

        with self._lock:
            try:
                wav_io = BytesIO()
                self._voice.synthesize(text, wav_io)
                wav_io.seek(0)
                return wav_io.read()
            except Exception as e:
                logger.error(f"TTS error: {e}")
                return f"TTS error: {str(e)}".encode()


class AudioServer:
    """Combined audio server with STT and TTS."""

    def __init__(self, stt_model: str = "ggml-base.bin", tts_voice: str = "en_US-lessac-medium.onnx"):
        self.stt = WhisperSTT(model_path=stt_model)
        self.tts = PiperTTS(voice_path=tts_voice)

        # Supported TTS voices
        self.voices = [
            {"id": "en_US-lessac-medium", "lang": "en", "name": "English (US)"},
            {"id": "en_US-lessac-low", "lang": "en", "name": "English (US, low)"},
            {"id": "en_GB-alan-medium", "lang": "en", "name": "English (UK)"},
            {"id": "fr_FR-siwis-medium", "lang": "fr", "name": "French"},
            {"id": "de_DE-karlsson-medium", "lang": "de", "name": "German"},
            {"id": "es_ES-sharvard-medium", "lang": "es", "name": "Spanish"},
            {"id": "it_IT-palia-medium", "lang": "it", "name": "Italian"},
        ]

    def load(self) -> Dict[str, bool]:
        """Load both models."""
        stt_ok = self.stt.load()
        tts_ok = self.tts.load()
        return {"stt": stt_ok, "tts": tts_ok}


def handle_request(servers: AudioServer, path: str, method: str, body: bytes) -> tuple:
    """Handle HTTP request."""
    headers = {"Access-Control-Allow-Origin": "*"}

    try:
        # STT endpoints
        if path == "/health" and method == "GET":
            stt_ok = servers.stt.is_available()
            tts_ok = servers.tts.is_available()
            status = "ok" if (stt_ok or tts_ok) else "error"
            response = json.dumps({
                "status": status,
                "stt": {"available": stt_ok, "model": servers.stt.model_path},
                "tts": {"available": tts_ok, "voice": servers.tts.voice_path}
            }).encode()
            return 200, {**headers, "Content-Type": "application/json", "Content-Length": len(response)}, response

        if path == "/v1/models" and method == "GET":
            response = json.dumps({
                "models": [
                    {"name": os.path.basename(servers.stt.model_path)},
                    {"name": os.path.basename(servers.tts.voice_path)}
                ]
            }).encode()
            return 200, {**headers, "Content-Type": "application/json", "Content-Length": len(response)}, response

        if path == "/voices" and method == "GET":
            response = json.dumps({"voices": servers.voices}).encode()
            return 200, {**headers, "Content-Type": "application/json", "Content-Length": len(response)}, response

        # Transcription
        if path == "/v1/audio/transcriptions" and method == "POST":
            try:
                data = json.loads(body)
            except:
                return 400, {**headers, "Content-Type": "application/json"}, b'{"error": "Invalid JSON"}'

            audio_url = data.get('audio_url', '')
            language = data.get('language', 'auto')

            if audio_url:
                try:
                    if audio_url.startswith('data:'):
                        audio_b64 = audio_url.split(',')[1]
                    else:
                        audio_b64 = audio_url
                    audio_data = base64.b64decode(audio_b64)
                except Exception as e:
                    return 400, {**headers, "Content-Type": "application/json"}, json.dumps({"error": f"Bad audio: {e}"}).encode()
            else:
                return 400, {**headers, "Content-Type": "application/json"}, b'{"error": "No audio"}'

            if language != 'auto':
                servers.stt.language = language

            result = servers.stt.transcribe(audio_data)
            response = json.dumps({
                "text": result.get("text", ""),
                "language": result.get("language", "auto")
            }).encode()
            return 200, {**headers, "Content-Type": "application/json", "Content-Length": len(response)}, response

        # TTS
        if path in ("/speak", "/v1/audio/speech") and method == "POST":
            try:
                data = json.loads(body)
            except:
                return 400, {**headers, "Content-Type": "application/json"}, b'{"error": "Invalid JSON"}'

            text = data.get('text', '')
            if not text:
                return 400, {**headers, "Content-Type": "application/json"}, b'{"error": "No text"}'

            audio_data = servers.tts.speak(text)
            if isinstance(audio_data, str):
                return 500, {**headers, "Content-Type": "application/json"}, json.dumps({"error": audio_data}).encode()

            return 200, {**headers, "Content-Type": "audio/wav", "Content-Length": len(audio_data)}, audio_data

        # Not found
        return 404, {**headers, "Content-Type": "application/json"}, b'{"error": "Not found"}'

    except Exception as e:
        logger.error(f"Request error: {e}")
        return 500, {**headers, "Content-Type": "application/json"}, json.dumps({"error": str(e)}).encode()


class WSGIApp:
    """WSGI application wrapper."""

    def __init__(self, servers: AudioServer):
        self.servers = servers

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '/')
        method = environ.get('REQUEST_METHOD', 'GET')
        content_length = int(environ.get('CONTENT_LENGTH', 0))
        body = environ['wsgi.input'].read(content_length) if content_length > 0 else b''

        status, headers, response = handle_request(self.servers, path, method, body)
        start_response(f'{status} OK', list(headers.items()))
        return [response]


def main():
    parser = argparse.ArgumentParser(description='Audio Server for PARC ROBOTICS')
    parser.add_argument('--stt-model', default='ggml-base.bin', help='Whisper model path')
    parser.add_argument('--tts-voice', default='en_US-lessac-medium.onnx', help='Piper voice path')
    parser.add_argument('--stt-port', type=int, default=8084, help='STT server port')
    parser.add_argument('--tts-port', type=int, default=8085, help='TTS server port')
    args = parser.parse_args()

    # Expand paths
    stt_model = os.path.expanduser(args.stt_model)
    tts_voice = os.path.expanduser(args.tts_voice)
    if not tts_voice.endswith('.onnx'):
        tts_voice += '.onnx'

    # Create server
    servers = AudioServer(stt_model=stt_model, tts_voice=tts_voice)

    print("=" * 60)
    print("PARC ROBOTICS - Audio Server")
    print("=" * 60)
    print(f"STT Model: {stt_model}")
    print(f"TTS Voice: {tts_voice}")
    print()

    # Load models
    print("Loading models...")
    results = servers.load()

    if not results['stt'] and not results['tts']:
        print("ERROR: Failed to load any audio models")
        print()
        print("For STT (Whisper.cpp):")
        print("  pip install whispercpp")
        print("  wget https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.bin")
        print()
        print("For TTS (Piper):")
        print("  pip install piper-tts")
        print("  wget https://github.com/rhasspy/piper/raw/master/voices/en_US-lessac-medium.onnx")
        sys.exit(1)

    print(f"STT: {'OK' if results['stt'] else 'NOT LOADED'}")
    print(f"TTS: {'OK' if results['tts'] else 'NOT LOADED'}")
    print()

    # Note: This runs STT on 8084 and TTS on 8085 as separate servers
    # For simplicity, we run them as two separate processes
    # This script can be run twice with different ports

    app = WSGIApp(servers)

    print(f"Server ready on port {args.stt_port} (STT) / {args.tts_port} (TTS)")
    print()
    print("Endpoints:")
    print("  GET  /health               - Health check")
    print("  GET  /v1/models            - List models")
    print("  GET  /voices               - List TTS voices")
    print("  POST /v1/audio/transcriptions - STT (Whisper)")
    print("  POST /speak                - TTS (Piper)")
    print()

    from wsgiref.simple_server import make_server

    # Create server on STT port (TTS would need separate process)
    httpd = make_server('0.0.0.0', args.stt_port, app)
    print(f"STT server listening on http://0.0.0.0:{args.stt_port}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
        httpd.shutdown()


if __name__ == '__main__':
    main()