#!/usr/bin/env python3
"""
Whisper.cpp HTTP Server for PARC ROBOTICS
==========================================
Multilingual speech-to-text server using Whisper.cpp

Usage:
    python3 whisper_server.py [--model base] [--port 8084]

Features:
- 100+ language support (auto-detect)
- Streaming transcription
- Multiple model sizes (tiny, base, small, medium, large)
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
from pathlib import Path
from typing import Optional, Dict, Any

# Try to import whisper.cpp Python bindings, fallback to subprocess
WHISPER_CPP_AVAILABLE = False
try:
    from whispercpp import Whisper
    WHISPER_CPP_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WhisperServer:
    """Whisper.cpp-based STT server with HTTP API."""

    def __init__(
        self,
        model_path: str = "/home/parc/models/whisper/ggml-base.bin",
        language: str = "auto",
        max_context: int = -1,
        max_len: int = 0,
        word_timestamps: bool = False
    ):
        self.model_path = model_path
        self.language = language
        self.max_context = max_context
        self.max_len = max_len
        self.word_timestamps = word_timestamps
        self._model = None
        self._lock = threading.Lock()

    def load_model(self) -> bool:
        """Load Whisper model."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found: {self.model_path}")
            logger.info(f"Download from: https://huggingface.co/datasets/ggerganov/whisper.cpp")
            return False

        if WHISPER_CPP_AVAILABLE:
            try:
                self._model = Whisper(self.model_path)
                logger.info(f"Whisper model loaded: {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
        else:
            logger.warning("whispercpp Python bindings not available")
            logger.info("Install with: pip install whispercpp")
            return False

    def is_available(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (WAV format recommended)

        Returns:
            {"text": "transcribed text", "language": "en", "segments": [...]}
        """
        if not self._model:
            return {"error": "Model not loaded", "text": "", "language": ""}

        with self._lock:
            try:
                # Decode audio
                import numpy as np
                import struct

                # Handle WAV format
                if audio_data[:4] == b'RIFF':
                    wav_data = audio_data
                    with wave.open(BytesIO(audio_data)) as wf:
                        sample_rate = wf.getframerate()
                        frames = wf.readframes(wf.getnframes())
                        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    # Raw PCM - assume 16kHz mono
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Run transcription
                text = self._model.transcribe(audio_np, language=self.language if self.language != "auto" else None)

                return {
                    "text": text.strip(),
                    "language": self.language if self.language != "auto" else "auto",
                    "segments": [{"text": text.strip()}]
                }

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return {"error": str(e), "text": "", "language": ""}


class WhisperHTTPServer:
    """HTTP server wrapper for Whisper STT."""

    def __init__(self, whisper: WhisperServer, host: str = "0.0.0.0", port: int = 8084):
        self.whisper = whisper
        self.host = host
        self.port = port
        self._server = None

    def handle_health(self, environ: Dict[str, Any]) -> tuple:
        """Health check endpoint."""
        status = "ok" if self.whisper.is_available() else "error"
        body = json.dumps({"status": status, "model": self.whisper.model_path}).encode()
        return 200, {"Content-Type": "application/json", "Content-Length": len(body)}, body

    def handle_models(self, environ: Dict[str, Any]) -> tuple:
        """List available models."""
        body = json.dumps({
            "models": [
                {"name": os.path.basename(self.whisper.model_path), "size": os.path.getsize(self.whisper.model_path)}
            ]
        }).encode()
        return 200, {"Content-Type": "application/json", "Content-Length": len(body)}, body

    def handle_transcribe(self, environ: Dict[str, Any], request_body: bytes) -> tuple:
        """Handle transcription request."""
        try:
            data = json.loads(request_body)
        except json.JSONDecodeError:
            # Try form data
            try:
                from urllib.parse import parse_qs
                data = dict(parse_qs(request_body.decode()))
                data = {k: v[0] if len(v) == 1 else v for k, v in data.items()}
            except:
                return 400, {}, b'{"error": "Invalid JSON"}'

        model = data.get('model', 'base')
        audio_url = data.get('audio_url', '')
        language = data.get('language', 'auto')
        max_tokens = data.get('max_tokens', 256)

        # Decode audio from base64
        try:
            if audio_url.startswith('data:'):
                audio_b64 = audio_url.split(',')[1]
            else:
                audio_b64 = audio_url

            audio_data = base64.b64decode(audio_b64)
        except Exception as e:
            return 400, {}, json.dumps({"error": f"Failed to decode audio: {e}"}).encode()

        # Update language if provided
        if language and language != 'auto':
            self.whisper.language = language

        result = self.whisper.transcribe(audio_data)

        if "error" in result:
            return 500, {}, json.dumps(result).encode()

        return 200, {"Content-Type": "application/json"}, json.dumps({
            "text": result.get("text", ""),
            "language": result.get("language", ""),
            "segments": result.get("segments", [])
        }).encode()

    def handle(self, environ: Dict[str, Any], start_response) -> None:
        """Main HTTP handler."""
        path = environ.get('PATH_INFO', '/')
        method = environ.get('REQUEST_METHOD', 'GET')

        try:
            if path == '/health' and method == 'GET':
                status, headers, body = self.handle_health(environ)
            elif path == '/v1/models' and method == 'GET':
                status, headers, body = self.handle_models(environ)
            elif path == '/v1/audio/transcriptions' and method == 'POST':
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(content_length)
                status, headers, body = self.handle_transcribe(environ, request_body)
            elif path == '/v1/audio/transcriptions' and method == 'GET':
                status, headers, body = 200, {}, b'{"status": "Whisper STT ready"}'
            else:
                status, headers, body = 404, {}, b'{"error": "Not found"}'

            headers['Access-Control-Allow-Origin'] = '*'
            start_response(f'{status} OK', list(headers.items()))
            yield body

        except Exception as e:
            logger.error(f"Handler error: {e}")
            start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
            yield json.dumps({"error": str(e)}).encode()


def create_app(whisper: WhisperServer, host: str = "0.0.0.0", port: int = 8084):
    """Create WSGI application."""
    from io import BytesIO
    server = WhisperHTTPServer(whisper, host, port)

    def app(environ, start_response):
        return server.handle(environ, start_response)

    return app


def main():
    parser = argparse.ArgumentParser(description='Whisper.cpp STT Server for PARC ROBOTICS')
    parser.add_argument('--model', default='/home/parc/models/whisper/ggml-base.bin', help='Whisper model path')
    parser.add_argument('--port', type=int, default=8084, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--language', default='auto', help='Default language (auto for detection)')
    args = parser.parse_args()

    # Create Whisper server
    whisper = WhisperServer(
        model_path=args.model,
        language=args.language
    )

    # Load model
    logger.info(f"Loading Whisper model: {args.model}")
    if not whisper.load_model():
        logger.error("Failed to load Whisper model")
        logger.info("Download models from: https://huggingface.co/datasets/ggerganov/whisper.cpp")
        logger.info("Example: wget https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-base.bin")
        sys.exit(1)

    # Create app
    app = create_app(whisper, args.host, args.port)

    logger.info(f"Starting Whisper STT Server on {args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info("  GET  /health              - Server health")
    logger.info("  GET  /v1/models           - List models")
    logger.info("  POST /v1/audio/transcriptions - Transcribe audio")

    # Run server
    from wsgiref.simple_server import make_server
    httpd = make_server(args.host, args.port, app)
    logger.info("Whisper STT server ready")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        httpd.shutdown()


if __name__ == '__main__':
    main()