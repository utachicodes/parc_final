#!/usr/bin/env python3
"""
Piper TTS Server for PARC ROBOTICS
===================================
Multilingual text-to-speech server using Piper ONNX

Usage:
    python3 piper_server.py [--voice en_US-lessac-medium] [--port 8085]

Features:
- Multiple voices across many languages
- Low latency ONNX inference
- WAV output format

Available voices:
- en_US-lessac-medium (English, medium)
- en_US-lessac-low (English, lower quality)
- en_GB-alan-medium (British English)
- fr_FR-siwis-medium (French)
- de_DE-karlsson-medium (German)
- es_ES-sharvard-medium (Spanish)
- it_IT-palia-medium (Italian)
"""

import argparse
import base64
import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# Check for piper
PIPER_AVAILABLE = False
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PiperServer:
    """Piper TTS server."""

    def __init__(self, voice_path: str, sample_rate: int = 22050):
        self.voice_path = voice_path
        self.sample_rate = sample_rate
        self._voice = None
        self._lock = threading.Lock()

    def load_voice(self) -> bool:
        """Load Piper voice."""
        if not os.path.exists(self.voice_path):
            logger.error(f"Voice not found: {self.voice_path}")
            logger.info("Download from: https://github.com/rhasspy/piper/raw/master/voices/")
            return False

        if PIPER_AVAILABLE:
            try:
                self._voice = PiperVoice(self.voice_path)
                logger.info(f"Piper voice loaded: {self.voice_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load voice: {e}")
                return False
        else:
            logger.warning("piper Python package not available")
            logger.info("Install with: pip install piper-tts")
            return False

    def is_available(self) -> bool:
        return self._voice is not None

    def speak(self, text: str) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            WAV audio bytes
        """
        if not self._voice:
            return b"TTS model not loaded"

        with self._lock:
            try:
                import wave
                from io import BytesIO

                # Synthesize
                wav_io = BytesIO()
                self._voice.synthesize(text, wav_io)
                wav_io.seek(0)

                # Get WAV bytes
                return wav_io.read()

            except Exception as e:
                logger.error(f"TTS error: {e}")
                return f"TTS error: {str(e)}".encode()


class PiperHTTPServer:
    """HTTP server wrapper for Piper TTS."""

    def __init__(self, piper: PiperServer, host: str = "0.0.0.0", port: int = 8085):
        self.piper = piper
        self.host = host
        self.port = port
        self._server = None

        # Available voices
        self.voices = [
            {"id": "en_US-lessac-medium", "lang": "en", "name": "English (US, medium)"},
            {"id": "en_US-lessac-low", "lang": "en", "name": "English (US, low)"},
            {"id": "en_GB-alan-medium", "lang": "en", "name": "English (UK)"},
            {"id": "fr_FR-siwis-medium", "lang": "fr", "name": "French"},
            {"id": "de_DE-karlsson-medium", "lang": "de", "name": "German"},
            {"id": "es_ES-sharvard-medium", "lang": "es", "name": "Spanish"},
            {"id": "it_IT-palia-medium", "lang": "it", "name": "Italian"},
        ]

    def handle_health(self, environ: Dict[str, Any]) -> tuple:
        """Health check endpoint."""
        status = "ok" if self.piper.is_available() else "error"
        body = json.dumps({"status": status, "voice": self.piper.voice_path}).encode()
        return 200, {"Content-Type": "application/json", "Content-Length": len(body)}, body

    def handle_voices(self, environ: Dict[str, Any]) -> tuple:
        """List available voices."""
        body = json.dumps({"voices": self.voices}).encode()
        return 200, {"Content-Type": "application/json", "Content-Length": len(body)}, body

    def handle_speak(self, environ: Dict[str, Any], request_body: bytes) -> tuple:
        """Handle TTS request."""
        try:
            data = json.loads(request_body.decode())
        except json.JSONDecodeError:
            return 400, {}, b'{"error": "Invalid JSON"}'

        text = data.get('text', '')
        voice = data.get('voice', os.path.basename(self.piper.voice_path).replace('.onnx', ''))

        if not text:
            return 400, {}, b'{"error": "No text provided"}'

        # Load requested voice if different
        if voice and voice != os.path.basename(self.piper.voice_path).replace('.onnx', ''):
            voice_path = self._get_voice_path(voice)
            if voice_path and os.path.exists(voice_path):
                self.piper.voice_path = voice_path
                self.piper.load_voice()

        audio_data = self.piper.speak(text)

        if isinstance(audio_data, bytes) and audio_data.startswith(b'TTS error'):
            return 500, {}, json.dumps({"error": audio_data.decode()}).encode()

        return 200, {"Content-Type": "audio/wav", "Content-Length": len(audio_data)}, audio_data

    def _get_voice_path(self, voice_id: str) -> str:
        """Get path for voice ID."""
        voice_dir = os.path.dirname(self.piper.voice_path) or "~/models/voices"
        voice_dir = os.path.expanduser(voice_dir)
        return os.path.join(voice_dir, f"{voice_id}.onnx")

    def handle(self, environ: Dict[str, Any], start_response) -> None:
        """Main HTTP handler."""
        path = environ.get('PATH_INFO', '/')
        method = environ.get('REQUEST_METHOD', 'GET')

        try:
            if path == '/health' and method == 'GET':
                status, headers, body = self.handle_health(environ)
            elif path == '/voices' and method == 'GET':
                status, headers, body = self.handle_voices(environ)
            elif path == '/speak' and method == 'POST':
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(content_length)
                status, headers, body = self.handle_speak(environ, request_body)
            elif path == '/v1/audio/speech' and method == 'POST':
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(content_length)
                status, headers, body = self.handle_speak(environ, request_body)
            else:
                status, headers, body = 404, {}, b'{"error": "Not found"}'

            headers['Access-Control-Allow-Origin'] = '*'
            start_response(f'{status} OK', list(headers.items()))
            yield body

        except Exception as e:
            logger.error(f"Handler error: {e}")
            start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
            yield json.dumps({"error": str(e)}).encode()


def create_app(piper: PiperServer, host: str = "0.0.0.0", port: int = 8085):
    """Create WSGI application."""
    server = PiperHTTPServer(piper, host, port)

    def app(environ, start_response):
        return server.handle(environ, start_response)

    return app


def main():
    parser = argparse.ArgumentParser(description='Piper TTS Server for PARC ROBOTICS')
    parser.add_argument('--voice', default='/home/parc/models/voices/en_US-lessac-medium.onnx', help='Piper voice ONNX file')
    parser.add_argument('--port', type=int, default=8085, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    # Expand path
    voice_path = os.path.expanduser(args.voice)
    if not voice_path.endswith('.onnx'):
        voice_path += '.onnx'

    # Create Piper server
    piper = PiperServer(voice_path=voice_path)

    # Load voice
    logger.info(f"Loading Piper voice: {voice_path}")
    if not piper.load_voice():
        logger.error("Failed to load Piper voice")
        logger.info("Download voices from: https://github.com/rhasspy/piper/raw/master/voices/")
        logger.info("Example:")
        logger.info("  wget https://github.com/rhasspy/piper/raw/master/voices/en_US-lessac-medium.onnx")
        logger.info("  wget https://github.com/rhasspy/piper/raw/master/voices/en_US-lessac-medium.onnx.json")
        sys.exit(1)

    # Create app
    app = create_app(piper, args.host, args.port)

    logger.info(f"Starting Piper TTS Server on {args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info("  GET  /health      - Server health")
    logger.info("  GET  /voices       - List available voices")
    logger.info("  POST /speak        - Synthesize speech ({\"text\": \"hello\"})")
    logger.info("  POST /v1/audio/speech - OpenAI-compatible TTS endpoint")

    # Run server
    from wsgiref.simple_server import make_server
    httpd = make_server(args.host, args.port, app)
    logger.info("Piper TTS server ready")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        httpd.shutdown()


if __name__ == '__main__':
    main()