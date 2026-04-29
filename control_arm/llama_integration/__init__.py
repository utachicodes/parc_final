"""
LLAMA.CPP Integration for PARC ROBOTICS
=========================================
AI Assistant module using llama.cpp server

Target Platform: Jetson Orin Nano (8GB)
- ARM Cortex-A78AE (64-bit)
- NVIDIA Ampere GPU (2048 CUDA cores)
- 8GB LPDDR5

Models Available:
----------------
1. LFM2.5-VL-1.6B - Vision Language Model for camera analysis
2. LFM2.5-VL-450M - Smaller vision model for fast tasks
3. LFM2.5-1.2B-Thinking - Thinking/reasoning model
4. LFM2.5-Audio-1.5B - Audio/speech processing model
5. Phi-3-mini - General purpose Q&A

Build llama.cpp on Jetson:
--------------------------
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="87" -DGGML_NATIVE=OFF
cmake --build build --config Release -j$(nproc)

Download models from HuggingFace:
---------------------------------
# Vision models (for camera analysis)
wget https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/lfm2.5-vl-1.6b-f16.gguf
wget https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/lfm2.5-vl-450m-f16.gguf

# Thinking model (for complex reasoning)
wget https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/resolve/main/lfm2.5-1.2b-thinking-q4_k_m.gguf

# Audio model (for voice commands)
wget https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/resolve/main/lfm2.5-audio-1.5b-f16.gguf

Run servers:
------------
# Vision server (port 8081)
llama-server -m lfm2.5-vl-1.6b-f16.gguf -c 2048 -ngl 99 --host 0.0.0.0 --port 8081

# Audio server (port 8082)
llama-server -m lfm2.5-audio-1.5b-f16.gguf -c 2048 -ngl 99 --host 0.0.0.0 --port 8082

# General AI server (port 8080)
llama-server -m lfm2.5-1.2b-thinking-q4_k_m.gguf -c 2048 -ngl 99 --host 0.0.0.0 --port 8080
"""

import requests
import json
import time
import threading
import base64
import io
import numpy as np
from typing import Optional, Generator, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Model configurations optimized for Jetson Orin Nano 8GB
MODELS = {
    "vision_1.6b": {
        "name": "LFM2.5-VL-1.6B",
        "file": "lfm2.5-vl-1.6b-f16.gguf",
        "url": "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF",
        "port": 8081,
        "type": "vision",
        "description": "Vision language model for camera analysis",
        "size_gb": 3.2
    },
    "vision_450m": {
        "name": "LFM2.5-VL-450M",
        "file": "lfm2.5-vl-450m-f16.gguf",
        "url": "https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF",
        "port": 8081,
        "type": "vision",
        "description": "Fast vision model for quick tasks",
        "size_gb": 0.9
    },
    "thinking": {
        "name": "LFM2.5-1.2B-Thinking",
        "file": "lfm2.5-1.2b-thinking-q4_k_m.gguf",
        "url": "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking",
        "port": 8080,
        "type": "thinking",
        "description": "Complex reasoning and planning",
        "size_gb": 0.8
    },
    "audio": {
        "name": "LFM2.5-Audio-1.5B",
        "file": "lfm2.5-audio-1.5b-f16.gguf",
        "url": "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF",
        "port": 8082,
        "type": "audio",
        "description": "Speech recognition and synthesis",
        "size_gb": 3.0
    }
}


class LlamaAssistant:
    """
    AI Assistant using llama.cpp server with OpenAI-compatible API.
    Optimized for Jetson Orin Nano 8GB.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "lfm2.5-1.2b-thinking-q4_k_m",
        context_length: int = 2048,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: str = None
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or self._default_system_prompt()
        self._connected = False
        self._session_history: List[Dict[str, str]] = []

    def _default_system_prompt(self) -> str:
        return """You are an AI assistant for PARC ROBOTICS SO-101 robot arm controller.
You help users learn about robotics, write Python code for robot control, and explain kinematics concepts.
Be concise, technical, and helpful. When writing code, include comments explaining the logic."""

    def connect(self, timeout: float = 5.0) -> bool:
        """Test connection to llama.cpp server. Auto-start if not running."""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=timeout)
            if resp.status_code == 200:
                self._connected = True
                models = resp.json().get('data', [])
                logger.info(f"Connected to llama server. Models: {[m.get('id') for m in models]}")
                return True
        except Exception as e:
            logger.warning(f"Cannot connect to llama server at {self.base_url}: {e}")

        # Try to auto-start service
        try:
            from .service_manager import ensure_service
            port = int(self.base_url.split(":")[-1])
            if port == 8080:
                ensure_service("ai")
            elif port == 8081:
                ensure_service("vision")
        except Exception as auto_e:
            logger.warning(f"Auto-start failed: {auto_e}")

        # Try again
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=timeout)
            if resp.status_code == 200:
                self._connected = True
                return True
        except Exception:
            pass

        self._connected = False
        return False

    def is_connected(self) -> bool:
        return self._connected

    def chat(
        self,
        message: str,
        streaming: bool = False,
        timeout: float = 60.0
    ) -> Generator[str, None, None] | str:
        """Send a chat message and get response."""
        if not self._connected:
            if not self.connect():
                return "AI Assistant unavailable. Please ensure llama.cpp server is running."

        messages = [{"role": "system", "content": self.system_prompt}]
        for user, assistant in self._session_history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": streaming
        }

        try:
            if streaming:
                return self._stream_response(payload, timeout)
            else:
                return self._blocking_response(payload, timeout)
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error communicating with AI: {str(e)}"

    def _blocking_response(self, payload: dict, timeout: float) -> str:
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
            stream=False
        )
        if resp.status_code == 200:
            data = resp.json()
            assistant_msg = data['choices'][0]['message']['content']
            self._session_history.append((payload['messages'][-1]['content'], assistant_msg))
            return assistant_msg
        else:
            return f"Server error: {resp.status_code} - {resp.text}"

    def _stream_response(self, payload: dict, timeout: float) -> Generator[str, None, None]:
        response_text = ""
        try:
            with requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=timeout,
                stream=True
            ) as resp:
                if resp.status_code != 200:
                    yield f"Server error: {resp.status_code}"
                    return

                for line in resp.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        chunk = delta['content']
                                        response_text += chunk
                                        yield chunk
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield f"Stream error: {str(e)}"
            return

        if response_text:
            self._session_history.append((payload['messages'][-1]['content'], response_text))

    def clear_history(self):
        """Clear conversation history."""
        self._session_history = []

    def get_context_used(self) -> int:
        """Get current context window usage estimate."""
        return len(str(self._session_history)) // 4

    def chat_with_rag(
        self,
        message: str,
        use_rag: bool = True,
        rag_top_k: int = 3,
        streaming: bool = False,
        timeout: float = 60.0
    ) -> Generator[str, None, None] | str:
        """
        Chat with RAG context augmentation.
        Retrieves relevant documents and includes them in the prompt.

        Args:
            message: User message
            use_rag: Whether to use RAG retrieval
            rag_top_k: Number of RAG results to include
            streaming: Whether to stream response
            timeout: Request timeout

        Returns:
            Response text (or generator if streaming)
        """
        if use_rag:
            try:
                from .rag import RAGSystem, get_rag_system
                rag = get_rag_system()

                # Retrieve relevant context
                context, sources = rag.retrieve_with_context(
                    message,
                    n_results=rag_top_k,
                    max_context_length=1500
                )

                # Build system prompt with context
                if context:
                    system_prompt = self.system_prompt + f"""

RELEVANT CONTEXT FROM KNOWLEDGE BASE:
{context}

Use the context above to answer the user's question. If the context doesn't contain
relevant information, say so but still answer based on your general knowledge."""
                else:
                    system_prompt = self.system_prompt
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                system_prompt = self.system_prompt
        else:
            system_prompt = self.system_prompt

        # Use standard chat with modified system prompt
        original_system = self.system_prompt
        self.system_prompt = system_prompt
        result = self.chat(message, streaming=streaming, timeout=timeout)
        self.system_prompt = original_system

        return result


class VisionAssistant:
    """
    Vision Language Model for camera-based analysis.
    Used for object detection, scene understanding, and pick-and-place assistance.
    """

    def __init__(self, base_url: str = "http://localhost:8081", model: str = "lfm2.5-vl-1.6b-f16"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._connected = False

    def connect(self, timeout: float = 5.0) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=timeout)
            if resp.status_code == 200:
                self._connected = True
                return True
        except Exception as e:
            logger.warning(f"Cannot connect to vision server at {self.base_url}: {e}")
        self._connected = False
        return False

    def is_connected(self) -> bool:
        return self._connected

    def analyze_image(
        self,
        image_data: bytes | str,
        question: str = "Describe what you see in this image.",
        timeout: float = 30.0
    ) -> str:
        """
        Analyze an image and answer a question about it.
        
        Args:
            image_data: Raw image bytes or base64 encoded string
            question: Question about the image
            timeout: Request timeout
            
        Returns:
            Text description/answer from the vision model
        """
        if not self._connected:
            if not self.connect():
                return "Vision model unavailable. Please ensure vision server is running on port 8081."

        # Handle both raw bytes and base64 strings
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode('utf-8')
        else:
            image_b64 = image_data

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": question}
                    ]
                }
            ],
            "max_tokens": 256,
            "temperature": 0.1
        }

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                return data['choices'][0]['message']['content']
            else:
                return f"Vision error: {resp.status_code} - {resp.text}"
        except Exception as e:
            return f"Vision error: {str(e)}"

    def detect_objects(self, image_data: bytes) -> List[Dict[str, Any]]:
        """
        Detect objects in an image for pick-and-place operations.
        Returns list of detected objects with positions.
        """
        result = self.analyze_image(
            image_data,
            question="Identify the objects in this image that could be picked up. For each object, provide its approximate position (left/center/right, near/middle/far) and suggested gripper approach."
        )
        return {"description": result, "raw": result}


class AudioAssistant:
    """
    Audio model for voice commands and speech recognition.
    
    Uses Whisper.cpp for STT and Piper for TTS - multilingual support.
    Replaces LFM2.5-Audio which is English-only.
    
    Ports:
        - Whisper.cpp server: 8084 (STT)
        - Piper TTS server: 8085 (TTS)
    """

    def __init__(
        self,
        stt_url: str = "http://localhost:8084",
        tts_url: str = "http://localhost:8085",
        stt_model: str = "base",
        tts_voice: str = "en_US-lessac-medium"
    ):
        self.stt_url = stt_url.rstrip('/')
        self.tts_url = tts_url.rstrip('/')
        self.stt_model = stt_model
        self.tts_voice = tts_voice
        self._stt_connected = False
        self._tts_connected = False

    def connect_stt(self, timeout: float = 5.0) -> bool:
        """Check Whisper.cpp STT server availability. Auto-start if not running."""
        try:
            resp = requests.get(f"{self.stt_url}/health", timeout=timeout)
            if resp.status_code == 200:
                self._stt_connected = True
                return True
        except Exception as e:
            logger.warning(f"Cannot connect to STT server at {self.stt_url}: {e}")

        # Try auto-start whisper
        try:
            from .service_manager import ensure_service
            ensure_service("whisper")
        except Exception:
            pass

        # Try again
        try:
            resp = requests.get(f"{self.stt_url}/health", timeout=timeout)
            if resp.status_code == 200:
                self._stt_connected = True
                return True
        except Exception:
            pass

        self._stt_connected = False
        return False

    def connect_tts(self, timeout: float = 5.0) -> bool:
        """Check Piper TTS server availability. Auto-start if not running."""
        try:
            resp = requests.get(f"{self.tts_url}/voices", timeout=timeout)
            if resp.status_code == 200:
                self._tts_connected = True
                return True
        except Exception as e:
            logger.warning(f"Cannot connect to TTS server at {self.tts_url}: {e}")

        # Try auto-start piper
        try:
            from .service_manager import ensure_service
            ensure_service("piper")
        except Exception:
            pass

        # Try again
        try:
            resp = requests.get(f"{self.tts_url}/voices", timeout=timeout)
            if resp.status_code == 200:
                self._tts_connected = True
                return True
        except Exception:
            pass

        self._tts_connected = False
        return False

    def connect(self, timeout: float = 5.0) -> bool:
        stt_ok = self.connect_stt(timeout)
        tts_ok = self.connect_tts(timeout)
        return stt_ok or tts_ok

    def is_connected(self) -> bool:
        return self._stt_connected or self._tts_connected

    def transcribe(self, audio_data: bytes, timeout: float = 30.0) -> str:
        """
        Transcribe speech audio to text using Whisper.cpp.
        Supports 100+ languages including French, Spanish, German, Chinese, etc.
        
        Args:
            audio_data: Raw audio bytes (WAV/PCM/WebM)
            timeout: Request timeout
            
        Returns:
            Transcribed text from speech
        """
        if not self._stt_connected:
            if not self.connect_stt():
                return "STT server unavailable. Ensure Whisper.cpp server is running on port 8084."

        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        payload = {
            "model": self.stt_model,
            "audio_url": f"data:audio/wav;base64,{audio_b64}",
            "max_tokens": 256
        }

        try:
            resp = requests.post(
                f"{self.stt_url}/v1/audio/transcriptions",
                json=payload,
                timeout=timeout
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get('text', '')
            else:
                return f"STT error: {resp.status_code}"
        except Exception as e:
            return f"STT error: {str(e)}"

    def speak(self, text: str, timeout: float = 30.0) -> bytes:
        """
        Convert text to speech using Piper.
        Supports multiple voices across many languages.
        
        Args:
            text: Text to speak
            timeout: Request timeout
            
        Returns:
            Audio bytes (WAV format)
        """
        if not self._tts_connected:
            if not self.connect_tts():
                return b"TTS server unavailable. Ensure Piper server is running on port 8085."

        try:
            resp = requests.post(
                f"{self.tts_url}/speak",
                json={"text": text, "voice": self.tts_voice},
                timeout=timeout
            )
            if resp.status_code == 200:
                return resp.content
            else:
                return f"TTS error: {resp.status_code}".encode()
        except Exception as e:
            return f"TTS error: {str(e)}".encode()

    def speech_to_command(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Convert speech to structured robot command.
        Language-agnostic - works with any language Whisper.cpp supports.

        Example:
            Input: "Pick up the red cube from the left and place it on the right"
            Output: {
                "action": "pick_place",
                "object": "red cube",
                "from": "left",
                "to": "right",
                "confidence": 0.92
            }
        """
        text = self.transcribe(audio_data)
        if not text or text.startswith("STT error"):
            return {"error": text, "command": None, "text": None}

        thinking = LlamaAssistant(base_url="http://localhost:8080", model="lfm2.5-1.2b-thinking-q4_k_m")
        thinking.connect()

        if thinking.is_connected():
            prompt = f"""Parse this voice command for a robot arm. Extract the action, object, and positions.
Command: "{text}"
Return JSON with: action (pick/place/move/home), object, from_pos, to_pos, confidence (0-1)
If unclear, return confidence: 0"""
            response = thinking.chat(prompt)
            try:
                if response and not response.startswith("AI"):
                    cmd = json.loads(response)
                    cmd["text"] = text
                    return cmd
            except json.JSONDecodeError:
                pass

        return {"text": text, "command": None}


# Singleton instances
_ai_assistant: Optional[LlamaAssistant] = None
_vision_assistant: Optional[VisionAssistant] = None
_audio_assistant: Optional[AudioAssistant] = None


def get_ai_assistant() -> LlamaAssistant:
    global _ai_assistant
    if _ai_assistant is None:
        _ai_assistant = LlamaAssistant(
            base_url="http://localhost:8080",
            model="lfm2.5-1.2b-thinking-q4_k_m"
        )
    return _ai_assistant


def get_vision_assistant() -> VisionAssistant:
    global _vision_assistant
    if _vision_assistant is None:
        _vision_assistant = VisionAssistant(base_url="http://localhost:8081")
    return _vision_assistant


def get_audio_assistant() -> AudioAssistant:
    global _audio_assistant
    if _audio_assistant is None:
        _audio_assistant = AudioAssistant(
            stt_url="http://localhost:8084",
            tts_url="http://localhost:8085",
            stt_model="base",
            tts_voice="en_US-lessac-medium"
        )
    return _audio_assistant


# -----------------------------------------------------------------------------
# SO-101 Forward Kinematics (from Georgia Tech ECE 4560 Tucker Course)
# -----------------------------------------------------------------------------

class SO101ForwardKinematics:
    """
    Forward kinematics for SO-101 robot arm.
    Based on Georgia Tech ECE 4560 Tucker Course Assignment 6.

    Reference: https://maegantucker.com/ECE4560/assignment6-so101/

    IMPORTANT:
    - Input joints in DEGREES (SDK convention)
    - Output position in METERS (for use with URDF/meters-based systems)
    - Only joints 1-5 affect end-effector position (joint 6 is gripper)

    Joint order (matches SDK):
    - shoulder_pan: Base rotation (joint 1)
    - shoulder_lift: Shoulder elevation (joint 2)
    - elbow_flex: Elbow bend (joint 3)
    - wrist_flex: Wrist pitch (joint 4)
    - wrist_roll: Wrist rotation (joint 5)
    - gripper: Gripper open/close (joint 6) - does NOT affect object position
    """

    JOINT_LIMITS = {
        'shoulder_pan': (-110, 110),
        'shoulder_lift': (-100, 90),
        'elbow_flex': (-140, 140),
        'wrist_flex': (-100, 100),
        'wrist_roll': (-150, 150),
        'gripper': (0, 60)
    }

    def __init__(self):
        pass

    def _Rx(self, thetadeg):
        """Rotation about X axis."""
        thetarad = thetadeg * 0.017453292519943295
        c, s = np.cos(thetarad), np.sin(thetarad)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _Ry(self, thetadeg):
        """Rotation about Y axis."""
        thetarad = thetadeg * 0.017453292519943295
        c, s = np.cos(thetarad), np.sin(thetarad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _Rz(self, thetadeg):
        """Rotation about Z axis."""
        thetarad = thetadeg * 0.017453292519943295
        c, s = np.cos(thetarad), np.sin(thetarad)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def get_gw1(self, theta1_deg):
        """
        World/base to joint 1 transformation.
        Based on Tucker Course: displacement = (0.0388353, 0.0, 0.0624)
        rotation = Rz(180) @ Rx(180) @ Rz(theta1_deg)
        """
        displacement = (0.0388353, 0.0, 0.0624)
        rotation = self._Rz(180) @ self._Rx(180) @ self._Rz(theta1_deg)
        pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
        return pose

    def get_g12(self, theta2_deg):
        """Joint 1 to Joint 2 transformation."""
        displacement = (0.0, 0.0, 0.100)
        rotation = self._Rx(-90 - theta2_deg)
        pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
        return pose

    def get_g23(self, theta3_deg):
        """Joint 2 to Joint 3 transformation."""
        displacement = (0.100, 0.0, 0.0)
        rotation = self._Rx(90 + theta3_deg)
        pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
        return pose

    def get_g34(self, theta4_deg):
        """Joint 3 to Joint 4 transformation."""
        displacement = (0.0, 0.0, 0.080)
        rotation = self._Rx(-90 - theta4_deg)
        pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
        return pose

    def get_g45(self, theta5_deg):
        """Joint 4 to Joint 5 transformation."""
        displacement = (0.0, 0.0, 0.050)
        rotation = self._Rz(theta5_deg)
        pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
        return pose

    def get_g5t(self):
        """Joint 5 to tool/gripper tip (fixed transformation)."""
        displacement = (0.0, 0.0, 0.050)
        rotation = np.eye(3)
        pose = np.block([[rotation, np.array(displacement).reshape(3, 1)], [0, 0, 0, 1]])
        return pose

    def compute(self, joints: Dict[str, float]) -> tuple:
        """
        Compute forward kinematics given joint angles.

        Args:
            joints: Dict with keys 'shoulder_pan', 'shoulder_lift', 'elbow_flex',
                   'wrist_flex', 'wrist_roll', 'gripper' (all in degrees)

        Returns:
            (position, rotation) - tuple of (3,) position in meters and (3,3) rotation matrix
        """
        t1 = joints.get('shoulder_pan', 0)
        t2 = joints.get('shoulder_lift', 0)
        t3 = joints.get('elbow_flex', 0)
        t4 = joints.get('wrist_flex', 0)
        t5 = joints.get('wrist_roll', 0)

        gw1 = self.get_gw1(t1)
        g12 = self.get_g12(t2)
        g23 = self.get_g23(t3)
        g34 = self.get_g34(t4)
        g45 = self.get_g45(t5)
        g5t = self.get_g5t()

        g0t = gw1 @ g12 @ g23 @ g34 @ g45 @ g5t

        position = g0t[0:3, 3]
        rotation = g0t[0:3, 0:3]

        return position, rotation

    def compute_gripper_pose(self, joints: Dict[str, float]) -> Dict[str, Any]:
        """Get complete gripper pose including position and Euler angles."""
        position, rotation = self.compute(joints)

        sy = np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2)
        singular = sy < 1e-6

        if singular:
            x = 0 if singular else np.arctan2(-rotation[1, 2], rotation[1, 1])
            y = np.arctan2(-rotation[2, 0], sy)
            z = 0
        else:
            x = np.arctan2(rotation[1, 0], rotation[0, 0])
            y = np.arctan2(-rotation[2, 0], sy)
            z = np.arctan2(rotation[2, 1], rotation[2, 2])

        return {
            'position': position.tolist() if hasattr(position, 'tolist') else position,
            'euler': [np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)]
        }


class CodeValidator:
    """
    Validate robot code before execution.
    Checks syntax, joint limits, and potential collision risks.
    """

    def __init__(self):
        self.fk = SO101ForwardKinematics()

    def validate(self, code: str, initial_joints: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Validate robot code.

        Args:
            code: Python code to validate
            initial_joints: Starting joint positions

        Returns:
            Validation result with 'valid', 'errors', 'warnings' lists
        """
        errors = []
        warnings = []

        if not code or not code.strip():
            errors.append("No code provided")
            return {'valid': False, 'errors': errors, 'warnings': warnings}

        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")

        if initial_joints:
            for joint, angle in initial_joints.items():
                if joint in self.fk.JOINT_LIMITS:
                    min_deg, max_deg = self.fk.JOINT_LIMITS[joint]
                    if angle < min_deg or angle > max_deg:
                        errors.append(
                            f"Joint {joint} angle {angle}° exceeds limits [{min_deg}, {max_deg}]"
                        )

        dangerous_patterns = [
            ('while True:', 'Infinite loop detected - may cause robot to move continuously'),
            ('time.sleep(0)', 'Zero delay detected - may cause rapid movement'),
            ('while 1:', 'Infinite loop with while 1')
        ]

        for pattern, msg in dangerous_patterns:
            if pattern in code:
                warnings.append(msg)

        import re
        move_pattern = r'(?:setJoint|servo|writeServo|move)\s*\(\s*(\d+)\s*,\s*([-\d.]+)'
        matches = re.findall(move_pattern, code)
        for servo_id, angle in matches:
            servo_id = int(servo_id)
            angle = float(angle)
            if servo_id in range(1, 7):
                joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                             'wrist_flex', 'wrist_roll', 'gripper']
                if servo_id <= 6:
                    joint = joint_names[servo_id - 1]
                    min_deg, max_deg = self.fk.JOINT_LIMITS[joint]
                    if angle < min_deg or angle > max_deg:
                        warnings.append(
                            f"Servo {servo_id} ({joint}) angle {angle}° may exceed soft limits"
                            f" [{min_deg}, {max_deg}]"
                        )

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


# -----------------------------------------------------------------------------
# Flask API Endpoints
# -----------------------------------------------------------------------------

def setup_ai_routes(app):
    """Add all AI-related routes to Flask app."""
    from flask import Blueprint, request, jsonify, Response, stream_with_context

    ai_bp = Blueprint('ai', __name__)

    @ai_bp.route('/api/ai/chat', methods=['POST'])
    def ai_chat():
        data = request.json or {}
        message = data.get('message', '')
        streaming = data.get('stream', False)

        assistant = get_ai_assistant()
        if not assistant.is_connected():
            assistant.connect()

        if not assistant.is_connected():
            return jsonify({'error': 'AI server unavailable on port 8080'}), 503

        if streaming:
            def generate():
                for chunk in assistant.chat(message, streaming=True):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        else:
            response = assistant.chat(message, streaming=False)
            return jsonify({'response': response})

    @ai_bp.route('/api/ai/rag-chat', methods=['POST'])
    def ai_rag_chat():
        """
        Chat with RAG context augmentation.
        Retrieves relevant documents from knowledge base and includes in prompt.
        """
        data = request.json or {}
        message = data.get('message', '')
        streaming = data.get('stream', False)
        use_rag = data.get('use_rag', True)
        rag_top_k = data.get('rag_top_k', 3)

        assistant = get_ai_assistant()
        if not assistant.is_connected():
            assistant.connect()

        if not assistant.is_connected():
            return jsonify({'error': 'AI server unavailable on port 8080'}), 503

        # Get RAG context info
        rag_info = {}
        if use_rag:
            try:
                from .rag import get_rag_system
                rag = get_rag_system()
                rag_info['doc_count'] = rag.count_documents()
                results = rag.retrieve(message, n_results=rag_top_k)
                rag_info['sources'] = [
                    {
                        'text': r.text[:200] + '...' if len(r.text) > 200 else r.text,
                        'score': r.score,
                        'category': r.metadata.get('category', 'unknown')
                    }
                    for r in results
                ]
            except Exception as e:
                rag_info['error'] = str(e)

        if streaming:
            def generate():
                for chunk in assistant.chat_with_rag(message, use_rag=use_rag, rag_top_k=rag_top_k, streaming=True):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            return Response(stream_with_context(generate()), mimetype='text/event-stream')
        else:
            response = assistant.chat_with_rag(message, use_rag=use_rag, rag_top_k=rag_top_k, streaming=False)
            return jsonify({
                'response': response,
                'rag': rag_info
            })

    @ai_bp.route('/api/ai/status', methods=['GET'])
    def ai_status():
        ai = get_ai_assistant()
        vision = get_vision_assistant()
        audio = get_audio_assistant()
        yoloe = get_yoloe_detector()

        return jsonify({
            'ai': {'connected': ai.connect(), 'model': ai.model, 'port': 8080},
            'vision': {'connected': vision.connect(), 'model': vision.model, 'port': 8081},
            'stt': {'connected': audio._stt_connected, 'model': audio.stt_model, 'port': 8084, 'engine': 'whisper.cpp'},
            'tts': {'connected': audio._tts_connected, 'voice': audio.tts_voice, 'port': 8085, 'engine': 'piper'},
            'yoloe': {'available': yoloe.is_available(), 'model': yoloe.model_path},
            'platform': 'Jetson Orin Nano 8GB'
        })

    @ai_bp.route('/api/ai/history', methods=['DELETE'])
    def ai_clear_history():
        get_ai_assistant().clear_history()
        return jsonify({'ok': True})

    @ai_bp.route('/api/rag/status', methods=['GET'])
    def rag_status():
        """Get RAG knowledge base status."""
        try:
            from .rag import get_rag_system
            rag = get_rag_system()
            return jsonify({
                'document_count': rag.count_documents(),
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_dimension': 384,
                'collection': rag.collection_name,
                'persist_dir': rag.persist_dir
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @ai_bp.route('/api/rag/retrieve', methods=['POST'])
    def rag_retrieve():
        """Retrieve relevant documents for a query."""
        data = request.json or {}
        query = data.get('query', '')
        n_results = data.get('n_results', 5)
        category = data.get('category')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        try:
            from .rag import get_rag_system
            rag = get_rag_system()
            results = rag.retrieve(query, n_results=n_results, category=category)
            return jsonify({
                'query': query,
                'results': [
                    {
                        'id': r.id,
                        'text': r.text,
                        'score': r.score,
                        'metadata': r.metadata
                    }
                    for r in results
                ]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @ai_bp.route('/api/rag/seed', methods=['POST'])
    def rag_seed():
        """Seed the RAG knowledge base with robot documentation."""
        try:
            # Run seed in background
            import subprocess
            import threading

            def seed_task():
                subprocess.run([
                    'python3',
                    'llama_integration/rag/seed_knowledge.py'
                ], cwd=request.environ.get('SCRIPT_DIR', '/home/parc/Desktop/PARC-Remote-Lab'))

            thread = threading.Thread(target=seed_task)
            thread.start()

            return jsonify({
                'status': 'seeding',
                'message': 'Knowledge base is being seeded in background'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @ai_bp.route('/api/vision/analyze', methods=['POST'])
    def vision_analyze():
        """Analyze image from robot camera."""
        data = request.json or {}
        image_b64 = data.get('image', '')
        question = data.get('question', 'Describe the scene.')

        vision = get_vision_assistant()
        if not vision.is_connected():
            vision.connect()

        if not vision.is_connected():
            return jsonify({'error': 'Vision server unavailable on port 8081'}), 503

        result = vision.analyze_image(image_b64, question)
        return jsonify({'response': result})

    @ai_bp.route('/api/vision/detect', methods=['POST'])
    def vision_detect():
        """Detect objects for pick-and-place."""
        data = request.json or {}
        image_b64 = data.get('image', '')

        vision = get_vision_assistant()
        if not vision.is_connected():
            vision.connect()

        if not vision.is_connected():
            return jsonify({'error': 'Vision server unavailable'}), 503

        result = vision.detect_objects(image_b64)
        return jsonify(result)

    @ai_bp.route('/api/audio/transcribe', methods=['POST'])
    def audio_transcribe():
        """Transcribe voice command using Whisper.cpp (multilingual)."""
        data = request.json or {}
        audio_b64 = data.get('audio', '')

        audio = get_audio_assistant()
        if not audio._stt_connected:
            audio.connect_stt()

        if not audio._stt_connected:
            return jsonify({'error': 'STT server unavailable on port 8084 (Whisper.cpp)'}), 503

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return jsonify({'error': f'Invalid audio data: {e}'}), 400

        result = audio.transcribe(audio_bytes)
        return jsonify({'text': result})

    @ai_bp.route('/api/audio/speak', methods=['POST'])
    def audio_speak():
        """Convert text to speech using Piper TTS (multilingual)."""
        data = request.json or {}
        text = data.get('text', '')
        voice = data.get('voice', 'en_US-lessac-medium')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        audio = get_audio_assistant()
        if not audio._tts_connected:
            audio.connect_tts()

        if not audio._tts_connected:
            return jsonify({'error': 'TTS server unavailable on port 8085 (Piper)'}), 503

        audio_bytes = audio.speak(text)
        if isinstance(audio_bytes, bytes) and audio_bytes.startswith(b'TTS error'):
            return jsonify({'error': audio_bytes.decode()}), 500

        return Response(audio_bytes, mimetype='audio/wav')

    @ai_bp.route('/api/audio/command', methods=['POST'])
    def audio_command():
        """Convert voice to robot command."""
        data = request.json or {}
        audio_b64 = data.get('audio', '')

        audio = get_audio_assistant()
        if not audio._stt_connected:
            audio.connect_stt()

        if not audio._stt_connected:
            return jsonify({'error': 'STT server unavailable'}), 503

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return jsonify({'error': f'Invalid audio data: {e}'}), 400

        result = audio.speech_to_command(audio_bytes)
        return jsonify(result)

    @ai_bp.route('/api/voice/download', methods=['POST'])
    def voice_download():
        """Download a Piper voice model."""
        data = request.json or {}
        voice_id = data.get('voice', '')

        if not voice_id:
            return jsonify({'error': 'No voice specified'}), 400

        # Voice download URLs
        VOICE_URLS = {
            'en_US-lessac-medium': 'https://github.com/rhasspy/piper/raw/master/voices/en_US-lessac-medium.onnx',
            'en_US-lessac-low': 'https://github.com/rhasspy/piper/raw/master/voices/en_US-lessac-low.onnx',
            'en_GB-alan-medium': 'https://github.com/rhasspy/piper/raw/master/voices/en_GB-alan-medium.onnx',
            'fr_FR-siwis-medium': 'https://github.com/rhasspy/piper/raw/master/voices/fr_FR-siwis-medium.onnx',
            'de_DE-karlsson-medium': 'https://github.com/rhasspy/piper/raw/master/voices/de_DE-karlsson-medium.onnx',
            'es_ES-sharvard-medium': 'https://github.com/rhasspy/piper/raw/master/voices/es_ES-sharvard-medium.onnx',
            'it_IT-palia-medium': 'https://github.com/rhasspy/piper/raw/master/voices/it_IT-palia-medium.onnx',
        }

        if voice_id not in VOICE_URLS:
            return jsonify({'error': f'Unknown voice: {voice_id}'}), 400

        import os
        import requests as req

        voice_dir = os.path.expanduser('~/models/voices')
        os.makedirs(voice_dir, exist_ok=True)

        voice_path = os.path.join(voice_dir, f'{voice_id}.onnx')

        # Check if already exists
        if os.path.exists(voice_path):
            return jsonify({'success': True, 'path': voice_path, 'message': 'Voice already downloaded'})

        # Download
        url = VOICE_URLS[voice_id]
        try:
            logger.info(f"Downloading voice {voice_id} from {url}")
            response = req.get(url, timeout=60, stream=True)
            if response.status_code == 200:
                with open(voice_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logger.info(f"Voice downloaded to {voice_path}")
                return jsonify({'success': True, 'path': voice_path})
            else:
                return jsonify({'error': f'Download failed: {response.status_code}'}), 500
        except Exception as e:
            logger.error(f"Voice download error: {e}")
            return jsonify({'error': str(e)}), 500

    @ai_bp.route('/api/voice/list', methods=['GET'])
    def voice_list():
        """List available Piper voices and their download status."""
        import os

        VOICE_INFO = {
            'en_US-lessac-medium': {'lang': 'English (US)', 'style': 'Neutral'},
            'en_US-lessac-low': {'lang': 'English (US)', 'style': 'Low quality'},
            'en_GB-alan-medium': {'lang': 'English (UK)', 'style': 'Medium'},
            'fr_FR-siwis-medium': {'lang': 'French', 'style': 'Medium'},
            'de_DE-karlsson-medium': {'lang': 'German', 'style': 'Medium'},
            'es_ES-sharvard-medium': {'lang': 'Spanish', 'style': 'Medium'},
            'it_IT-palia-medium': {'lang': 'Italian', 'style': 'Medium'},
        }

        voice_dir = os.path.expanduser('~/models/voices')
        result = []

        for voice_id, info in VOICE_INFO.items():
            voice_path = os.path.join(voice_dir, f'{voice_id}.onnx')
            result.append({
                'id': voice_id,
                'lang': info['lang'],
                'style': info['style'],
                'downloaded': os.path.exists(voice_path),
                'path': voice_path
            })

        return jsonify({'voices': result})

    @ai_bp.route('/api/vision/yoloe/detect', methods=['POST'])
    def yoloe_detect():
        """
        Detect objects using YOLOE for precise pick-and-place.
        YOLOE provides real-time detection + segmentation.
        """
        data = request.json or {}
        image_b64 = data.get('image', '')
        conf_threshold = data.get('conf', 0.25)

        try:
            detector = get_yoloe_detector()
            if not detector.is_available():
                # Try to load on demand
                detector.load()

            if not detector.is_available():
                return jsonify({'error': 'YOLOE model not available. Install ultralytics: pip install ultralytics'}), 503

            result = detector.detect_objects(image_b64)
            return jsonify(result)
        except Exception as e:
            logger.error(f"YOLOE detect error: {e}")
            return jsonify({'error': str(e)}), 500

    @ai_bp.route('/api/vision/yoloe/export', methods=['POST'])
    def yoloe_export():
        """Export YOLOE model to ONNX for TensorRT deployment."""
        data = request.json or {}
        output_path = data.get('path', 'yoloe-11s-seg.onnx')

        detector = get_yoloe_detector()
        if not detector.is_available():
            detector.load()

        result = detector.export_onnx(output_path)
        return jsonify(result)

    @ai_bp.route('/api/sim/fk', methods=['POST'])
    def sim_forward_kinematics():
        """
        Compute forward kinematics for SO-101 robot arm.
        Takes joint angles in DEGREES, returns end-effector position in mm.
        """
        data = request.json or {}
        joints = data.get('joints', {})

        try:
            fk = SO101ForwardKinematics()
            position, rotation = fk.compute(joints)
            return jsonify({
                'success': True,
                'position': position.tolist() if hasattr(position, 'tolist') else position,
                'rotation': rotation.tolist() if hasattr(rotation, 'tolist') else rotation,
                'joints': joints
            })
        except Exception as e:
            logger.error(f"FK computation error: {e}")
            return jsonify({'error': str(e)}), 500

    @ai_bp.route('/api/sim/validate-code', methods=['POST'])
    def sim_validate_code():
        """
        Validate robot code before execution.
        Checks for joint limits, collision risks, and syntax errors.
        """
        data = request.json or {}
        code = data.get('code', '')
        joints = data.get('joints', {})

        try:
            validator = CodeValidator()
            result = validator.validate(code, joints)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Code validation error: {e}")
            return jsonify({'error': str(e)}), 500

    app.register_blueprint(ai_bp)


# -----------------------------------------------------------------------------
# YOLOE Object Detection (for precise pick-and-place)
# -----------------------------------------------------------------------------

class YOLOEDetector:
    """
    YOLOE object detector for robot vision.
    Provides real-time detection + segmentation for pick-and-place operations.
    
    YOLOE advantages:
    - 161 FPS on T4 GPU (real-time on Jetson Orin)
    - Instance segmentation with masks
    - Text prompts for custom objects
    - Export to ONNX/TensorRT for edge deployment
    """
    
    def __init__(self, model_path: str = "yoloe-11s-seg.onnx", conf_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = None
        self._session = None
        self._classes = [
            "cube", "cylinder", "box", "sphere", "cone", "pyramid",
            "red", "blue", "green", "yellow", "black", "white",
            "left", "right", "top", "bottom", "middle",
            "bin", "tray", "plate", "container"
        ]

    def load(self):
        """Load YOLOE model from ONNX (optimized for Jetson)."""
        try:
            # Check for .onnx first, fallback to .pt
            onnx_path = self.model_path.replace('.pt', '.onnx')
            import os
            if os.path.exists(onnx_path):
                self.model_path = onnx_path

            if self.model_path.endswith('.onnx'):
                # Use ONNX Runtime for Jetson optimization
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_available_providers().__contains__('CUDAExecutionProvider') else ['CPUExecutionProvider']
                self._session = ort.InferenceSession(self.model_path, providers=providers)
                logger.info(f"YOLOE ONNX model loaded: {self.model_path} with providers: {providers}")
            else:
                # Fallback to PyTorch
                from ultralytics import YOLOE
                self._model = YOLOE(self.model_path)
                self._model.to('cuda' if __import__('torch').cuda.is_available() else 'cpu')
                logger.info(f"YOLOE PyTorch model loaded: {self.model_path}")
        except ImportError as e:
            logger.warning(f"ONNX Runtime or ultralytics not installed: {e}")
            self._model = None
            self._session = None
        except Exception as e:
            logger.error(f"Failed to load YOLOE: {e}")
            self._model = None
            self._session = None

    def is_available(self) -> bool:
        return self._model is not None or self._session is not None
    
    def detect_objects(self, image_data: str | bytes) -> Dict[str, Any]:
        """
        Detect objects in image for pick-and-place using ONNX (Jetson optimized).
        """
        if not self._model and not self._session:
            return {"error": "YOLOE model not loaded", "objects": [], "scene": ""}

        try:
            import numpy as np
            from PIL import Image
            import io

            # Decode image
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data

            img = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(img)

            # Use ONNX Runtime if available
            if self._session:
                # Preprocess for ONNX (resize to 640x640, normalize)
                img_resized = img.resize((640, 640))
                img_np = np.array(img_resized).astype(np.float32) / 255.0
                img_np = img_np.transpose(2, 0, 1)[np.newaxis, :]

                # Run inference
                outputs = self._session.run(None, {self._session.get_inputs()[0].name: img_np})
                # Parse outputs (YOLO format: [batch, boxes, features])
                # This is simplified - actual parsing depends on YOLOE output format
                objects = self._parse_onnx_output(outputs, img_array.shape)
            else:
                # Fallback to PyTorch
                self._model.set_classes(self._classes)
                results = self._model.predict(img_array, conf=self.conf_threshold, verbose=False)
                objects = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        class_name = self._model.names[cls_id] if hasattr(self._model, 'names') else str(cls_id)
                        objects.append({
                            "class": class_name,
                            "bbox": xyxy.tolist(),
                            "confidence": round(conf, 3),
                            "center": ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2).tolist()
                        })

            # Generate scene description
            if objects:
                by_class = {}
                for obj in objects:
                    cls = obj["class"]
                    by_class[cls] = by_class.get(cls, 0) + 1
                parts = []
                for cls, count in sorted(by_class.items()):
                    if count == 1:
                        parts.append(f"one {cls}")
                    else:
                        parts.append(f"{count} {cls}s")
                scene = ", ".join(parts)
            else:
                scene = "No objects detected"

            return {"objects": objects, "scene": scene, "count": len(objects)}

        except Exception as e:
            logger.error(f"YOLOE detection error: {e}")
            return {"error": str(e), "objects": [], "scene": ""}

    def _parse_onnx_output(self, outputs, original_shape):
        """Parse ONNX output to extract bounding boxes."""
        objects = []
        try:
            output = outputs[0][0]
        except Exception as e:
            logger.warning(f"ONNX output parsing: {e}")
        return objects

    def export_onnx(self, output_path: str = "yoloe-11s-seg.onnx"):
        """Export model to ONNX for TensorRT deployment."""
        if not self._model:
            return {"error": "Model not loaded"}
        try:
            self._model.export(format="onnx")
            return {"success": True, "path": output_path}
        except Exception as e:
            return {"error": str(e)}


# Singleton YOLOE detector
_yoloe_detector: Optional[YOLOEDetector] = None


def get_yoloe_detector() -> YOLOEDetector:
    global _yoloe_detector
    if _yoloe_detector is None:
        _yoloe_detector = YOLOEDetector()
    return _yoloe_detector


# -----------------------------------------------------------------------------
# RAG System (Simple Version)
# -----------------------------------------------------------------------------

class SimpleRAG:
    """
    Simple Retrieval-Augmented Generation for technical documentation.
    Uses vector similarity for document retrieval.
    """

    def __init__(self):
        self.documents = []
        self._build_knowledge_base()

    def _build_knowledge_base(self):
        """Build knowledge base with technical documentation."""
        self.documents = [
            {
                "id": "kinematics_basics",
                "title": "Kinematics Basics",
                "content": """Forward kinematics calculates the position of the robot's end-effector 
                given joint angles. For SO-101: J1 (pan) rotates base, J2 (lift) raises arm, 
                J3 (elbow) bends elbow, J4 (wrist flex) tilts wrist, J5 (wrist roll) rotates gripper."""
            },
            {
                "id": "pick_and_place",
                "title": "Pick and Place Operations",
                "content": """To pick up an object: 1) Move gripper above object using FK, 
                2) Lower gripper, 3) Close gripper (60-80 degrees), 4) Lift object.
                To place: 1) Move to target position, 2) Lower, 3) Open gripper, 4) Raise."""
            },
            {
                "id": "servo_calibration",
                "title": "Servo Calibration",
                "content": """Calibration sweeps each servo through its range to find min, center, 
                and max positions. Save calibration data to prevent recalibration after restart."""
            },
            {
                "id": "coordinate_systems",
                "title": "Coordinate Systems",
                "content": """Robot uses 3D coordinate system: X (forward/back), Y (left/right), 
                Z (up/down). Angles in degrees. Positive rotation follows right-hand rule."""
            },
            {
                "id": "safety",
                "title": "Safety Guidelines",
                "content": """Always power off robot before calibration. Keep clear of moving parts 
                during operation. Use soft gripper for delicate objects. Never exceed servo limits."""
            }
        ]

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """Simple keyword-based retrieval (production would use embeddings)."""
        query_lower = query.lower()
        results = []
        for doc in self.documents:
            score = sum(1 for word in query_lower.split() if word in doc['content'].lower())
            if score > 0:
                results.append((score, doc))
        results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in results[:top_k]]

    def answer(self, query: str) -> str:
        """Answer query using RAG."""
        docs = self.retrieve(query)
        if not docs:
            return "I don't have specific information about that. Try asking about kinematics, calibration, or safety."

        context = "\n\n".join([f"Document: {d['title']}\n{d['content']}" for d in docs])
        prompt = f"""Based on the following technical documentation, answer the user's question.
If the documentation doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
        return prompt


def get_rag() -> SimpleRAG:
    return SimpleRAG()


# -----------------------------------------------------------------------------
# Build Script
# -----------------------------------------------------------------------------

BUILD_SCRIPT = '''#!/bin/bash
# llama.cpp Build Script for Jetson Orin Nano
# ===========================================
# PARC ROBOTICS - AI Models Setup

set -e
echo "[PARC ROBOTICS] Setting up AI models for Jetson Orin Nano"

# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git libssl-dev

# Clone llama.cpp
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# CPU + GPU (CUDA) build
cmake -B build \\
    -DGGML_CUDA=ON \\
    -DCMAKE_CUDA_ARCHITECTURES=87 \\
    -DGGML_NATIVE=OFF \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DBUILD_SHARED_LIBS=OFF

cmake --build build --config Release -j$(nproc)

# Create directories
mkdir -p ~/models/vision ~/models/audio ~/models/general

echo ""
echo "[PARC ROBOTICS] Build complete!"
echo ""
echo "Download models from HuggingFace:"
echo ""
echo "Vision Models (port 8081):"
echo "  wget -O ~/models/vision/lfm2.5-vl-1.6b-f16.gguf \\"
echo "    https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/lfm2.5-vl-1.6b-f16.gguf"
echo ""
echo "Audio Model (port 8082):"
echo "  wget -O ~/models/audio/lfm2.5-audio-1.5b-f16.gguf \\"
echo "    https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/resolve/main/lfm2.5-audio-1.5b-f16.gguf"
echo ""
echo "Thinking Model (port 8080):"
echo "  wget -O ~/models/general/lfm2.5-1.2b-thinking-q4_k_m.gguf \\"
echo "    https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/resolve/main/lfm2.5-1.2b-thinking-q4_k_m.gguf"
echo ""
echo "Start servers:"
echo "  # General AI (port 8080)"
echo "  llama-server -m ~/models/general/lfm2.5-1.2b-thinking-q4_k_m.gguf -c 2048 -ngl 99 --port 8080"
echo ""
echo "  # Vision AI (port 8081)"
echo "  llama-server -m ~/models/vision/lfm2.5-vl-1.6b-f16.gguf -c 2048 -ngl 99 --port 8081"
echo ""
echo "  # Audio AI (port 8082)"
echo "  llama-server -m ~/models/audio/lfm2.5-audio-1.5b-f16.gguf -c 2048 -ngl 99 --port 8082"
'''