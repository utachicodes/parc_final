#!/usr/bin/env python3
"""
YOLOE Vision Server for PARC ROBOTICS
======================================
Standalone server for real-time object detection using YOLOE.
Runs on port 8083 to complement LFM2.5-VL on port 8081.

Usage:
    python3 vision_server.py [--model yoloe-11s-seg.pt] [--port 8083]

YOLOE provides:
- 161 FPS on T4 GPU (real-time on Jetson Orin)
- Instance segmentation with masks
- Text prompts for custom objects
- Export to ONNX/TensorRT for edge deployment
"""

import argparse
import base64
import json
import logging
import sys
import time
from io import BytesIO
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLOEVisionServer:
    """YOLOE-based vision server for object detection."""

    def __init__(self, model_path: str = "yoloe-11s-seg.pt", conf_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = None
        self._classes = [
            "cube", "cylinder", "box", "sphere", "cone", "pyramid",
            "red", "blue", "green", "yellow", "black", "white",
            "left", "right", "top", "bottom", "middle",
            "bin", "tray", "plate", "container",
            "gripper", "robot", "arm", "base", "sensor"
        ]

    def load(self) -> bool:
        """Load YOLOE model."""
        try:
            from ultralytics import YOLOE
            import torch

            logger.info(f"Loading YOLOE model: {self.model_path}")
            self._model = YOLOE(self.model_path)

            # Use GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._model.to(device)
            logger.info(f"YOLOE loaded on {device}")

            return True
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load YOLOE: {e}")
            return False

    def is_available(self) -> bool:
        return self._model is not None

    def detect(self, image_data: str) -> Dict[str, Any]:
        """
        Detect objects in image.

        Args:
            image_data: Base64 encoded image string

        Returns:
            Detection results with objects, scene description
        """
        if not self._model:
            return {"error": "Model not loaded", "objects": [], "scene": ""}

        try:
            # Decode image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))
            img_array = np.array(img)

            # Run detection
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
                        "center": [float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2)]
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
            logger.error(f"Detection error: {e}")
            return {"error": str(e), "objects": [], "scene": ""}

    def export_tensorrt(self, output_dir: str = ".") -> Dict[str, Any]:
        """Export model to TensorRT for Jetson optimization."""
        if not self._model:
            return {"error": "Model not loaded"}

        try:
            logger.info("Exporting to TensorRT engine...")
            export_path = self._model.export(format="engine")
            return {"success": True, "path": export_path}
        except Exception as e:
            logger.error(f"Export error: {e}")
            return {"error": str(e)}


def create_flask_app(server: YOLOEVisionServer):
    """Create Flask app for YOLOE server."""
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'ok' if server.is_available() else 'error',
            'model': server.model_path,
            'available': server.is_available()
        })

    @app.route('/detect', methods=['POST'])
    def detect():
        data = request.json or {}
        image_b64 = data.get('image', '')
        conf = data.get('conf', server.conf_threshold)

        if not image_b64:
            return jsonify({'error': 'No image provided'}), 400

        # Temporarily update conf if different
        original_conf = server.conf_threshold
        if conf != original_conf:
            server.conf_threshold = conf

        result = server.detect(image_b64)

        # Restore conf
        server.conf_threshold = original_conf

        return jsonify(result)

    @app.route('/export', methods=['POST'])
    def export_model():
        data = request.json or {}
        output_dir = data.get('output_dir', '.')
        return jsonify(server.export_tensorrt(output_dir))

    @app.route('/classes', methods=['GET'])
    def list_classes():
        return jsonify({'classes': server._classes})

    @app.route('/classes', methods=['POST'])
    def set_classes():
        data = request.json or {}
        classes = data.get('classes', [])
        if classes:
            server._classes = classes
            return jsonify({'success': True, 'classes': server._classes})
        return jsonify({'error': 'No classes provided'}), 400

    return app


def main():
    parser = argparse.ArgumentParser(description='YOLOE Vision Server for PARC ROBOTICS')
    parser.add_argument('--model', default='yoloe-11s-seg.pt', help='YOLOE model path')
    parser.add_argument('--port', type=int, default=8083, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    # Create server
    server = YOLOEVisionServer(model_path=args.model, conf_threshold=args.conf)

    # Load model
    logger.info(f"Loading YOLOE model: {args.model}")
    if not server.load():
        logger.error("Failed to load YOLOE model")
        sys.exit(1)

    # Create Flask app
    app = create_flask_app(server)

    logger.info(f"Starting YOLOE Vision Server on {args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info("  GET  /health          - Server health")
    logger.info("  POST /detect          - Detect objects in image")
    logger.info("  POST /export          - Export to TensorRT")
    logger.info("  GET  /classes         - List detection classes")
    logger.info("  POST /classes         - Set detection classes")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()