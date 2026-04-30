"""
Streaming utilities for teleop scripts.
Enables MJPEG streaming to stdout instead of cv2.imshow()
"""
import os
import sys
import cv2
import numpy as np

STREAM_MODE = os.environ.get('TELEOP_STREAM', '0') == '1'

def get_frame():
    """Get frame from video capture - placeholder, overridden by script"""
    return None

def stream_frame(frame):
    """
    Output frame as MJPEG to stdout if in stream mode.
    If not in stream mode, display locally with cv2.imshow()
    """
    if STREAM_MODE:
        # Encode as JPEG and write to stdout in MJPEG format
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buf = cv2.imencode('.jpg', frame, encode_param)
        sys.stdout.buffer.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
        sys.stdout.buffer.write(buf.tobytes())
        sys.stdout.buffer.write(b'\r\n')
        sys.stdout.buffer.flush()
    else:
        # Normal display mode - handled by script
        pass

def should_quit():
    """Check if quit signal received (for stream mode via stdin)"""
    if STREAM_MODE:
        # In stream mode, check if parent process is still running
        return False
    return None

def cleanup_stream():
    """Cleanup at end"""
    if STREAM_MODE:
        sys.stdout.buffer.flush()