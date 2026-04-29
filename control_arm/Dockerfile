FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    "flask>=3.0.0" \
    "opencv-python-headless>=4.9.0" \
    "mediapipe>=0.10.0" \
    "numpy>=1.24.0" \
    "pyserial>=3.5" \
    "requests>=2.31.0" \
    "Pillow>=10.0.0"

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
