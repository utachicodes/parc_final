FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir mujoco websockets numpy trimesh

COPY . .

EXPOSE 38000 38001

CMD ["python", "server.py"]
