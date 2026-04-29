# SO-101 Robot Arm — Simulation & Control

Control a real **SO-101 robot arm** from a MuJoCo simulation. Move the simulated arm and the physical arm follows in real time.

```
┌─────────────────────┐        ┌─────────────────────┐
│   MuJoCo Viewer     │  qpos  │   Bridge / Server   │  joints  ┌──────────────┐
│   (viewer.py)       │───────>│   (bridge.py or     │─────────>│  Real SO-101 │
│                     │        │    server.py)        │          │  (app.py)    │
└─────────────────────┘        └─────────────────────┘          └──────────────┘
```

---

## Project Structure

```
simulation/
├── docker-compose.yml          # Orchestrates all services
├── .env                        # Configuration (ports, serial port, sync toggle)
├── simulation/
│   ├── Dockerfile
│   ├── viewer.py      # Native MuJoCo window — interactive, forwards joints to real arm
│   ├── server.py      # MuJoCo physics server: HTTP + WebSocket, drives web 3D viewer
│   ├── bridge.py      # Connects server.py WebSocket → real arm controller
│   └── web/           # Browser-based 3D viewer (Three.js)
└── control_arm/
    ├── Dockerfile
    └── app.py         # Flask app — controls the physical SO-101 via servo SDK
```

---

## Requirements

The servo SDK (`scservo_sdk`) must be present under `control_arm/stservo-env/`.

The MuJoCo model assets (XML + meshes) must exist at the path set by `SO101_DIR` in `.env`.

---

## Docker (recommended)

Everything runs from a single command. No terminals to juggle.

### 1. Configure `.env`

Edit `.env` at the project root — the defaults work out of the box if your setup matches:

| Variable | Default | Description |
|---|---|---|
| `COMPOSE_PROFILES` | _(empty)_ | Set to `sync` to enable simulation→arm bridge |
| `SERIAL_PORT` | `/dev/ttyACM0` | Serial port the SO-101 is on |
| `SO101_DIR` | `/home/parc/Desktop/...` | Path to the MuJoCo model directory |
| `RATE_HZ` | `20` | Bridge command rate (Hz) |
| `CONTROLLER_PORT` | `5000` | Exposed port for the controller UI |
| `HTTP_PORT` | `38000` | Exposed port for the web 3D viewer |
| `WS_PORT` | `38001` | Exposed WebSocket port |

### 2. Start

**Simulation + controller only (no real-arm sync):**
```bash
docker compose up --build
```

**With sync bridge (simulation drives the real arm):**
```bash
# in .env set:  COMPOSE_PROFILES=sync
docker compose up --build
```

Or inline without editing `.env`:
```bash
COMPOSE_PROFILES=sync docker compose up --build
```

### 3. Open

| Service | URL |
|---|---|
| Controller UI | http://localhost:5000 |
| Web 3D viewer | http://localhost:38000 |

Connect the arm in the controller UI, then interact with the web viewer — the bridge forwards every joint move to the real arm automatically.

---

## Manual (without Docker)

### Requirements

```bash
# Simulation
pip install mujoco websockets numpy trimesh

# Controller
pip install flask opencv-python mediapipe numpy pyserial
```

### Option A — Native MuJoCo Viewer (simplest)

```bash
# Terminal 1
cd control_arm && python app.py

# Terminal 2 — forwards joints directly to the arm
cd simulation && python viewer.py --controller http://127.0.0.1:5000
```

Hold **Ctrl** and **left-drag** any body part in the viewer window to move the arm.

Optional: `--rate 10` to slow commands down if servos feel choppy.

### Option B — Web Viewer + Bridge

```bash
# Terminal 1
cd control_arm && python app.py

# Terminal 2
cd simulation && python server.py
# Open http://localhost:38000

# Terminal 3
cd simulation && python bridge.py
```

---

## Joint Mapping

| qpos index | MuJoCo joint | Controller joint  |
|:----------:|:------------:|:-----------------:|
| 0          | Rotation     | shoulder_pan      |
| 1          | Pitch        | shoulder_lift     |
| 2          | Elbow        | elbow_flex        |
| 3          | Wrist_Pitch  | wrist_flex        |
| 4          | Wrist_Roll   | wrist_roll        |
| 5          | Jaw          | gripper           |

MuJoCo stores angles in **radians**; the controller expects **degrees** — the bridge and viewer handle this conversion automatically.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `[FAIL]` lines in viewer/bridge output | The arm is not connected — open `http://localhost:5000` and connect first |
| Arm doesn't move after perturbation | Check joint change threshold (0.5°) — try a larger drag |
| `ConnectionRefusedError` on bridge start | Start `server.py` before `bridge.py` |
| Viewer window appears frozen | Physics is running — try Ctrl+drag to interact |
