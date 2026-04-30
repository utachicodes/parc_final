# Changelog

## [Unreleased] — 2026-04-30

### Fixed

#### BUG-1 · Camera stops when hand tracking starts
**Root cause:** `TeleopThread.run()` in `control_arm/app.py` launched the teleoperation
subprocess with `stdout=subprocess.DEVNULL`. The `read_frames()` thread inside the same
method then tried to call `self.process.stdout.read(4096)`, but `stdout` was `None`
(because DEVNULL disables pipe creation). This raised an `AttributeError` that was
silently swallowed by a bare `except: break`, terminating the frame-reader immediately.
Result: `teleop.get_frame()` always returned `None`, `/teleop/video_feed` served blank
frames, and the UI camera appeared frozen.

**Fix (`control_arm/app.py`):**
- Changed `stdout=subprocess.DEVNULL` → `stdout=subprocess.PIPE` so the frame-reader
  can actually receive data from the subprocess.
- Added `env['TELEOP_STREAM'] = '1'` to the subprocess environment so that all
  teleoperation scripts switch into MJPEG-to-stdout mode instead of trying to open a
  local `cv2.imshow()` window.
- Added `env['TELEOP_CAM_INDEX'] = str(self.cam_index)` so the configured camera index
  is forwarded to the subprocess.

---

#### BUG-2 · Robot unresponsive in Sign-Gesture mode
**Root cause:** `teleoperation/sign_gesture.py` hardcoded `PORT = "/dev/ttyACM0"` and
`BAUD = 1_000_000` as literals, completely ignoring the `TELEOP_PORT` / `TELEOP_BAUD`
environment variables that Flask injects. If the robot is on a different port the
connection silently fell back to `ttyACM0` or failed without feedback.

**Fix (`teleoperation/sign_gesture.py`):**
- `PORT = os.environ.get('TELEOP_PORT', '/dev/ttyACM0')`
- `BAUD = int(os.environ.get('TELEOP_BAUD', '1000000'))`

---

#### BUG-3 · `cv2.waitKey()` called in headless stream mode (sign_gesture)
**Root cause:** `sign_gesture.py` called `cv2.waitKey(1)` unconditionally inside the
capture loop. In stream mode (no display) OpenCV has no event loop attached, which can
stall or error out on some platforms, and was also the reason keyboard commands (`b`,
`r`, `q`) would fire unexpectedly during streaming.

**Fix (`teleoperation/sign_gesture.py`):**
- Moved `cv2.waitKey()` and all key-handling inside the `if not STREAM_MODE:` branch,
  so it only runs when a window is actually shown.
- `stream_frame(frame)` / `cv2.imshow()` call order corrected to output first, then
  poll keys.

---

#### BUG-4 · Wrong STREAM_MODE default in two_hands.py
**Root cause:** `teleoperation/two_hands.py` defaulted `TELEOP_STREAM` to `'0'`
(stream off), while `one_hand.py` defaulted to `'1'`. Inconsistent defaults meant
two-hand mode never streamed frames even after BUG-1 was fixed.

**Fix (`teleoperation/two_hands.py`):**
- `STREAM_MODE = os.environ.get('TELEOP_STREAM', '1') == '1'`

---

#### BUG-5 · Camera index ignored — all scripts hardcoded `VideoCapture(0)`
**Root cause:** All three teleoperation scripts (`one_hand.py`, `two_hands.py`,
`sign_gesture.py`) opened `cv2.VideoCapture(0)` regardless of the camera index
configured in Flask settings or passed via `TeleopThread(cam_index=...)`.

**Fix (all three scripts):**
- Read `TELEOP_CAM_INDEX` from the environment before opening the capture device.
  ```python
  cam_idx = int(os.environ.get('TELEOP_CAM_INDEX', '0'))
  cap = cv2.VideoCapture(cam_idx)
  ```

---

#### BUG-6 · Camera view not shown after starting teleop (play.html)
**Root cause:** The local `startTeleop()` in `play.html` removed the `setView('camera')`
call and replaced it with a comment "Don't switch view - teleop shows its own OpenCV window."
Since the backend now streams MJPEG via stdout, the UI must call `setView('camera')` to
display `/teleop/video_feed`. Additionally, `setView()` was changed to route both
`currentTeleop` and `currentTracking` to `/teleop/video_feed`, breaking regular
face/hand/pose tracking which needs `/video_feed`.

**Fix (`control_arm/templates/pages/play.html`):**
- Restored three-way routing in `setView('camera')`:
  `currentTeleop` → `/teleop/video_feed`, `currentTracking` → `/video_feed`, else → `/gripper_cam/video_feed`
- Added `setView('camera')` back to `startTeleop()` so the stream is visible immediately.
- Set `currentTracking = null` (not `mode`) when teleop starts — these are separate state flags.

---

---

## [Unreleased] — 2026-04-30 (Session 2)

### Fixed

#### BUG-7 · High latency in one_hand and two_hands camera feeds
**Root cause:** Triple-overhead pipeline: script encoded JPEG (q=95) → Flask decoded to numpy → Flask re-encoded (q=95). Also, `read_frames()` read only 4096 bytes per chunk (12+ reads per frame), and the camera capture buffer let frames pile up.

**Fix:**
- `TeleopThread.frame` now stores **raw JPEG bytes** (not numpy) — eliminated the decode step in Flask entirely.
- `get_frame()` returns bytes; `teleop_video_feed()` serves them directly with zero re-encoding.
- Read chunk size: 4096 → **65536 bytes** (1 read per frame on average).
- All three teleop scripts: `CAP_PROP_BUFFERSIZE=1`, `640×480` forced, JPEG quality 95 → **70**.
- Added `PYTHONUNBUFFERED=1` + `-u` flag to subprocess launch to prevent Python pipe buffering.

---

#### BUG-8 · All three teleop scripts crashed with ImportError on startup
**Root cause:** Scripts imported `PacketHandler` from `scservo_sdk`, which doesn't exist. The SDK exports `sms_sts` (a subclass of `protocol_packet_handler`). All method calls also had the wrong signature: 4 args `(port, id, addr, val)` instead of 3 `(id, addr, val)`.

**Fix (one_hand.py, two_hands.py, sign_gesture.py):**
- `from scservo_sdk import PortHandler, PacketHandler` → `from scservo_sdk import PortHandler, sms_sts`
- `PacketHandler(0)` → `sms_sts(self.port)` (created after port is opened)
- `packet.write2ByteTxRx(port, id, 42, pos)` → `packet.WritePosEx(id, pos, 300, 50)`
- `packet.write1ByteTxRx(port, id, addr, val)` → `packet.write1ByteTxRx(id, addr, val)`

---

#### BUG-9 · Camera conflict when switching teleop modes
**Root cause:** `api_teleop/start` waited only 0.3s after stopping the previous subprocess before starting the next one. The OS had not yet released the camera file descriptor, causing the new subprocess's `cap.read()` to fail silently.

**Fix (`control_arm/app.py`):** Increased stop-to-start delay from 0.3s to **1.0s**.

---

#### BUG-10 · IK endpoint accepted wrong key and started from zero joints
**Root cause:** `api_sim_ik` only accepted `position` key; callers using `target` got the default `[0,0,0.3]`. Also the solver always started from `joints=zeros`, which diverges for targets far from home.

**Fix (`control_arm/app.py`):**
- Added `data.get('position') or data.get('target', ...)` fallback.
- When robot is connected, seeds IK with current robot joint angles (fast convergence for local moves).
- Increased `max_iterations` 100 → 300, `tolerance` 1mm → 5mm.

---

### Files changed
| File | Change |
|------|--------|
| `control_arm/app.py` | `stdout=PIPE`, `TELEOP_STREAM=1`, `TELEOP_CAM_INDEX` injected into subprocess env |
| `teleoperation/sign_gesture.py` | PORT/BAUD from env vars; cam index from env; `cv2.waitKey` moved inside non-stream branch |
| `teleoperation/two_hands.py` | `STREAM_MODE` default changed to `'1'`; cam index from env |
| `teleoperation/one_hand.py` | cam index from env |
| `control_arm/templates/pages/play.html` | Restored `setView('camera')` in `startTeleop()`; fixed three-way video feed routing; added `flipCamera()` helper |
