"""
SO101 Robot Arm Server with GLB conversion
"""

import asyncio
import json
import threading
import time
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import TCPServer
from urllib.parse import urlparse
import zipfile
import io

import numpy as np
import mujoco
import websockets

SO101_DIR = "/home/parc/Desktop/InverseKinematics/so101-inverse-kinematics-main/so101"
WEB_DIR = "/home/parc/Desktop/InverseKinematics/web"
STL_DIR = SO101_DIR + "/assets"
PORT = 38000
WS_PORT = 38001

# GLB cache
glb_cache = None

def create_glb():
    """Convert STL files to GLB in memory."""
    global glb_cache
    
    try:
        import trimesh
    except ImportError:
        print("trimesh not installed, using placeholder GLB")
        return None
    
    # Map body names to STL files
    body_stl_map = {
        'Base': ['Base.stl', 'Base_Motor.stl'],
        'Rotation_Pitch': ['Rotation_Pitch.stl', 'Rotation_Pitch_Motor.stl'],
        'Upper_Arm': ['Upper_Arm.stl', 'Upper_Arm_Motor.stl'],
        'Lower_Arm': ['Lower_Arm.stl', 'Lower_Arm_Motor.stl'],
        'Wrist_Pitch_Roll': ['Wrist_Pitch_Roll.stl', 'Wrist_Pitch_Roll_Motor.stl'],
        'Fixed_Jaw': ['Fixed_Jaw.stl', 'Fixed_Jaw_Motor.stl'],
        'Moving_Jaw': ['Moving_Jaw.stl']
    }
    
    # Colors
    orange = [228, 75, 0, 255]   # e54b00
    black = [25, 25, 25, 255]
    
    scene = trimesh.Scene()
    
    for body_name, stl_files in body_stl_map.items():
        for i, stl_file in enumerate(stl_files):
            stl_path = os.path.join(STL_DIR, stl_file)
            if os.path.exists(stl_path):
                try:
                    mesh = trimesh.load_mesh(stl_path)
                    
                    # Try to apply color via material
                    if hasattr(mesh, 'material'):
                        del mesh.material
                    
                    mesh.name = body_name
                    scene.add_geometry(mesh, geom_name=f"{body_name}_{i}")
                except Exception as e:
                    print(f"Error loading {stl_file}: {e}")
    
    # Export as GLB
    try:
        glb_data = scene.export(file_type='glb')
        glb_cache = glb_data
        print(f"GLB created: {len(glb_data)} bytes")
        return glb_data
    except Exception as e:
        print(f"GLB export error: {e}")
        return None

# Create GLB on startup
print("Converting STL files to GLB...")
create_glb()

class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.connected = set()
        
        os.chdir(SO101_DIR)
        self.model = mujoco.MjModel.from_xml_path("so_101.xml")
        self.data = mujoco.MjData(self.model)
        
        self.data.qpos[:] = [0, -1.57, 1.57, 0, 0, 0]
        mujoco.mj_step(self.model, self.data)
        
        self.body_names = [self.model.body(i).name for i in range(1, self.model.nbody) if self.model.body(i).name]
        self.joint_names = [self.model.joint(j).name for j in range(self.model.njnt) if self.model.joint(j).name]
        self.body_ids = {n: self.model.body(i).id for i, n in enumerate(self.body_names) if n}
    
    def get_state(self):
        with self.lock:
            state = {"type": "state", "body_names": self.body_names, "joint_names": self.joint_names,
                     "positions": {}, "quaternions": {}, "qpos": list(self.data.qpos[:6]), "time": float(self.data.time)}
            for name, bid in self.body_ids.items():
                xpos = self.data.xpos[bid]
                state["positions"][name] = [float(xpos[0]), float(xpos[1]), float(xpos[2])]
                xquat = self.data.xquat[bid]
                state["quaternions"][name] = [float(xquat[1]), float(xquat[2]), float(xquat[3]), float(xquat[0])]
            return state
    
    def set_targets(self, targets):
        with self.lock:
            for name, value in targets.items():
                jid = mujoco.mj_name2id(self.model, 0, name)
                if jid >= 0: self.data.ctrl[jid] = value
    
    def add_client(self, c): self.connected.add(c)
    def remove_client(self, c): self.connected.discard(c)
    
    def physics_loop(self):
        while self.running:
            with self.lock: mujoco.mj_step(self.model, self.data)
            time.sleep(1/120)
    
    def stop(self): self.running = False

robot = RobotState()
threading.Thread(target=robot.physics_loop, daemon=True).start()

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        
        # API endpoint
        if path == "/api/state":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(robot.get_state()).encode())
            return
        
        # Serve GLB model
        if path == "/model.glb":
            if glb_cache:
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Disposition', 'attachment; filename=so101.glb')
                self.end_headers()
                self.wfile.write(glb_cache)
            else:
                self.send_error(404)
            return
        
        # Root -> index.html
        if path in ["/", "/index.html"]:
            path = "/index.html"
        
        file_path = os.path.join(WEB_DIR, path.lstrip('/'))
        print(f"Request: {path} -> {file_path}, exists={os.path.exists(file_path)}")
        if os.path.exists(file_path) and os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1].lstrip('.')
            self.send_response(200)
            self.send_header('Content-Type', {'html': 'text/html', 'js': 'application/javascript'}.get(ext, 'text/plain'))
            self.end_headers()
            with open(file_path, 'rb') as f: self.wfile.write(f.read())
        else: self.send_error(404)
    
    def do_POST(self):
        if self.path == "/api/control":
            length = int(self.headers.get('Content-Length', 0))
            data = self.rfile.read(length)
            try:
                obj = json.loads(data)
                if "targets" in obj:
                    robot.set_targets(obj["targets"])
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                    return
            except: pass
        self.send_response(400)
        self.end_headers()
    
    def log_message(self, *args): pass

async def handle(ws):
    robot.add_client(ws)
    print(f"Client: {ws.remote_address}")
    await ws.send(json.dumps({"type": "init", "body_names": robot.body_names, "joint_names": robot.joint_names}))
    async for msg in ws:
        try:
            data = json.loads(msg)
            if data.get("type") == "control": robot.set_targets(data.get("targets", {}))
        except: pass
    robot.remove_client(ws)

async def broadcast():
    while True:
        msg = json.dumps(robot.get_state())
        dead = set()
        for c in robot.connected:
            try: await asyncio.wait_for(c.send(msg), timeout=0.01)
            except: dead.add(c)
        for c in dead: robot.remove_client(c)
        await asyncio.sleep(1/60)

async def main():
    print(f"SO101 Server on port {PORT}")
    print(f"Bodies: {robot.body_names}")
    print(f"Joints: {robot.joint_names}")
    
    # Run HTTP in thread
    class ReuseAddrTCPServer(TCPServer):
        allow_reuse_address = True
    
    http_server = ReuseAddrTCPServer(("127.0.0.1", PORT), Handler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    
    broadcast_task = asyncio.create_task(broadcast())
    
    print(f"Open http://localhost:{PORT}")
    print("Press Ctrl+C to stop")
    
    try:
        async with websockets.serve(handle, "127.0.0.1", WS_PORT):
            await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        robot.stop()
        broadcast_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        robot.stop()
        print("\nServer stopped.")