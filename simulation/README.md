# SO101 Robot Arm Server

1. Install dependencies:
```bash
pip install mujoco websockets numpy
```

2. Run the server:
```bash
cd /home/parc/Desktop/InverseKinematics
python server.py
```

3. Open in browser:
```
http://localhost:57257
```

Replace placeholder meshes with real GLB model:

```javascript
// In browser console:
loadRealModel('/path/to/so101.glb')
```

The code, instructions, and GLTF guide are complete.