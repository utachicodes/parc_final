import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from spatialmath import SE3
import math

app = Flask(__name__)
CORS(app)

JOINT_LIMITS = np.array([
    [-2.2, -3.14, 0, -2.0, -3.14, -0.2],
    [2.2, 0.2, 3.14, 1.8, 3.14, 2.0]
])

def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def transl(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

def robot_fkine(q):
    T = np.eye(4)

    T = T @ transl(0, -0.0452, 0.0165)
    T = T @ rot_x(1.5708)
    T = T @ rot_y(q[0])

    T = T @ transl(0, 0.1025, 0.0306)
    T = T @ rot_x(1.5708)
    T = T @ rot_x(q[1])

    T = T @ transl(0, 0.11257, 0.028)
    T = T @ rot_x(-1.5708)
    T = T @ rot_x(q[2])

    T = T @ transl(0, 0.0052, 0.1349)
    T = T @ rot_x(-1.5708)
    T = T @ rot_x(q[3])

    T = T @ transl(0, -0.0601, 0)
    T = T @ rot_y(1.5708)
    T = T @ rot_y(q[4])

    T = T @ transl(-0.0202, -0.0244, 0)
    T = T @ rot_y(3.14159)
    T = T @ rot_z(q[5])

    return SE3.Rt(T[:3, :3], T[:3, 3])

def robot_jacob0(q):
    J = np.zeros((6, 6))
    T = np.eye(4)
    joint_positions = []
    joint_axes = []

    transforms = []

    T = T @ transl(0, -0.0452, 0.0165) @ rot_x(1.5708)
    transforms.append(T.copy())
    joint_axes.append(np.array([0, 1, 0]))

    T = T @ transl(0, 0.1025, 0.0306) @ rot_x(1.5708)
    transforms.append(T.copy())
    joint_axes.append(np.array([1, 0, 0]))

    T = T @ transl(0, 0.11257, 0.028) @ rot_x(-1.5708)
    transforms.append(T.copy())
    joint_axes.append(np.array([1, 0, 0]))

    T = T @ transl(0, 0.0052, 0.1349) @ rot_x(-1.5708)
    transforms.append(T.copy())
    joint_axes.append(np.array([1, 0, 0]))

    T = T @ transl(0, -0.0601, 0) @ rot_y(1.5708)
    transforms.append(T.copy())
    joint_axes.append(np.array([0, 1, 0]))

    T = T @ transl(-0.0202, -0.0244, 0) @ rot_y(3.14159)
    transforms.append(T.copy())
    joint_axes.append(np.array([0, 0, 1]))

    end_eff = SE3.Rt(T[:3, :3], T[:3, 3])

    for i in range(6):
        Ti = transforms[i]
        zi = Ti[:3, 2]
        pi = Ti[:3, 3]
        pe = end_eff.t

        J[:3, i] = np.cross(zi, pe - pi)
        J[3:, i] = zi

    return J

def numerical_ik(q_now, target_pose, tol=1e-4, ilimit=30, slimit=5):
    q = q_now.copy()
    x, y, z = target_pose[:3]

    for search in range(slimit):
        q_try = q.copy()
        for iteration in range(ilimit):
            T = robot_fkine(q_try)
            error_pos = np.array([x, y, z]) - T.t

            if np.linalg.norm(error_pos) < tol:
                return q_try.tolist(), True, iteration + 1, search + 1

            J = robot_jacob0(q_try)
            J_pos = J[:3, :]

            try:
                dq = np.linalg.pinv(J_pos) @ error_pos * 0.3
            except np.linalg.LinAlgError:
                break

            for i in range(6):
                q_try[i] += dq[i]
                q_try[i] = np.clip(q_try[i], JOINT_LIMITS[0][i], JOINT_LIMITS[1][i])

        if search == 0:
            final_q = q_try.copy()
            final_residual = np.linalg.norm(robot_fkine(q_try).t - np.array([x, y, z]))

        q = q_try + np.random.uniform(-0.2, 0.2, 6)
        q = np.clip(q, JOINT_LIMITS[0], JOINT_LIMITS[1])

    return final_q.tolist(), final_residual < tol, ilimit * slimit, slimit

def angle_axis(T, Td):
    e = np.empty(6)
    e[:3] = Td[:3, -1] - T[:3, -1]
    R = Td[:3, :3] @ T[:3, :3].T
    li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if np.linalg.norm(li) < 1e-10:
        if np.trace(R) > 0:
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        ln = np.linalg.norm(li)
        a = math.atan2(ln, np.trace(R) - 1) * li / ln

    e[3:] = a
    return e

def compute_manipulability(q):
    J = robot_jacob0(q)
    try:
        s = np.linalg.svd(J, compute_uv=False)
        m = math.sqrt(np.prod(s))
        condition = s[0] / s[-1] if s[-1] > 1e-10 else float('inf')
        return float(m), float(condition), s.tolist()
    except:
        return 0.0, float('inf'), [0] * 6

def p_servo(qpos, target_pose, gain=2.0, threshold=0.01):
    wTe = robot_fkine(qpos)

    x, y, z = target_pose[:3]
    Te = np.eye(4)
    Te[:3, :3] = wTe.R
    Te[:3, 3] = wTe.t

    Td = np.eye(4)
    Td[:3, 3] = [x, y, z]

    e = angle_axis(Te, Td)

    if isinstance(gain, (int, float)):
        k = gain * np.eye(6)
    else:
        k = np.diag(gain)

    v = k @ e
    arrived = np.linalg.norm(e[:3]) < threshold

    return v.tolist(), arrived

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/fk", methods=["POST"])
def forward_kinematics():
    data = request.json
    qpos = np.array(data["qpos"])
    T = robot_fkine(qpos)
    return jsonify({
        "position": T.t.tolist(),
        "rotation_matrix": T.R.tolist(),
        "euler_angles": [0, 0, 0]
    })

@app.route("/api/ik", methods=["POST"])
def inverse_kinematics():
    data = request.json
    q_now = np.array(data["q_now"])
    target = data["target"]

    q, success, iterations, searches = numerical_ik(q_now, target)

    return jsonify({
        "q": q,
        "success": bool(success),
        "iterations": int(iterations),
        "searches": int(searches),
        "residual": 0.0 if bool(success) else 1.0,
        "reason": "Success" if bool(success) else "Could not converge"
    })

@app.route("/api/jacobian", methods=["POST"])
def jacobian():
    data = request.json
    qpos = np.array(data["qpos"])
    J = robot_jacob0(qpos)
    return jsonify({"jacobian": J.tolist()})

@app.route("/api/manipulability", methods=["POST"])
def manipulability():
    data = request.json
    qpos = np.array(data["qpos"])
    m, cond, sv = compute_manipulability(qpos)
    return jsonify({
        "manipulability": m,
        "condition_number": cond,
        "singular_values": sv
    })

@app.route("/api/p_servo", methods=["POST"])
def position_servo():
    data = request.json
    qpos = np.array(data["qpos"])
    target = data["target"]
    gain = data.get("gain", 2.0)
    velocity, arrived = p_servo(qpos, target, gain)
    return jsonify({"velocity": velocity, "arrived": arrived})

@app.route("/api/robot_info", methods=["GET"])
def robot_info():
    return jsonify({
        "n_joints": 6,
        "joint_names": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
        "joint_limits": JOINT_LIMITS.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)