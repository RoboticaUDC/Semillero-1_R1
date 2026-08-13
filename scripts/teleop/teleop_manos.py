#!/usr/bin/env python3
"""
teleop_r1_v6.py — igual que v5 pero AHORA funciona con el modelo que tiene
                  las manos Revo2 (y sigue funcionando con el modelo viejo).

QUE SE ARREGLO RESPECTO A v5
============================
El problema NO era el XML. Era que v5 escribia/leia el estado por INDICE FIJO
y contiguo (qpos[7:31], qpos[21:31], qvel[6:30], ctrl[:24]). Eso solo es valido
si los 24 joints del cuerpo estan pegados uno detras de otro en qpos.

Al agregar las manos, MuJoCo mete las 11 articulaciones de la mano izquierda
ENTRE el brazo izquierdo y el derecho. Entonces:

    qpos[26..30]  ya NO es el brazo derecho -> son dedos de la mano izquierda.
    El brazo derecho real quedo en qpos[37..41] y v5 nunca lo tocaba.
    Resultado: brazo derecho congelado (~90) y balance/lectura desalineados.

FIX: resolver TODAS las direcciones por NOMBRE una sola vez (jnt_qposadr,
jnt_dofadr, y el actuador que mueve cada joint) y usar esos arreglos de indices
para leer/escribir. Asi da igual cuantos DOF de mano haya en medio.

Los dedos se dejan con torque 0 (sueltos). Si luego quieres controlarlos,
sus actuadores estan disponibles por nombre (ver HAND_JOINTS al final).

Teclas:  T teleop | M espejo | K cinematico/dinamico | B soporte | R reset | ESC salir
"""

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

from amo.math_utils import quat_to_euler
from amo.paths import scene

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import threading
import time

import numpy as np
import mujoco
import mujoco.viewer

# =============================================================================
XML_PATH = scene("r1_manos")          # <-- pon aqui el XML CON MANOS
CAM_URL = "http://127.0.0.1:4747/video"
SHOW_CAMERA = True

MIRROR = True
FORWARD_SIGN = -1.0

KINEMATIC_HOLD = True
HOLD_BASE = False

TRACK_YAW = True
YAW_SIGN = 1.0
YAW_GAIN = 1.0
YAW_DEADZONE = 0.10
YAW_CLIP = 0.6
YAW_EMA = 0.15

ARM_CONF_LO = 0.75
ARM_CONF_HI = 1.0

NUM_DOFS = 24                     # joints del CUERPO (12 piernas + 2 cintura + 10 brazos)
DT = 0.002
ARM_UPDATE_HZ = 30

ARM_EMA = 0.18
ARM_MAX_STEP = 0.04

IK_ITERS = 10
IK_DAMP = 3e-3
IK_EPS = 1e-4
IK_GOOD = 0.12
IK_BAD = 0.55
W_UPPER = 2.0

KP_COM_X, KD_COM_X = 6.0, 1.2
KP_COM_Y, KD_COM_Y = 6.0, 1.2
COM_CLIP = 0.30

KP_ROLL, KD_ROLL = 2.5, 0.15
KP_PITCH, KD_PITCH = 2.5, 0.12
HIP_ROLL_GAIN = 0.45
HIP_PITCH_GAIN = 0.25

# =============================================================================
# ORDEN CANONICO de los 24 joints del cuerpo. Debe coincidir 1:1 con IDLE,
# STIFFNESS, DAMPING y TORQUE_LIMITS (mismo orden, mismos indices).
BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_roll_joint", "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
]

# Dedos (opcional, por si luego quieres cerrarlos). 11 por mano.
HAND_JOINTS = [
    "left_thumb_metacarpal_joint", "left_thumb_proximal_joint", "left_thumb_distal_joint",
    "left_index_proximal_joint", "left_index_distal_joint",
    "left_middle_proximal_joint", "left_middle_distal_joint",
    "left_ring_proximal_joint", "left_ring_distal_joint",
    "left_pinky_proximal_joint", "left_pinky_distal_joint",
    "right_thumb_metacarpal_joint", "right_thumb_proximal_joint", "right_thumb_distal_joint",
    "right_index_proximal_joint", "right_index_distal_joint",
    "right_middle_proximal_joint", "right_middle_distal_joint",
    "right_ring_proximal_joint", "right_ring_distal_joint",
    "right_pinky_proximal_joint", "right_pinky_distal_joint",
]

# Indices DENTRO del vector de 24 (no son indices de qpos)
L_HIP_PITCH, L_HIP_ROLL = 0, 1
R_HIP_PITCH, R_HIP_ROLL = 6, 7
L_ANK_PITCH, L_ANK_ROLL = 4, 5
R_ANK_PITCH, R_ANK_ROLL = 10, 11
WAIST_YAW_B = 13          # posicion de waist_yaw dentro de BODY_JOINTS
ARM_SLICE = slice(14, 24) # los 10 valores de brazos dentro del vector de 24

STIFFNESS = np.array([
    120, 120, 100, 220, 120, 80,
    120, 120, 100, 220, 120, 80,
    250, 250,
    50, 50, 40, 30, 20,
    50, 50, 40, 30, 20,
], dtype=np.float32)

DAMPING = np.array([
    6, 6, 5, 9, 8, 6,
    6, 6, 5, 9, 8, 6,
    25, 25,
    5, 5, 4, 4, 4,
    5, 5, 4, 4, 4,
], dtype=np.float32)

TORQUE_LIMITS = np.array([
    88, 139, 88, 139, 50, 50,
    88, 139, 88, 139, 50, 50,
    88, 50,
    25, 25, 25, 25, 25,
    25, 25, 25, 25, 25,
], dtype=np.float32)

IDLE = np.array([
    -0.10, 0.0, 0.0, 0.20, -0.10, 0.0,
    -0.10, 0.0, 0.0, 0.20, -0.10, 0.0,
     0.0, 0.0,
     0.18, 0.18, 0.0, 1.5, 0.0,
     0.18,-0.18, 0.0, 1.5, 0.0,
], dtype=np.float32)

ARM_JOINTS = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint"],
}
ARM_BODIES = {
    "left": ["left_shoulder_pitch_link", "left_elbow_link", "left_wrist_roll_link"],
    "right": ["right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_link"],
}
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
Q_IDLE = {"left": np.array([0.18, 0.18, 0.0, 1.5]),
          "right": np.array([0.18, -0.18, 0.0, 1.5])}


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _smoothstep(x, lo, hi):
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    t = (x - lo) / (hi - lo)
    return t * t * (3 - 2 * t)


def build_index_maps(model):
    """Resuelve por NOMBRE las direcciones de qpos, qvel y ctrl de los 24
    joints del cuerpo. Esto es lo que hace que todo funcione con o sin manos."""
    # joint -> actuador que lo mueve
    joint_to_act = {}
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jn is not None:
            joint_to_act[jn] = aid

    qpos_adr, qvel_adr, ctrl_id = [], [], []
    for name in BODY_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"No existe el joint '{name}' en el modelo.")
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        qvel_adr.append(int(model.jnt_dofadr[jid]))
        ctrl_id.append(int(joint_to_act[name]))
    return (np.array(qpos_adr), np.array(qvel_adr), np.array(ctrl_id))


# =============================================================================
class ArmIK:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.qadr, self.limits, self.bodies = {}, {}, {}
        for side in ("left", "right"):
            adr, lim = [], []
            for jn in ARM_JOINTS[side]:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                adr.append(self.model.jnt_qposadr[jid])
                lim.append(self.model.jnt_range[jid].copy())
            self.qadr[side] = np.array(adr)
            self.limits[side] = np.array(lim)
            self.bodies[side] = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                                 for b in ARM_BODIES[side]]

    def _fk(self, side, q):
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0
        self.data.qpos[self.qadr[side]] = q
        mujoco.mj_forward(self.model, self.data)
        sh, el, wr = (self.data.xpos[b] for b in self.bodies[side])
        return _norm(el - sh), _norm(wr - el)

    def _res(self, side, q, u_t, w_t):
        u, w = self._fk(side, q)
        return np.concatenate([W_UPPER * (u - u_t), w - w_t])

    def _gn(self, side, u_t, w_t, q0):
        lo, hi = self.limits[side][:, 0], self.limits[side][:, 1]
        q = np.clip(np.asarray(q0, dtype=np.float64).copy(), lo, hi)
        for _ in range(IK_ITERS):
            r = self._res(side, q, u_t, w_t)
            if np.linalg.norm(r) < 1e-3:
                break
            J = np.zeros((6, 4))
            for i in range(4):
                dq = q.copy()
                dq[i] += IK_EPS
                J[:, i] = (self._res(side, dq, u_t, w_t) - r) / IK_EPS
            H = J.T @ J + IK_DAMP * np.eye(4)
            q = np.clip(q + np.linalg.solve(H, -J.T @ r), lo, hi)
        return q, float(np.linalg.norm(self._res(side, q, u_t, w_t)))

    def solve(self, side, u_t, w_t, q_prev):
        q, err = self._gn(side, u_t, w_t, q_prev)
        if err < IK_GOOD:
            return q, err
        s = 1.0 if side == "left" else -1.0
        for s0 in (np.array([0.0, 0.05 * s, 0.0, 0.15]),
                   np.array([-1.2, 0.10 * s, 0.0, 0.60]),
                   np.array([0.0, 1.20 * s, 0.0, 0.40]),
                   Q_IDLE[side]):
            qi, ei = self._gn(side, u_t, w_t, s0)
            if ei < err:
                q, err = qi, ei
            if err < IK_GOOD:
                break
        return q, err


# =============================================================================
class PoseTracker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.dirs = None
        self.conf = (0.0, 0.0)
        self.yaw = 0.0
        self.running = True
        self.mirror = MIRROR

    def run(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            print("[POSE] Falta mediapipe/opencv.")
            return
        cap = cv2.VideoCapture(CAM_URL)
        if not cap.isOpened():
            print(f"[POSE] No abre la camara: {CAM_URL}")
            return
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        pose = mp_pose.Pose(model_complexity=1,
                            min_detection_confidence=0.6,
                            min_tracking_confidence=0.6)
        print("[POSE] Camara lista.")
        while self.running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                lm = res.pose_world_landmarks.landmark
                flip = np.array([1.0, -1.0, 1.0])
                P = lambda i: np.array([lm[i].x, lm[i].y, lm[i].z]) * flip
                LS, RS, LE, RE, LW, RW = P(11), P(12), P(13), P(14), P(15), P(16)
                LH, RH = P(23), P(24)
                up = _norm(0.5 * (LS + RS) - 0.5 * (LH + RH))
                left = LS - RS
                left = _norm(left - np.dot(left, up) * up)
                fwd = _norm(np.cross(left, up)) * FORWARD_SIGN

                segLu = _norm(LE - LS)
                segLf = _norm(LW - LE)
                segRu = _norm(RE - RS)
                segRf = _norm(RW - RE)

                inplane = lambda v: float(np.sqrt(max(0.0, 1.0 - v[2] * v[2])))
                cL = min(inplane(segLu), inplane(segLf))
                cR = min(inplane(segRu), inplane(segRf))

                sh = LS - RS
                yaw_raw = np.arctan2(sh[2], abs(sh[0]) + 1e-6)

                def body(v):
                    return np.array([np.dot(v, fwd), np.dot(v, left), np.dot(v, up)])

                with self.lock:
                    self.dirs = (body(segLu), body(segLf), body(segRu), body(segRf))
                    self.conf = (cL, cR)
                    self.yaw = float(yaw_raw)

                if SHOW_CAMERA:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks,
                                           mp_pose.POSE_CONNECTIONS)
                    self._draw_frame_gizmo(cv2, frame, res, fwd, left, up,
                                           yaw_raw, cL, cR)
            else:
                with self.lock:
                    self.dirs = None
            if SHOW_CAMERA:
                cv2.putText(frame, f"MIRROR={'ON' if self.mirror else 'OFF'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Teleop R1", frame)
                cv2.waitKey(1)
        cap.release()
        if SHOW_CAMERA:
            cv2.destroyAllWindows()

    def _draw_frame_gizmo(self, cv2, frame, res, fwd, left, up, yaw_raw, cL, cR):
        h, w = frame.shape[:2]
        img = res.pose_landmarks.landmark
        cx = int((img[11].x + img[12].x) * 0.5 * w)
        cy = int((img[11].y + img[12].y) * 0.5 * h)
        scale = 70

        def axis(vec, color):
            ex = int(cx + vec[0] * scale)
            ey = int(cy - vec[1] * scale)
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 3, tipLength=0.3)

        axis(fwd,  (0, 0, 255))
        axis(left, (0, 255, 0))
        axis(up,   (255, 0, 0))
        cv2.putText(frame, f"YAW={np.degrees(yaw_raw):+.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        def tag(c):
            return "HOLD" if c < ARM_CONF_LO else ("full" if c > ARM_CONF_HI else "mix")
        cv2.putText(frame, f"L={cL:.2f}[{tag(cL)}]  R={cR:.2f}[{tag(cR)}]",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def get(self):
        with self.lock:
            return None if self.dirs is None else tuple(d.copy() for d in self.dirs)

    def get_conf(self):
        with self.lock:
            return self.conf

    def get_yaw(self):
        with self.lock:
            return self.yaw

    def stop(self):
        self.running = False


def mirror_dir(v):
    return np.array([v[0], -v[1], v[2]])


# =============================================================================
class R1Teleop:
    def __init__(self):
        print(f"Cargando modelo: {XML_PATH}")
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        # ---- mapas de indices por NOMBRE (clave del arreglo) ----------------
        self.bqpos, self.bqvel, self.bctrl = build_index_maps(self.model)
        print(f"[MODEL] nq={self.model.nq} nv={self.model.nv} nu={self.model.nu}")
        print(f"[MAP] right_shoulder_pitch -> qpos[{self.bqpos[19]}] "
              f"qvel[{self.bqvel[19]}] ctrl[{self.bctrl[19]}]  (antes se asumia qpos[26])")

        self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                         for b in FOOT_BODIES]

        print("Preparando IK...")
        self.ik = ArmIK(XML_PATH)
        self.q_ik = {s: Q_IDLE[s].copy() for s in ("left", "right")}

        self.teleop_on = True
        self.kinematic = KINEMATIC_HOLD
        self.hold_base = HOLD_BASE
        self.arm_target = IDLE[ARM_SLICE].copy()
        self.waist_yaw_target = 0.0
        self._com_err_prev = np.zeros(2)
        self._last_print = 0.0

        try:
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
            self._has_gyro = True
        except Exception:
            self._has_gyro = False

        self.tracker = PoseTracker()
        self.tracker.start()
        self.reset()
        self._setup_viewer()

    # ---- helpers de estado (por nombre) -------------------------------------
    def _write_body_qpos(self, vec24):
        """Escribe los 24 joints del cuerpo en sus qpos reales."""
        self.data.qpos[self.bqpos] = vec24

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.74]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self._write_body_qpos(IDLE)          # <-- por nombre, ya no qpos[7:31]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.arm_target = IDLE[ARM_SLICE].copy()
        self.waist_yaw_target = 0.0
        self.q_ik = {s: Q_IDLE[s].copy() for s in ("left", "right")}
        self._com_err_prev = np.zeros(2)
        self._base0 = self.data.qpos[0:7].copy()

    def _setup_viewer(self):
        self.should_exit = False

        def key_cb(k):
            if k == 256:
                self.should_exit = True
            elif k in (ord('t'), ord('T')):
                self.teleop_on = not self.teleop_on
                print(f"\n[TELEOP] {'ON' if self.teleop_on else 'OFF'}")
            elif k in (ord('m'), ord('M')):
                self.tracker.mirror = not self.tracker.mirror
                print(f"\n[MIRROR] {'ON' if self.tracker.mirror else 'OFF'}")
            elif k in (ord('k'), ord('K')):
                self.kinematic = not self.kinematic
                self.reset()
                print(f"\n[MODO] {'CINEMATICO (fijo, solo brazos)' if self.kinematic else 'DINAMICO (equilibrio real)'}")
            elif k in (ord('b'), ord('B')):
                self.hold_base = not self.hold_base
                print(f"\n[SOPORTE] {'ON' if self.hold_base else 'OFF'} (solo aplica en modo dinamico)")
            elif k in (ord('r'), ord('R')):
                self.reset()
                print("\n[RESET]")

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_cb)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -15
        self.viewer.cam.azimuth = 180

    # ---- balance dinamico ---------------------------------------------------
    def balance_targets(self):
        pd = IDLE.copy()
        roll, pitch, yaw = quat_to_euler(self.data.qpos[3:7])
        w = (self.data.sensor("imu_ang_vel").data if self._has_gyro
             else self.data.qvel[3:6])

        com = self.data.subtree_com[0][:2]
        feet = 0.5 * (self.data.xpos[self.foot_ids[0]][:2] +
                      self.data.xpos[self.foot_ids[1]][:2])
        d = com - feet
        c, s = np.cos(-yaw), np.sin(-yaw)
        err = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        derr = (err - self._com_err_prev) / DT
        self._com_err_prev = err

        com_pitch = np.clip(KP_COM_X * err[0] + KD_COM_X * derr[0], -COM_CLIP, COM_CLIP)
        com_roll = np.clip(-(KP_COM_Y * err[1] + KD_COM_Y * derr[1]), -COM_CLIP, COM_CLIP)

        pitch_eff = pitch + com_pitch
        roll_eff = roll + com_roll

        rc = -(KP_ROLL * roll_eff + KD_ROLL * w[0])
        pc = -(KP_PITCH * pitch_eff + KD_PITCH * w[1])

        pd[L_ANK_PITCH] -= pc
        pd[R_ANK_PITCH] -= pc
        pd[L_ANK_ROLL] += rc
        pd[R_ANK_ROLL] += rc
        pd[L_HIP_PITCH] -= pc * HIP_PITCH_GAIN
        pd[R_HIP_PITCH] -= pc * HIP_PITCH_GAIN
        pd[L_HIP_ROLL] += rc * HIP_ROLL_GAIN
        pd[R_HIP_ROLL] += rc * HIP_ROLL_GAIN
        return pd

    def _update_waist_yaw(self):
        if not TRACK_YAW:
            self.waist_yaw_target = 0.0
            return
        yaw = self.tracker.get_yaw() if self.teleop_on else 0.0
        if yaw is None:
            yaw = 0.0
        if self.tracker.mirror:
            yaw = -yaw
        if abs(yaw) < YAW_DEADZONE:
            yaw = 0.0
        else:
            yaw = yaw - np.sign(yaw) * YAW_DEADZONE
        tgt = np.clip(YAW_SIGN * YAW_GAIN * yaw, -YAW_CLIP, YAW_CLIP)
        self.waist_yaw_target = (1 - YAW_EMA) * self.waist_yaw_target + YAW_EMA * tgt

    def update_arms(self):
        dirs = self.tracker.get() if self.teleop_on else None
        if dirs is None:
            self.q_ik = {s: Q_IDLE[s].copy() for s in ("left", "right")}
            desired = IDLE[ARM_SLICE].astype(np.float64)
        else:
            uL, wL_, uR, wR = dirs
            conf = self.tracker.get_conf()
            gL = _smoothstep(conf[0], ARM_CONF_LO, ARM_CONF_HI)
            gR = _smoothstep(conf[1], ARM_CONF_LO, ARM_CONF_HI)
            if self.tracker.mirror:
                tL = (mirror_dir(uR), mirror_dir(wR))
                tR = (mirror_dir(uL), mirror_dir(wL_))
                gL, gR = gR, gL
            else:
                tL, tR = (uL, wL_), (uR, wR)
            qL, eL = self.ik.solve("left", tL[0], tL[1], self.q_ik["left"])
            qR, eR = self.ik.solve("right", tR[0], tR[1], self.q_ik["right"])
            now = time.time()
            if now - self._last_print > 1.0:
                self._last_print = now
                modo = "CINE" if self.kinematic else "DYN "
                print(f"\r[IK] izq={eL:.3f} der={eR:.3f}  gate L={gL:.2f} R={gR:.2f}  "
                      f"yaw={self.waist_yaw_target:+.2f}  [{modo}]", end="", flush=True)
            if eL < IK_BAD:
                self.q_ik["left"] = gL * qL + (1 - gL) * self.q_ik["left"]
            if eR < IK_BAD:
                self.q_ik["right"] = gR * qR + (1 - gR) * self.q_ik["right"]
            desired = np.zeros(10)
            desired[0:4] = self.q_ik["left"]
            desired[5:9] = self.q_ik["right"]

        smoothed = (1 - ARM_EMA) * self.arm_target + ARM_EMA * desired
        delta = np.clip(smoothed - self.arm_target, -ARM_MAX_STEP, ARM_MAX_STEP)
        self.arm_target = (self.arm_target + delta).astype(np.float32)
        return self.arm_target

    def apply_torque(self, target):
        # lee q y qd de los joints del cuerpo POR NOMBRE (no por slice)
        q = self.data.qpos[self.bqpos]
        qd = self.data.qvel[self.bqvel]
        tau = STIFFNESS * (target - q) - DAMPING * qd
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
        self.data.ctrl[self.bctrl] = tau        # escribe a los actuadores correctos
        # los dedos quedan en 0 (sueltos); no los tocamos

    # ---- pasos por modo -----------------------------------------------------
    def _step_kinematic(self, arms):
        self.data.qpos[0:7] = self._base0
        self.data.qvel[:] = 0.0
        self._write_body_qpos(IDLE)
        if TRACK_YAW:
            self.data.qpos[self.bqpos[WAIST_YAW_B]] = self.waist_yaw_target
        self.data.qpos[self.bqpos[14:24]] = arms      # brazos a sus qpos reales
        mujoco.mj_forward(self.model, self.data)

    def _step_dynamic(self, arms):
        target = self.balance_targets()
        if TRACK_YAW:
            target[WAIST_YAW_B] = self.waist_yaw_target
        target[ARM_SLICE] = arms
        self.apply_torque(target)
        mujoco.mj_step(self.model, self.data)
        if self.hold_base:
            self.data.qpos[0:7] = self._base0
            self.data.qvel[0:6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def run(self):
        print("\n== TELEOP R1 v6 (con manos Revo2) ==")
        print("T: on/off | M: espejo | K: cine/dinamico | B: soporte | R: reset | ESC: salir")
        print(f"Modo inicial: {'CINEMATICO (fijo)' if self.kinematic else 'DINAMICO'}\n")
        decim = int(round((1.0 / ARM_UPDATE_HZ) / DT))
        arms = IDLE[ARM_SLICE].copy()
        step = 0
        try:
            while self.viewer.is_running() and not self.should_exit:
                t0 = time.time()
                if step % decim == 0:
                    self._update_waist_yaw()
                    arms = self.update_arms()

                if self.kinematic:
                    self._step_kinematic(arms)
                else:
                    self._step_dynamic(arms)

                self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()
                step += 1
                dt = DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        except KeyboardInterrupt:
            pass
        finally:
            self.tracker.stop()
            try:
                if self.viewer.is_running():
                    self.viewer.close()
            except Exception:
                pass
            print("\nFin")


if __name__ == "__main__":
    R1Teleop().run()