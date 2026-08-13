#!/usr/bin/env python3
"""
teleop_r1_v7.py — v6 + RASTREO DE DEDOS.

NUEVO RESPECTO A v6
===================
1) MediaPipe HOLISTIC en vez de Pose.
   Holistic entrega en una sola pasada:
     - pose_world_landmarks  -> brazos/torso (igual que antes)
     - left_hand_landmarks   -> 21 puntos de la mano izquierda
     - right_hand_landmarks  -> 21 puntos de la mano derecha
   Asi cuerpo y manos quedan coherentes (cada mano atada a su muneca) sin
   correr dos detectores.

2) De los 21 puntos de cada mano se estima un "curl" (flexion) por dedo en
   [0,1] a partir de los angulos entre falanges. Ese curl se mapea al RANGO
   REAL de cada joint Revo2 (se leen del XML con jnt_range), de modo que:
     dedo estirado  -> curl 0 -> joint en su minimo (abierto)
     dedo cerrado   -> curl 1 -> joint en su maximo (cerrado)
   Cada dedo es independiente (pulgar, indice, medio, anular, menique) y el
   pulgar ademas mueve su metacarpo (cierre a traves de la palma).

3) La "palma" (orientacion de la mano) NO es un DOF nuevo del Revo2: ya la
   controlan el brazo + wrist_roll via IK. O sea la mano ya te sigue en
   posicion/orientacion; lo que agregamos aqui es el cierre de los dedos.

4) Escritura robusta por NOMBRE (igual criterio que v6): los 22 joints de
   dedos se resuelven con jnt_qposadr/jnt_dofadr/actuador, nunca por indice
   fijo. En modo CINEMATICO se escribe qpos directo; en DINAMICO se usa un
   PD suave con los limites de torque de cada actuador de dedo.

CAMARA (DroidCam)
=================
DroidCam por USB: `adb forward tcp:4747 tcp:4747` y CAM_URL abajo.
DroidCam por WiFi: pon CAM_URL = "http://IP_DEL_CELULAR:4747/video".

Teclas: T teleop | M espejo | K cine/dinamico | B soporte | F dedos on/off | R reset | ESC salir
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
XML_PATH = scene("r1_manos")          # XML CON MANOS
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

# --- Dedos -------------------------------------------------------------------
TRACK_FINGERS = True
FINGER_EMA = 0.35          # suavizado del curl (mas alto = mas responsivo)
FINGER_MAX_STEP = 0.20     # rad max por update por joint de dedo
CURL_FULL = 2.6            # suma de angulos (rad) que consideramos "puno cerrado"
THUMB_FULL = 2.0           # idem para el pulgar (tiene una falange menos)
HAND_KP = 4.0              # PD de dedos (solo modo dinamico)
HAND_KD = 0.15
HAND_HOLD_ON_LOSS = True   # si se pierde la mano: True=mantiene ultima pose, False=abre

NUM_DOFS = 24
DT = 0.002
ARM_UPDATE_HZ = 30

ARM_EMA = 0.18
ARM_MAX_STEP = 0.04

IK_ITERS = 6
IK_DAMP = 3e-3
IK_EPS = 1e-4
IK_GOOD = 0.18
IK_BAD = 0.55
W_UPPER = 2.0

KP_COM_X, KD_COM_X = 6.0, 1.2
KP_COM_Y, KD_COM_Y = 6.0, 1.2
COM_CLIP = 0.30

KP_ROLL, KD_ROLL = 2.5, 0.15
KP_PITCH, KD_PITCH = 2.5, 0.12
HIP_ROLL_GAIN = 0.45
HIP_PITCH_GAIN = 0.25


RENDER_HZ = 60        # cuadros por segundo del visor
KINEMATIC_HZ = 60     # ritmo del bucle en modo cinematico

# =============================================================================
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

# 11 joints por mano, en el mismo orden por lado.
# [thumb_meta, thumb_prox, thumb_dist, index_prox, index_dist,
#  middle_prox, middle_dist, ring_prox, ring_dist, pinky_prox, pinky_dist]
HAND_JOINTS_L = [
    "left_thumb_metacarpal_joint", "left_thumb_proximal_joint", "left_thumb_distal_joint",
    "left_index_proximal_joint", "left_index_distal_joint",
    "left_middle_proximal_joint", "left_middle_distal_joint",
    "left_ring_proximal_joint", "left_ring_distal_joint",
    "left_pinky_proximal_joint", "left_pinky_distal_joint",
]
HAND_JOINTS_R = [
    "right_thumb_metacarpal_joint", "right_thumb_proximal_joint", "right_thumb_distal_joint",
    "right_index_proximal_joint", "right_index_distal_joint",
    "right_middle_proximal_joint", "right_middle_distal_joint",
    "right_ring_proximal_joint", "right_ring_distal_joint",
    "right_pinky_proximal_joint", "right_pinky_distal_joint",
]
HAND_JOINTS = HAND_JOINTS_L + HAND_JOINTS_R

# expande 5 curls (pulgar,indice,medio,anular,menique) -> 11 factores por mano
# el pulgar alimenta [meta, prox, dist]; el resto [prox, dist]
def _curl_factors(c5):
    return np.array([c5[0], c5[0], c5[0],
                     c5[1], c5[1],
                     c5[2], c5[2],
                     c5[3], c5[3],
                     c5[4], c5[4]], dtype=np.float64)

L_HIP_PITCH, L_HIP_ROLL = 0, 1
R_HIP_PITCH, R_HIP_ROLL = 6, 7
L_ANK_PITCH, L_ANK_ROLL = 4, 5
R_ANK_PITCH, R_ANK_ROLL = 10, 11
WAIST_YAW_B = 13
ARM_SLICE = slice(14, 24)

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


def _angle(u, v):
    du = np.linalg.norm(u); dv = np.linalg.norm(v)
    if du < 1e-9 or dv < 1e-9:
        return 0.0
    c = np.clip(np.dot(u, v) / (du * dv), -1.0, 1.0)
    return float(np.arccos(c))


def _smoothstep(x, lo, hi):
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    t = (x - lo) / (hi - lo)
    return t * t * (3 - 2 * t)


def resolve(model, names):
    """nombre -> (qpos_adr, qvel_adr, ctrl_id) para una lista de joints."""
    joint_to_act = {}
    for aid in range(model.nu):
        jid = model.actuator_trnid[aid, 0]
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jn is not None:
            joint_to_act[jn] = aid
    qpos_adr, qvel_adr, ctrl_id = [], [], []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"No existe el joint '{name}'.")
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        qvel_adr.append(int(model.jnt_dofadr[jid]))
        ctrl_id.append(int(joint_to_act[name]))
    return np.array(qpos_adr), np.array(qvel_adr), np.array(ctrl_id)


# ================= indices de landmarks de mano de MediaPipe ==================
# 0 wrist | pulgar 1-4 | indice 5-8 | medio 9-12 | anular 13-16 | menique 17-20
_FINGERS = {
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}


def hand_curls_from_landmarks(pts):
    """pts: array (21,3) de la mano. Devuelve 5 curls [0,1]:
       [pulgar, indice, medio, anular, menique]."""
    # dedos largos: suma de angulos en PIP y DIP
    curls = []
    # pulgar primero
    a, b, c, d = pts[1], pts[2], pts[3], pts[4]
    ang = _angle(b - a, c - b) + _angle(c - b, d - c)
    curls.append(np.clip(ang / THUMB_FULL, 0.0, 1.0))
    for key in ("index", "middle", "ring", "pinky"):
        i0, i1, i2, i3 = _FINGERS[key]
        a, b, c, d = pts[i0], pts[i1], pts[i2], pts[i3]
        ang = _angle(b - a, c - b) + _angle(c - b, d - c)
        curls.append(np.clip(ang / CURL_FULL, 0.0, 1.0))
    return np.array(curls, dtype=np.float64)


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
        # curls por mano HUMANA (5 c/u); None si no se ve esa mano
        self.curls = {"left": None, "right": None}
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
        mp_holistic = mp.solutions.holistic
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        holistic = mp_holistic.Holistic(model_complexity=1,
                                        refine_face_landmarks=False,
                                        min_detection_confidence=0.6,
                                        min_tracking_confidence=0.6)
        print("[POSE] Camara lista (Holistic: cuerpo + manos).")
        while self.running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # ---- cuerpo (brazos/yaw) ----
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

                segLu = _norm(LE - LS); segLf = _norm(LW - LE)
                segRu = _norm(RE - RS); segRf = _norm(RW - RE)
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
            else:
                with self.lock:
                    self.dirs = None

            # ---- manos (dedos) ----
            def _pts(hlm):
                return np.array([[p.x, p.y, p.z] for p in hlm.landmark], dtype=np.float64)
            new = {"left": None, "right": None}
            if res.left_hand_landmarks:
                new["left"] = hand_curls_from_landmarks(_pts(res.left_hand_landmarks))
            if res.right_hand_landmarks:
                new["right"] = hand_curls_from_landmarks(_pts(res.right_hand_landmarks))
            with self.lock:
                self.curls = new

            if SHOW_CAMERA:
                if res.pose_landmarks:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks,
                                           mp_holistic.POSE_CONNECTIONS)
                    self._draw_frame_gizmo(cv2, frame, res)
                for hlm in (res.left_hand_landmarks, res.right_hand_landmarks):
                    if hlm:
                        mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"MIRROR={'ON' if self.mirror else 'OFF'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Teleop R1", frame)
                cv2.waitKey(1)
        cap.release()
        if SHOW_CAMERA:
            cv2.destroyAllWindows()

    def _draw_frame_gizmo(self, cv2, frame, res):
        with self.lock:
            cL, cR = self.conf
            yaw_raw = self.yaw
            curls = {k: (None if v is None else v.copy()) for k, v in self.curls.items()}
        # recomputar ejes solo para dibujar
        lm = res.pose_world_landmarks.landmark
        flip = np.array([1.0, -1.0, 1.0])
        P = lambda i: np.array([lm[i].x, lm[i].y, lm[i].z]) * flip
        LS, RS, LH, RH = P(11), P(12), P(23), P(24)
        up = _norm(0.5 * (LS + RS) - 0.5 * (LH + RH))
        left = LS - RS
        left = _norm(left - np.dot(left, up) * up)
        fwd = _norm(np.cross(left, up)) * FORWARD_SIGN

        h, w = frame.shape[:2]
        img = res.pose_landmarks.landmark
        cx = int((img[11].x + img[12].x) * 0.5 * w)
        cy = int((img[11].y + img[12].y) * 0.5 * h)

        def axis(vec, color):
            ex = int(cx + vec[0] * 70); ey = int(cy - vec[1] * 70)
            cv2.arrowedLine(frame, (cx, cy), (ex, ey), color, 3, tipLength=0.3)
        axis(fwd, (0, 0, 255)); axis(left, (0, 255, 0)); axis(up, (255, 0, 0))
        cv2.putText(frame, f"YAW={np.degrees(yaw_raw):+.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        def tag(c):
            return "HOLD" if c < ARM_CONF_LO else ("full" if c > ARM_CONF_HI else "mix")
        cv2.putText(frame, f"L={cL:.2f}[{tag(cL)}]  R={cR:.2f}[{tag(cR)}]",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        def ftxt(c5):
            return "--" if c5 is None else " ".join(f"{v:.1f}" for v in c5)
        cv2.putText(frame, f"handL[T I M R P]={ftxt(curls['left'])}",
                    (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        cv2.putText(frame, f"handR[T I M R P]={ftxt(curls['right'])}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    def get(self):
        with self.lock:
            return None if self.dirs is None else tuple(d.copy() for d in self.dirs)

    def get_conf(self):
        with self.lock:
            return self.conf

    def get_yaw(self):
        with self.lock:
            return self.yaw

    def get_curls(self):
        with self.lock:
            return {k: (None if v is None else v.copy()) for k, v in self.curls.items()}

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

        self.bqpos, self.bqvel, self.bctrl = resolve(self.model, BODY_JOINTS)
        print(f"[MODEL] nq={self.model.nq} nv={self.model.nv} nu={self.model.nu}")

        # ---- dedos: resolver por nombre + leer limites del XML --------------
        self.has_hands = False
        try:
            self.hqpos, self.hqvel, self.hctrl = resolve(self.model, HAND_JOINTS)
            lo, hi, clo, chi = [], [], [], []
            for name, aid in zip(HAND_JOINTS, self.hctrl):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                r = self.model.jnt_range[jid]
                lo.append(float(r[0])); hi.append(float(r[1]))
                cr = self.model.actuator_ctrlrange[aid]
                if self.model.actuator_ctrllimited[aid]:
                    clo.append(float(cr[0])); chi.append(float(cr[1]))
                else:
                    clo.append(-2.0); chi.append(2.0)
            self.hand_lo = np.array(lo); self.hand_hi = np.array(hi)
            self.hand_clo = np.array(clo); self.hand_chi = np.array(chi)
            self.has_hands = True
            print(f"[MODEL] manos detectadas: {len(HAND_JOINTS)} joints de dedos.")
        except Exception as e:
            print(f"[MODEL] sin manos en el XML ({e}); dedos desactivados.")

        self.track_fingers = TRACK_FINGERS and self.has_hands
        self.hand_target = (self.hand_lo.copy() if self.has_hands else None)

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

    def _write_body_qpos(self, vec24):
        self.data.qpos[self.bqpos] = vec24

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.74]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self._write_body_qpos(IDLE)
        if self.has_hands:
            self.data.qpos[self.hqpos] = self.hand_lo   # manos abiertas
            self.hand_target = self.hand_lo.copy()
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
                print(f"\n[MODO] {'CINEMATICO' if self.kinematic else 'DINAMICO'}")
            elif k in (ord('b'), ord('B')):
                self.hold_base = not self.hold_base
                print(f"\n[SOPORTE] {'ON' if self.hold_base else 'OFF'}")
            elif k in (ord('f'), ord('F')):
                if self.has_hands:
                    self.track_fingers = not self.track_fingers
                    print(f"\n[DEDOS] {'ON' if self.track_fingers else 'OFF'}")
            elif k in (ord('r'), ord('R')):
                self.reset()
                print("\n[RESET]")

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_cb)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -15
        self.viewer.cam.azimuth = 180

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
        pd[L_ANK_PITCH] -= pc; pd[R_ANK_PITCH] -= pc
        pd[L_ANK_ROLL] += rc; pd[R_ANK_ROLL] += rc
        pd[L_HIP_PITCH] -= pc * HIP_PITCH_GAIN; pd[R_HIP_PITCH] -= pc * HIP_PITCH_GAIN
        pd[L_HIP_ROLL] += rc * HIP_ROLL_GAIN; pd[R_HIP_ROLL] += rc * HIP_ROLL_GAIN
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
                fng = "F" if self.track_fingers else "-"
                print(f"\r[IK] izq={eL:.3f} der={eR:.3f}  gate L={gL:.2f} R={gR:.2f}  "
                      f"yaw={self.waist_yaw_target:+.2f}  [{modo}|{fng}]", end="", flush=True)
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

    def update_hands(self):
        """Devuelve el vector de 22 targets (rad) para los joints de dedos."""
        if not self.track_fingers:
            return self.hand_target
        curls = self.tracker.get_curls() if self.teleop_on else {"left": None, "right": None}

        # con espejo: robot-izq sigue a la mano humana derecha, y viceversa
        if self.tracker.mirror:
            src_L, src_R = curls["right"], curls["left"]
        else:
            src_L, src_R = curls["left"], curls["right"]

        def side_target(src, lo11, hi11):
            if src is None:
                if HAND_HOLD_ON_LOSS:
                    return None            # mantener lo que ya habia
                return lo11.copy()          # abrir
            return lo11 + (hi11 - lo11) * _curl_factors(src)

        tgtL = side_target(src_L, self.hand_lo[:11], self.hand_hi[:11])
        tgtR = side_target(src_R, self.hand_lo[11:], self.hand_hi[11:])

        desired = self.hand_target.copy()
        if tgtL is not None:
            desired[:11] = tgtL
        if tgtR is not None:
            desired[11:] = tgtR

        # suavizado + limite de paso
        smoothed = (1 - FINGER_EMA) * self.hand_target + FINGER_EMA * desired
        delta = np.clip(smoothed - self.hand_target, -FINGER_MAX_STEP, FINGER_MAX_STEP)
        self.hand_target = np.clip(self.hand_target + delta, self.hand_lo, self.hand_hi)
        return self.hand_target

    def apply_torque(self, target):
        q = self.data.qpos[self.bqpos]
        qd = self.data.qvel[self.bqvel]
        tau = STIFFNESS * (target - q) - DAMPING * qd
        self.data.ctrl[self.bctrl] = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

    def apply_finger_torque(self, hand_target):
        if not self.has_hands:
            return
        q = self.data.qpos[self.hqpos]
        qd = self.data.qvel[self.hqvel]
        tau = HAND_KP * (hand_target - q) - HAND_KD * qd
        self.data.ctrl[self.hctrl] = np.clip(tau, self.hand_clo, self.hand_chi)

    def _step_kinematic(self, arms, hands):
        self.data.qpos[0:7] = self._base0
        self.data.qvel[:] = 0.0
        self._write_body_qpos(IDLE)
        if TRACK_YAW:
            self.data.qpos[self.bqpos[WAIST_YAW_B]] = self.waist_yaw_target
        self.data.qpos[self.bqpos[14:24]] = arms
        if self.has_hands and hands is not None:
            self.data.qpos[self.hqpos] = hands
        mujoco.mj_forward(self.model, self.data)

    def _step_dynamic(self, arms, hands):
        target = self.balance_targets()
        if TRACK_YAW:
            target[WAIST_YAW_B] = self.waist_yaw_target
        target[ARM_SLICE] = arms
        self.apply_torque(target)
        if self.has_hands and hands is not None:
            self.apply_finger_torque(hands)
        mujoco.mj_step(self.model, self.data)
        if self.hold_base:
            self.data.qpos[0:7] = self._base0
            self.data.qvel[0:6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def run(self):
        print("\n== TELEOP R1 v7 (perf) ==")
        print("T on/off | M espejo | K cine/dyn | B soporte | F dedos | R reset | ESC salir")
 
        arm_dt = 1.0 / ARM_UPDATE_HZ
        render_dt = 1.0 / RENDER_HZ
        arms = IDLE[ARM_SLICE].copy()
        hands = self.hand_target
        t_arm = -1e9
        t_render = -1e9
        t_fps = time.perf_counter()
        frames = 0
        sim_time = 0.0
        wall0 = time.perf_counter()
 
        try:
            while self.viewer.is_running() and not self.should_exit:
                now = time.perf_counter() - wall0
 
                # brazos/dedos/yaw: solo a ARM_UPDATE_HZ (aqui vive la IK)
                updated = False
                if now - t_arm >= arm_dt:
                    t_arm = now
                    self._update_waist_yaw()
                    arms = self.update_arms()
                    hands = self.update_hands()
                    updated = True
 
                # avanzar la simulacion
                if self.kinematic:
                    if updated or (now - t_render >= render_dt):
                        self._step_kinematic(arms, hands)
                    sim_time = now
                else:
                    n = 0
                    while sim_time < now and n < 8:
                        self._step_dynamic(arms, hands)
                        sim_time += DT
                        n += 1
                    if sim_time < now - 0.25:
                        sim_time = now
 
                # render: solo a RENDER_HZ
                if now - t_render >= render_dt:
                    t_render = now
                    self.viewer.cam.lookat[:] = self.data.qpos[:3]
                    self.viewer.sync()
                    frames += 1
                    if time.perf_counter() - t_fps >= 2.0:
                        fps = frames / (time.perf_counter() - t_fps)
                        rtf = 1.0 if self.kinematic else min(1.0, sim_time / max(now, 1e-6))
                        print(f"\r[PERF] render={fps:4.1f} fps  rtf~{rtf:.2f}   ",
                              end="", flush=True)
                        frames = 0
                        t_fps = time.perf_counter()
 
                time.sleep(0.001)
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