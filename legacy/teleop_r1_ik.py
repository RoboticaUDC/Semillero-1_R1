#!/usr/bin/env python3
"""
teleop_r1_ik.py — El R1 te imita, usando CINEMATICA INVERSA sobre el modelo real.

Diferencia con la version anterior:
  Antes calculabamos los angulos con formulas, asumiendo como estaban orientados
  los frames internos del hombro. Eso fallaba (el codo doblaba al reves).
  Ahora, en cada frame, le preguntamos a MuJoCo: "que angulos hacen que el brazo
  del robot apunte igual que el brazo de la persona?" y lo resolvemos numericamente.
  Asi los signos y offsets salen del modelo, no de suposiciones.

  Piernas + cintura : balance activo (reforzado en el eje lateral).
  Brazos            : IK contra las direcciones de tu brazo y antebrazo.

Requisitos:
    pip install mujoco numpy mediapipe opencv-python

Teclas (ventana de MuJoCo):
    T   : teleop ON/OFF
    M   : modo espejo
    R   : reset
    ESC : salir
"""

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ----------------------------------------------------------

from amo.paths import scene

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import threading
import time

import numpy as np
import mujoco
import mujoco.viewer

# =============================================================================
# CONFIGURACION
# =============================================================================
XML_PATH = scene("r1")
CAM_INDEX = 0
SHOW_CAMERA = True

MIRROR = True
FORWARD_SIGN = -1.0      # si los brazos van al reves adelante/atras, cambia a 1.0

NUM_DOFS = 24
DT = 0.002
ARM_UPDATE_HZ = 50

ARM_EMA = 0.20
ARM_MAX_STEP = 0.06

# IK
IK_ITERS = 4
IK_DAMP = 1e-2
IK_EPS = 1e-4

# =============================================================================
# INDICES (orden MuJoCo)
# =============================================================================
L_HIP_PITCH, L_HIP_ROLL = 0, 1
R_HIP_PITCH, R_HIP_ROLL = 6, 7
L_ANK_PITCH, L_ANK_ROLL = 4, 5
R_ANK_PITCH, R_ANK_ROLL = 10, 11

# =============================================================================
# GANANCIAS  (roll reforzado: los brazos de lado ya no lo tumban)
# =============================================================================
STIFFNESS = np.array([
    100, 100, 100, 200, 80, 60,
    100, 100, 100, 200, 80, 60,
    250, 250,
    50, 50, 40, 30, 20,
    50, 50, 40, 30, 20,
], dtype=np.float32)

DAMPING = np.array([
    5, 5, 5, 8, 5, 5,
    5, 5, 5, 8, 5, 5,
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
    -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,
    -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,
     0.0, 0.0,
     0.18, 0.18, 0.0, 1.5, 0.0,
     0.18,-0.18, 0.0, 1.5, 0.0,
], dtype=np.float32)

KP_ROLL, KD_ROLL = 2.5, 0.15
KP_PITCH, KD_PITCH = 2.0, 0.10
HIP_ROLL_GAIN = 0.45
HIP_PITCH_GAIN = 0.20

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
# indices en el vector de 24 (orden MuJoCo)
ARM_QIDX = {"left": [14, 15, 16, 17], "right": [19, 20, 21, 22]}


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def quat_to_euler(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.array([roll, pitch, yaw])


# =============================================================================
# CINEMATICA INVERSA DE UN BRAZO (4 DOF: sh_pitch, sh_roll, sh_yaw, elbow)
# =============================================================================
class ArmIK:
    """Encuentra los 4 angulos que hacen que el brazo del robot apunte
    en las mismas direcciones (brazo y antebrazo) que el de la persona."""

    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.qadr = {}
        self.limits = {}
        self.bodies = {}
        for side in ("left", "right"):
            adr, lim = [], []
            for jn in ARM_JOINTS[side]:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    raise RuntimeError(f"No encuentro el joint {jn}")
                adr.append(self.model.jnt_qposadr[jid])
                lim.append(self.model.jnt_range[jid].copy())
            self.qadr[side] = np.array(adr)
            self.limits[side] = np.array(lim)
            self.bodies[side] = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in ARM_BODIES[side]
            ]
            if min(self.bodies[side]) < 0:
                raise RuntimeError(f"No encuentro los cuerpos del brazo {side}")

    def _fk(self, side, q):
        """Devuelve (dir_brazo, dir_antebrazo) en el frame del pelvis."""
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0                     # quat identidad
        self.data.qpos[self.qadr[side]] = q
        mujoco.mj_forward(self.model, self.data)
        sh, el, wr = (self.data.xpos[b] for b in self.bodies[side])
        return _norm(el - sh), _norm(wr - el)

    def solve(self, side, u_target, w_target, q0):
        """Gauss-Newton amortiguado, arrancando desde q0 (frame anterior)."""
        q = np.clip(q0.copy(), self.limits[side][:, 0], self.limits[side][:, 1])
        target = np.concatenate([u_target, w_target])

        for _ in range(IK_ITERS):
            u, w = self._fk(side, q)
            r = np.concatenate([u, w]) - target
            if np.linalg.norm(r) < 1e-3:
                break
            # jacobiano por diferencias finitas (6x4)
            J = np.zeros((6, 4))
            for i in range(4):
                dq = q.copy()
                dq[i] += IK_EPS
                ui, wi = self._fk(side, dq)
                J[:, i] = (np.concatenate([ui, wi]) - np.concatenate([u, w])) / IK_EPS
            # paso amortiguado
            H = J.T @ J + IK_DAMP * np.eye(4)
            step = np.linalg.solve(H, -J.T @ r)
            q = np.clip(q + step, self.limits[side][:, 0], self.limits[side][:, 1])
        return q


# =============================================================================
# CAPTURA DE POSE
# =============================================================================
class PoseTracker(threading.Thread):
    def __init__(self, cam_index=CAM_INDEX, show=SHOW_CAMERA):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.show = show
        self.lock = threading.Lock()
        self.dirs = None       # (u_L, w_L, u_R, w_R) en frame del cuerpo
        self.running = True
        self.mirror = MIRROR

    def run(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            print("[POSE] Falta mediapipe/opencv. Sin teleop.")
            return

        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            print(f"[POSE] No abre la camara {self.cam_index}.")
            return

        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        pose = mp_pose.Pose(model_complexity=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        print("[POSE] Camara lista.")

        while self.running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_world_landmarks:
                lm = res.pose_world_landmarks.landmark
                flip = np.array([1.0, -1.0, 1.0])   # y de mediapipe crece hacia abajo
                P = lambda i: np.array([lm[i].x, lm[i].y, lm[i].z]) * flip
                LS, RS, LE, RE, LW, RW = P(11), P(12), P(13), P(14), P(15), P(16)
                LH, RH = P(23), P(24)

                up = _norm(0.5 * (LS + RS) - 0.5 * (LH + RH))
                left = LS - RS
                left = _norm(left - np.dot(left, up) * up)
                fwd = _norm(np.cross(left, up)) * FORWARD_SIGN

                def body(v):
                    return np.array([np.dot(v, fwd), np.dot(v, left), np.dot(v, up)])

                uL, wL = body(_norm(LE - LS)), body(_norm(LW - LE))
                uR, wR = body(_norm(RE - RS)), body(_norm(RW - RE))

                with self.lock:
                    self.dirs = (uL, wL, uR, wR)

                if self.show:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks,
                                           mp_pose.POSE_CONNECTIONS)
            else:
                with self.lock:
                    self.dirs = None

            if self.show:
                cv2.putText(frame, f"MIRROR={'ON' if self.mirror else 'OFF'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Teleop R1", frame)
                cv2.waitKey(1)

        cap.release()
        if self.show:
            cv2.destroyAllWindows()

    def get(self):
        with self.lock:
            return None if self.dirs is None else tuple(d.copy() for d in self.dirs)

    def stop(self):
        self.running = False


def mirror_dir(v):
    """Refleja una direccion respecto al plano sagital (y -> -y)."""
    return np.array([v[0], -v[1], v[2]])


# =============================================================================
# ENTORNO
# =============================================================================
class R1TeleopIK:
    def __init__(self):
        print(f"Cargando modelo: {XML_PATH}")
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        print("Preparando cinematica inversa...")
        self.ik = ArmIK(XML_PATH)
        self.q_ik = {"left": np.array(IDLE[[14, 15, 16, 17]], dtype=np.float64),
                     "right": np.array(IDLE[[19, 20, 21, 22]], dtype=np.float64)}

        self.teleop_on = True
        self.arm_target = IDLE[14:24].copy()

        try:
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
            self._has_gyro = True
        except Exception:
            self._has_gyro = False

        self.tracker = PoseTracker()
        self.tracker.start()
        self.reset()
        self._setup_viewer()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.74]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:7 + NUM_DOFS] = IDLE
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.arm_target = IDLE[14:24].copy()

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
            elif k in (ord('r'), ord('R')):
                self.reset()

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_cb)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -15
        self.viewer.cam.azimuth = 180

    def balance_targets(self):
        pd = IDLE.copy()
        roll, pitch, _ = quat_to_euler(self.data.qpos[3:7])
        ang_vel = (self.data.sensor("imu_ang_vel").data if self._has_gyro
                   else self.data.qvel[3:6])
        roll_corr = -(KP_ROLL * roll + KD_ROLL * ang_vel[0])
        pitch_corr = -(KP_PITCH * pitch + KD_PITCH * ang_vel[1])

        pd[L_ANK_PITCH] -= pitch_corr
        pd[R_ANK_PITCH] -= pitch_corr
        pd[L_ANK_ROLL] += roll_corr
        pd[R_ANK_ROLL] += roll_corr
        pd[L_HIP_PITCH] -= pitch_corr * HIP_PITCH_GAIN
        pd[R_HIP_PITCH] -= pitch_corr * HIP_PITCH_GAIN
        pd[L_HIP_ROLL] += roll_corr * HIP_ROLL_GAIN
        pd[R_HIP_ROLL] += roll_corr * HIP_ROLL_GAIN
        return pd

    def update_arms(self):
        dirs = self.tracker.get() if self.teleop_on else None
        if dirs is None:
            desired = IDLE[14:24].astype(np.float64)
        else:
            uL, wL, uR, wR = dirs
            if self.tracker.mirror:
                # tu brazo derecho -> brazo izquierdo del robot
                tgt_L = (mirror_dir(uR), mirror_dir(wR))
                tgt_R = (mirror_dir(uL), mirror_dir(wL))
            else:
                tgt_L = (uL, wL)
                tgt_R = (uR, wR)

            self.q_ik["left"] = self.ik.solve("left", tgt_L[0], tgt_L[1], self.q_ik["left"])
            self.q_ik["right"] = self.ik.solve("right", tgt_R[0], tgt_R[1], self.q_ik["right"])

            desired = np.zeros(10)
            desired[0:4] = self.q_ik["left"]
            desired[4] = 0.0                  # wrist_roll izq
            desired[5:9] = self.q_ik["right"]
            desired[9] = 0.0                  # wrist_roll der

        smoothed = (1 - ARM_EMA) * self.arm_target + ARM_EMA * desired
        delta = np.clip(smoothed - self.arm_target, -ARM_MAX_STEP, ARM_MAX_STEP)
        self.arm_target = (self.arm_target + delta).astype(np.float32)
        return self.arm_target

    def apply_torque(self, target):
        q = self.data.qpos[7:7 + NUM_DOFS]
        qd = self.data.qvel[6:6 + NUM_DOFS]
        tau = STIFFNESS * (target - q) - DAMPING * qd
        self.data.ctrl[:NUM_DOFS] = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

    def run(self):
        print("\n== TELEOP R1 (IK) ==")
        print("T: on/off | M: espejo | R: reset | ESC: salir\n")
        decim = int(round((1.0 / ARM_UPDATE_HZ) / DT))
        arms = IDLE[14:24].copy()
        step = 0
        try:
            while self.viewer.is_running() and not self.should_exit:
                t0 = time.time()
                target = self.balance_targets()
                if step % decim == 0:
                    arms = self.update_arms()
                target[14:24] = arms
                self.apply_torque(target)
                mujoco.mj_step(self.model, self.data)
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
    R1TeleopIK().run()