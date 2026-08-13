#!/usr/bin/env python3
"""
teleop_r1.py — El R1 te imita en tiempo real.

  Piernas + cintura : controlador de BALANCE ACTIVO (PD + correccion de
                      roll/pitch en tobillos y caderas). El robot se mantiene
                      de pie y COMPENSA el desequilibrio que causan los brazos.
  Brazos            : copian tus brazos, capturados con la webcam usando
                      MediaPipe Pose.

Requisitos:
    pip install mujoco torch numpy mediapipe opencv-python
    (torch solo si luego quieres cargar una politica; aqui no hace falta)

Uso:
    python teleop_r1.py

Teclas (en la ventana de MuJoCo):
    T   : activar/desactivar el seguimiento de brazos (teleop ON/OFF)
    M   : alternar modo espejo
    R   : reset del robot
    ESC : salir

Notas:
  - Ponte de frente a la camara, a ~2 m, que se te vea de la cintura para arriba.
  - La muneca (wrist_roll) no se estima; queda en 0.
  - Si un brazo se mueve al reves, cambia MIRROR (tecla M) o revisa FORWARD_SIGN.
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
SHOW_CAMERA = True       # ventana con tu esqueleto dibujado (bueno para el video)

MIRROR = True            # True: el robot es tu espejo (tu mano derecha = su izquierda)
FORWARD_SIGN = -1.0
USE_YAW = False        # True para activar la rotacion del hombro (mas ruidosa)       # si el robot mete los brazos hacia atras, pon -1.0

NUM_DOFS = 24
DT = 0.002               # paso de fisica
ARM_UPDATE_HZ = 50       # frecuencia de refresco de los targets de brazo

# Suavizado y limite de velocidad de los brazos (clave para no desequilibrar)
ARM_EMA = 0.10           # 0..1 (mas bajo = mas suave/lento)
ARM_MAX_STEP = 0.05      # rad por paso de control (limita tirones bruscos)

# =============================================================================
# INDICES (orden MuJoCo, qpos[7:31])
#  0-5   pierna izq: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
#  6-11  pierna der: idem
#  12-13 cintura: waist_roll, waist_yaw
#  14-18 brazo izq: sh_pitch, sh_roll, sh_yaw, elbow, wrist_roll
#  19-23 brazo der: idem
# =============================================================================
L_HIP_PITCH, L_HIP_ROLL = 0, 1
R_HIP_PITCH, R_HIP_ROLL = 6, 7
L_ANK_PITCH, L_ANK_ROLL = 4, 5
R_ANK_PITCH, R_ANK_ROLL = 10, 11
ARM_L = slice(14, 19)
ARM_R = slice(19, 24)

# =============================================================================
# GANANCIAS (las tuyas, probadas en bandaEstabilidad_R1.py)
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

# Balance
KP_ROLL, KD_ROLL = 2.5, 0.15
KP_PITCH, KD_PITCH = 2.0, 0.10
HIP_ROLL_GAIN = 0.45
HIP_PITCH_GAIN = 0.20

# Limites de los joints de brazo (del r1.xml)
ARM_LIMITS_L = np.array([
    [-3.1416, 2.0944],    # sh_pitch
    [-0.22689, 2.4784],   # sh_roll  (positivo = abre a la izquierda)
    [-1.9199, 1.9199],    # sh_yaw
    [-0.97564, 2.1852],   # elbow
    [-1.9199, 1.9199],    # wrist_roll
])
ARM_LIMITS_R = np.array([
    [-3.1416, 2.0944],
    [-2.47849, 0.2268],   # sh_roll  (negativo = abre a la derecha)
    [-1.9199, 1.9199],
    [-0.97564, 2.1852],
    [-1.9199, 1.9199],
])


# =============================================================================
# UTILIDADES
# =============================================================================
def quat_to_euler(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.array([roll, pitch, yaw])


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def arm_angles_from_points(shoulder, elbow, wrist, f, l, up):
    """Convierte 3 puntos 3D (hombro, codo, muneca) a angulos de un brazo IZQUIERDO.

    Marco del cuerpo: x = adelante (f), y = izquierda (l), z = arriba (up).
    Brazo en reposo (colgando) => vector (0, 0, -1) => todos los angulos 0.

    Modelo del hombro del R1 (anidado): pitch (eje Y) -> roll (eje X) -> yaw (eje Z).
      u = (-cos(roll)*sin(pitch),  sin(roll),  -cos(roll)*cos(pitch))

    Devuelve (sh_pitch, sh_roll, sh_yaw, elbow).
    """
    def to_body(v):
        return np.array([np.dot(v, f), np.dot(v, l), np.dot(v, up)])

    u = to_body(_norm(elbow - shoulder))     # brazo (hombro -> codo)
    w = to_body(_norm(wrist - elbow))        # antebrazo (codo -> muneca)

    ux, uy, uz = u
    sh_roll = np.arcsin(np.clip(uy, -1.0, 1.0))
    sh_pitch = np.arctan2(-ux, -uz)

    # Codo: angulo entre brazo y antebrazo (0 = estirado)
    elbow_ang = np.arccos(np.clip(np.dot(w, u), -1.0, 1.0))

    # Yaw: direccion en la que dobla el codo, alrededor del eje del brazo
    ex = np.array([np.cos(sh_pitch), 0.0, -np.sin(sh_pitch)])
    ey = np.array([np.sin(sh_roll) * np.sin(sh_pitch),
                   np.cos(sh_roll),
                   np.sin(sh_roll) * np.cos(sh_pitch)])
    p = w - np.dot(w, u) * u
    if USE_YAW and np.linalg.norm(p) > 1e-3 and elbow_ang > 0.6:
        ex_yaw = -_norm(p)
        sh_yaw = np.arctan2(np.dot(ex_yaw, ey), np.dot(ex_yaw, ex))
    else:
        sh_yaw = 0.0

    return sh_pitch, sh_roll, sh_yaw, elbow_ang


def mirror_arm(a):
    """Convierte angulos de brazo izquierdo en los del derecho (reflejo sagital)."""
    sh_pitch, sh_roll, sh_yaw, elbow_ang = a
    return sh_pitch, -sh_roll, -sh_yaw, elbow_ang


# =============================================================================
# CAPTURA DE POSE (hilo aparte, para no frenar la fisica)
# =============================================================================
class PoseTracker(threading.Thread):
    """Lee la webcam y publica los 10 angulos de brazo del R1."""

    def __init__(self, cam_index=CAM_INDEX, show=SHOW_CAMERA):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.show = show
        self.lock = threading.Lock()
        self.arm_angles = None      # np.array(10) o None si no hay deteccion
        self.running = True
        self.ok = False
        self.mirror = MIRROR

    def run(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            print("[POSE] mediapipe/opencv no instalados. Corriendo sin teleop.")
            print("       pip install mediapipe opencv-python")
            return

        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            print(f"[POSE] No pude abrir la camara {self.cam_index}. Sin teleop.")
            return

        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        pose = mp_pose.Pose(model_complexity=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        self.ok = True
        print("[POSE] Camara lista. Ponte de frente, que se te vea el torso.")

        while self.running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_world_landmarks:
                lm = res.pose_world_landmarks.landmark
                P = lambda i: np.array([lm[i].x, lm[i].y, lm[i].z])
                # MediaPipe: y crece hacia ABAJO -> lo invertimos para tener z=arriba
                flip = np.array([1.0, -1.0, 1.0])
                LS, RS = P(11) * flip, P(12) * flip
                LE, RE = P(13) * flip, P(14) * flip
                LW, RW = P(15) * flip, P(16) * flip
                LH, RH = P(23) * flip, P(24) * flip

                mid_sh = 0.5 * (LS + RS)
                mid_hip = 0.5 * (LH + RH)
                up = _norm(mid_sh - mid_hip)
                left = LS - RS
                left = _norm(left - np.dot(left, up) * up)
                fwd = _norm(np.cross(left, up)) * FORWARD_SIGN

                # angulos "como brazo izquierdo" para cada lado
                a_left = arm_angles_from_points(LS, LE, LW, fwd, left, up)
                a_right_raw = arm_angles_from_points(RS, RE, RW, fwd, left, up)

                if self.mirror:
                    # espejo: tu brazo derecho -> brazo izquierdo del robot
                    robot_L = mirror_arm(a_right_raw)
                    robot_R = mirror_arm(a_left)
                else:
                    # mismo lado: tu izquierdo -> izquierdo del robot
                    robot_L = a_left
                    robot_R = a_right_raw

                angs = np.array([
                    robot_L[0], robot_L[1], robot_L[2], robot_L[3], 0.0,
                    robot_R[0], robot_R[1], robot_R[2], robot_R[3], 0.0,
                ], dtype=np.float32)

                with self.lock:
                    self.arm_angles = angs

                if self.show:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks,
                                           mp_pose.POSE_CONNECTIONS)
            else:
                with self.lock:
                    self.arm_angles = None

            if self.show:
                cv2.putText(frame, f"MIRROR={'ON' if self.mirror else 'OFF'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Teleop R1 - tu pose", frame)
                cv2.waitKey(1)

        cap.release()
        if self.show:
            cv2.destroyAllWindows()

    def get(self):
        with self.lock:
            return None if self.arm_angles is None else self.arm_angles.copy()

    def stop(self):
        self.running = False


# =============================================================================
# ENTORNO
# =============================================================================
class R1Teleop:
    def __init__(self, xml_path=XML_PATH):
        print(f"Cargando modelo: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        self.teleop_on = True
        self.arm_target = IDLE[14:24].copy()   # target suavizado de brazos

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

        def key_cb(keycode):
            if keycode == 256:
                self.should_exit = True
            elif keycode in (ord('t'), ord('T')):
                self.teleop_on = not self.teleop_on
                print(f"\n[TELEOP] {'ON' if self.teleop_on else 'OFF'}")
            elif keycode in (ord('m'), ord('M')):
                self.tracker.mirror = not self.tracker.mirror
                print(f"\n[MIRROR] {'ON' if self.tracker.mirror else 'OFF'}")
            elif keycode in (ord('r'), ord('R')):
                self.reset()
                print("\n[RESET]")

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_cb)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -15
        self.viewer.cam.azimuth = 180

    # ---- balance activo -----------------------------------------------------
    def balance_targets(self):
        pd = IDLE.copy()
        quat = self.data.qpos[3:7]
        roll, pitch, _ = quat_to_euler(quat)

        if self._has_gyro:
            ang_vel = self.data.sensor("imu_ang_vel").data
        else:
            ang_vel = self.data.qvel[3:6]

        roll_corr = -(KP_ROLL * roll + KD_ROLL * ang_vel[0])
        pitch_corr = -(KP_PITCH * pitch + KD_PITCH * ang_vel[1])

        # tobillos: la primera linea de defensa
        pd[L_ANK_PITCH] -= pitch_corr
        pd[R_ANK_PITCH] -= pitch_corr
        pd[L_ANK_ROLL] += roll_corr
        pd[R_ANK_ROLL] += roll_corr

        # caderas: apoyo (mismo eje que la correccion correspondiente)
        pd[L_HIP_PITCH] -= pitch_corr * HIP_PITCH_GAIN
        pd[R_HIP_PITCH] -= pitch_corr * HIP_PITCH_GAIN
        pd[L_HIP_ROLL] += roll_corr * HIP_ROLL_GAIN
        pd[R_HIP_ROLL] += roll_corr * HIP_ROLL_GAIN
        return pd

    # ---- brazos -------------------------------------------------------------
    def update_arms(self):
        raw = self.tracker.get() if self.teleop_on else None
        if raw is None:
            desired = IDLE[14:24]
        else:
            desired = raw.copy()
            desired[0:5] = np.clip(desired[0:5], ARM_LIMITS_L[:, 0], ARM_LIMITS_L[:, 1])
            desired[5:10] = np.clip(desired[5:10], ARM_LIMITS_R[:, 0], ARM_LIMITS_R[:, 1])

        # suavizado exponencial + limite de velocidad (evita tirones que tumban)
        smoothed = (1 - ARM_EMA) * self.arm_target + ARM_EMA * desired
        delta = np.clip(smoothed - self.arm_target, -ARM_MAX_STEP, ARM_MAX_STEP)
        self.arm_target = self.arm_target + delta
        return self.arm_target

    # ---- torque -------------------------------------------------------------
    def apply_torque(self, target):
        q = self.data.qpos[7:7 + NUM_DOFS]
        qd = self.data.qvel[6:6 + NUM_DOFS]
        tau = STIFFNESS * (target - q) - DAMPING * qd
        self.data.ctrl[:NUM_DOFS] = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

    # ---- loop ---------------------------------------------------------------
    def run(self):
        print("\n== TELEOP R1 ==")
        print("T: teleop ON/OFF | M: espejo | R: reset | ESC: salir\n")
        decim = int(round((1.0 / ARM_UPDATE_HZ) / DT))
        target = IDLE.copy()
        step = 0
        try:
            while self.viewer.is_running() and not self.should_exit:
                t0 = time.time()

                target = self.balance_targets()
                if step % decim == 0:
                    self._arms = self.update_arms()
                target[14:24] = self._arms

                self.apply_torque(target)
                mujoco.mj_step(self.model, self.data)

                self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()

                step += 1
                dt = DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        except KeyboardInterrupt:
            print("\nInterrumpido")
        finally:
            self.tracker.stop()
            try:
                if self.viewer.is_running():
                    self.viewer.close()
            except Exception:
                pass
            print("\nFin")


if __name__ == "__main__":
    env = R1Teleop()
    env._arms = IDLE[14:24].copy()
    env.run()