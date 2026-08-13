
# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

from amo.paths import scene
# -----------------------------------------------------------------------------
# play_amo_stable.py
# Control PD puro para mantener al G1 de pie de forma estable.
# NO usa la policy de IA todavía — solo control de posición.
# Una vez que el robot se mantenga parado, se puede re-habilitar la IA.
# -----------------------------------------------------------------------------

import numpy as np
import mujoco
import mujoco_viewer
import glfw
from collections import deque
import math

# ── Pose base de pie ────────────────────────────────────────────────────────
# Orden: qpos[7..29] → 23 joints articulados
# (piernas x2, cintura x3, brazos x2)
BASE_POSE = np.array([
    # Pierna izquierda
    -0.35,   # left_hip_pitch
     0.12,   # left_hip_roll
     0.00,   # left_hip_yaw
     0.50,   # left_knee
    -0.25,   # left_ankle_pitch
    -0.06,   # left_ankle_roll
    # Pierna derecha
    -0.35,   # right_hip_pitch
    -0.12,   # right_hip_roll
     0.00,   # right_hip_yaw
     0.50,   # right_knee
    -0.25,   # right_ankle_pitch
     0.06,   # right_ankle_roll
    # Cintura
     0.00,   # waist_yaw
     0.00,   # waist_roll
     0.00,   # waist_pitch
    # Brazo izquierdo
     0.30,   # left_shoulder_pitch
     0.30,   # left_shoulder_roll
     0.00,   # left_shoulder_yaw
     0.50,   # left_elbow
    # Brazo derecho
     0.30,   # right_shoulder_pitch
    -0.30,   # right_shoulder_roll
     0.00,   # right_shoulder_yaw
     0.50,   # right_elbow
], dtype=np.float32)

# ── Ganancias PD ─────────────────────────────────────────────────────────────
# Piernas necesitan más fuerza que brazos
KP = np.array([
    200, 150, 150, 300, 100, 60,   # pierna izq: hip(3) knee ankle(2)
    200, 150, 150, 300, 100, 60,   # pierna der
    150, 100, 100,                 # cintura
     80,  60,  60,  80,            # brazo izq
     80,  60,  60,  80,            # brazo der
], dtype=np.float32)

KD = np.array([
      5,   4,   4,   6,   3,  2,   # pierna izq
      5,   4,   4,   6,   3,  2,   # pierna der
      4,   3,   3,              # cintura
      2,   2,   2,  2,          # brazo izq
      2,   2,   2,  2,          # brazo der
], dtype=np.float32)

# ── Estado global de comandos ────────────────────────────────────────────────
commands = {
    'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0,
    'height': 0.0,
}

# Fase de marcha para caminar
walk_phase = 0.0
walking = False
step_size = 0.12    # amplitud del paso (rad)
walk_freq = 1.2     # Hz

def key_callback(window, key, scancode, action, mods):
    global commands, walking

    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    s = 0.05
    if   key == glfw.KEY_W: commands['vx']  = min(commands['vx']  + s,  0.5)
    elif key == glfw.KEY_S: commands['vx']  = max(commands['vx']  - s, -0.5)
    elif key == glfw.KEY_A: commands['vy']  = min(commands['vy']  + s,  0.4)
    elif key == glfw.KEY_D: commands['vy']  = max(commands['vy']  - s, -0.4)
    elif key == glfw.KEY_Q: commands['vyaw']= min(commands['vyaw']+ s,  1.0)
    elif key == glfw.KEY_E: commands['vyaw']= max(commands['vyaw']- s, -1.0)
    elif key == glfw.KEY_Z: commands['height'] = min(commands['height'] + 0.02, 0.15)
    elif key == glfw.KEY_X: commands['height'] = max(commands['height'] - 0.02,-0.10)
    elif key == glfw.KEY_SPACE:
        commands = {k: 0.0 for k in commands}
        print("\n[STOP] Comandos a cero")

    walking = abs(commands['vx']) > 0.01 or abs(commands['vy']) > 0.01

    print(f"\r[CMD] vx={commands['vx']:+.2f} vy={commands['vy']:+.2f} "
          f"vyaw={commands['vyaw']:+.2f} height={commands['height']:+.2f}  ",
          end="", flush=True)


def get_target_pose(t, dt):
    """
    Calcula la pose objetivo en función del tiempo y comandos.
    Cuando hay comando de marcha, modula los joints de piernas
    con un patrón de paso simple.
    """
    global walk_phase, walking

    target = BASE_POSE.copy()

    # Altura del torso: ajusta hip pitch y knee
    h = commands['height']
    target[0]  += h * 0.6   # left_hip_pitch
    target[6]  += h * 0.6   # right_hip_pitch
    target[3]  -= h * 1.2   # left_knee
    target[9]  -= h * 1.2   # right_knee
    target[4]  -= h * 0.6   # left_ankle_pitch
    target[10] -= h * 0.6   # right_ankle_pitch

    # Patrón de marcha simple: alternancia de piernas
    if walking:
        walk_phase += 2.0 * math.pi * walk_freq * dt

        # Amplitud escala con velocidad
        amp = min(abs(commands['vx']) + abs(commands['vy']), 0.5) * step_size / 0.5

        # Pierna izquierda: fase normal
        sl = math.sin(walk_phase)
        # Pierna derecha: fase opuesta
        sr = math.sin(walk_phase + math.pi)

        # Hip pitch (avance de pierna)
        target[0]  += sl * amp        # left_hip_pitch
        target[6]  += sr * amp        # right_hip_pitch
        # Knee (dobla al levantar)
        target[3]  += max(0, -sl) * amp * 0.8   # left_knee
        target[9]  += max(0, -sr) * amp * 0.8   # right_knee
        # Ankle (compensa)
        target[4]  -= sl * amp * 0.4  # left_ankle_pitch
        target[10] -= sr * amp * 0.4  # right_ankle_pitch

        # Yaw (girar): diferencia de hip yaw entre piernas
        yaw_amp = commands['vyaw'] * 0.1
        target[2]  += yaw_amp   # left_hip_yaw
        target[8]  -= yaw_amp   # right_hip_yaw

        # Lateral (vy): hip roll
        vy_amp = commands['vy'] * 0.15
        target[1]  += vy_amp    # left_hip_roll
        target[7]  += vy_amp    # right_hip_roll

    return target


class HumanoidEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(scene("g1"))
        self.data  = mujoco.MjData(self.model)
        self.dt    = self.model.opt.timestep

        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        glfw.set_key_callback(self.viewer.window, key_callback)

        # Suavizado de comandos
        self.cmd_pos = BASE_POSE.copy()

        self.reset()

        print(f"\n[INFO] dt={self.dt*1000:.1f}ms  nu={self.model.nu}")
        print("[CONTROLES]")
        print("  W/S = adelante/atrás    A/D = lateral")
        print("  Q/E = girar             Z/X = altura")
        print("  SPACE = parar\n")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.793     # altura correcta
        self.data.qpos[3] = 1.0       # quaternion w (vertical)
        self.data.qpos[4:7] = 0.0
        self.data.qpos[7:30] = BASE_POSE
        self.cmd_pos = BASE_POSE.copy()
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        # Pose objetivo según comandos
        target = get_target_pose(self.data.time, self.dt)

        # Suavizado exponencial (evita saltos bruscos)
        alpha = 0.08
        self.cmd_pos += alpha * (target - self.cmd_pos)

        # Posición y velocidad actual de joints articulados
        q  = self.data.qpos[7:30].astype(np.float32)
        qd = self.data.qvel[6:29].astype(np.float32)

        # Control PD
        error    = self.cmd_pos - q
        torque   = KP * error - KD * qd
        torque   = np.clip(torque, -60, 60)

        self.data.ctrl[:] = torque

        mujoco.mj_step(self.model, self.data)
        self.viewer.render()

    def run(self):
        self.reset()
        while not glfw.window_should_close(self.viewer.window):
            self.step()
        self.viewer.close()


if __name__ == "__main__":
    env = HumanoidEnv()
    env.run()