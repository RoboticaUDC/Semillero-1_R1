#!/usr/bin/env python3
"""
calibrar_brazos.py — Descubre el sentido real de cada joint de brazo del R1.

No hay fisica: solo cinematica. El robot flota y tu mueves un joint a la vez
para ver hacia donde se dobla. Con eso ajustamos el mapeo de la teleoperacion.

Teclas (en la ventana de MuJoCo):
    N / P      : siguiente / anterior joint
    FLECHA ARRIBA / ABAJO : +0.1 / -0.1 rad al joint actual
    0          : poner el joint actual en 0
    I          : volver toda la pose a IDLE
    Z          : poner TODOS los brazos en 0 (pose de referencia)
    ESC        : salir

QUE REPORTAR:
  Para cada joint, ponlo en +1.0 y dime hacia donde se mueve el brazo.
  Ej: "left_elbow_joint a +1.0 -> el antebrazo va hacia adelante/atras/arriba"
"""

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

from amo.paths import scene

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = scene("r1")
NUM_DOFS = 24

IDLE = np.array([
    -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,
    -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,
     0.0, 0.0,
     0.18, 0.18, 0.0, 1.5, 0.0,
     0.18,-0.18, 0.0, 1.5, 0.0,
], dtype=np.float64)

ARM_NAMES = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll",
]
ARM_START = 14   # indice del primer joint de brazo (orden MuJoCo)


class Calibrador:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.pose = IDLE.copy()
        self.idx = 0          # 0..9 dentro de los brazos
        self.should_exit = False
        self._apply()
        self._viewer()

    def _apply(self):
        self.data.qpos[0:3] = [0.0, 0.0, 1.0]        # flotando
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:7 + NUM_DOFS] = self.pose
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)     # solo cinematica

    def _status(self):
        j = ARM_START + self.idx
        print(f"\r[{self.idx}] {ARM_NAMES[self.idx]:24s} = {self.pose[j]:+.2f} rad "
              f"({np.degrees(self.pose[j]):+6.1f} deg)      ", end="", flush=True)

    def _viewer(self):
        def key_cb(k):
            j = ARM_START + self.idx
            if k == 256:
                self.should_exit = True
            elif k in (ord('n'), ord('N')):
                self.idx = (self.idx + 1) % len(ARM_NAMES)
            elif k in (ord('p'), ord('P')):
                self.idx = (self.idx - 1) % len(ARM_NAMES)
            elif k == 265:      # flecha arriba
                self.pose[j] += 0.1
            elif k == 264:      # flecha abajo
                self.pose[j] -= 0.1
            elif k == ord('0'):
                self.pose[j] = 0.0
            elif k in (ord('i'), ord('I')):
                self.pose = IDLE.copy()
            elif k in (ord('z'), ord('Z')):
                self.pose[ARM_START:ARM_START + 10] = 0.0
            else:
                return
            self._apply()
            self._status()

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_cb)
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -10
        self.viewer.cam.azimuth = 135   # vista 3/4 para ver adelante/atras

    def run(self):
        print(__doc__)
        print("Empieza con Z (todos los brazos a 0) y luego sube un joint a +1.0\n")
        self._status()
        try:
            while self.viewer.is_running() and not self.should_exit:
                self.viewer.sync()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.viewer.close()
            except Exception:
                pass
            print("\nFin")


if __name__ == "__main__":
    Calibrador().run()