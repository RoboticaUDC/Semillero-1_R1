#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

from amo.math_utils import quat_to_euler
from amo.paths import scene

import time
import numpy as np
import mujoco
import mujoco.viewer
import os

os.environ.setdefault("MUJOCO_GL", "glfw")


# ==========================================================
# UTIL
# ==========================================================

# ==========================================================
# ENV
# ==========================================================

class R1StableEnv:

    NUM_DOFS = 24
    DT = 0.002

    # =========================
    # Gains (tipo G1 pero ajustados)
    # =========================
    KP_ROLL  = 1.0
    KP_PITCH = 2
    KD_ROLL  = 0.06
    KD_PITCH = 0.10

    # =========================
    # PD gains por joint
    # =========================
    STIFFNESS = np.array([
        100,100,100,200,80,20,
        100,100,100,200,80,20,
        250,250,
        50,50,40,30,20,
        50,50,40,30,20
    ], dtype=np.float32)

    DAMPING = np.array([
        5,5,5,8,5,5,
        5,5,5,8,5,5,
        25,25,
        5,5,4,4,4,
        5,5,4,4,4
    ], dtype=np.float32)

    TORQUE_LIMITS = np.array([
        88,139,88,139,50,50,
        88,139,88,139,50,50,
        88,50,
        25,25,25,25,25,
        25,25,25,25,25
    ], dtype=np.float32)

    # =========================
    # Pose estable (base tipo G1 pero adaptada R1)
    # =========================
    IDLE = np.array([
        -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,
        -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,
         0.0, 0.0,
         0.18, 0.18, 0.0, 1.5, 0.0,
         0.18,-0.18, 0.0, 1.5, 0.0
    ], dtype=np.float32)

    # ==========================================================
    # INIT
    # ==========================================================

    def __init__(self, xml_path=None):
        xml_path = xml_path or scene("r1")

        print("📁 Cargando R1...")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = self.DT
        self.data = mujoco.MjData(self.model)

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 180

        print("🤖 R1 Stable Controller listo")

    # ==========================================================
    # STATE
    # ==========================================================

    def read_state(self):

        self.dof_pos = self.data.qpos[7:7+self.NUM_DOFS]
        self.dof_vel = self.data.qvel[6:6+self.NUM_DOFS]

        self.quat = self.data.qpos[3:7]

        # fallback IMU
        try:
            self.ang_vel = self.data.sensor("imu_ang_vel").data
        except:
            self.ang_vel = self.data.qvel[3:6]

    # ==========================================================
    # BALANCE (G1-style core)
    # ==========================================================

    def balance_control(self):

        pd = self.IDLE.copy()

        roll, pitch, yaw = quat_to_euler(self.quat)

        roll_rate = self.ang_vel[0]
        pitch_rate = self.ang_vel[1]

        roll_corr = -(self.KP_ROLL * roll + self.KD_ROLL * roll_rate)
        pitch_corr = -(self.KP_PITCH * pitch + self.KD_PITCH * pitch_rate)

        # -------------------------
        # ankles correction
        # -------------------------
        #tobillos
        pd[4]  -= pitch_corr
        pd[10] -= pitch_corr
        #laterales
        pd[5]  += roll_corr
        pd[11] += roll_corr
        #laterales
        # hips stabilization
        pd[0] -= roll_corr * 0.25
        pd[6] -= roll_corr * 0.25
        #caderas
        pd[2] += pitch_corr * 0.15
        pd[8] += pitch_corr * 0.15

        return pd

    # ==========================================================
    # TORQUE
    # ==========================================================

    def compute_torque(self, target):

        torque = (target - self.dof_pos) * self.STIFFNESS \
                 - self.dof_vel * self.DAMPING

        return np.clip(torque, -self.TORQUE_LIMITS, self.TORQUE_LIMITS)

    # ==========================================================
    # LOOP
    # ==========================================================

    def run(self):

        print("🚀 R1 estable iniciado (tipo G1)")

        try:
            while self.viewer.is_running():

                t0 = time.time()

                self.read_state()

                # -------------------------
                # core stability
                # -------------------------
                pd_target = self.balance_control()

                # -------------------------
                # apply torque
                # -------------------------
                torque = self.compute_torque(pd_target)

                self.data.ctrl[:self.NUM_DOFS] = torque

                mujoco.mj_step(self.model, self.data)

                self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()

                dt = self.DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

        except KeyboardInterrupt:
            print("⛔ detenido")

        finally:
            try:
                self.viewer.close()
            except:
                pass
            print("✅ terminado")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    env = R1StableEnv()
    env.run()