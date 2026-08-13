#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

from amo.control import ArmController
from amo.math_utils import quat_to_euler
from amo.paths import scene

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import time
import numpy as np
import mujoco
import mujoco.viewer



# =============================================================================
# KEYS
# =============================================================================

KEY_ESCAPE = 256
KEY_F1     = 290
KEY_F6     = 295


# =============================================================================
# SCENE
# =============================================================================

PATH_SCENE = scene("g1")


# =============================================================================
# UTIL
# =============================================================================

# =============================================================================
# VIEWER STATE
# =============================================================================

class ViewerState:
    def __init__(self):
        self.should_exit = False


# =============================================================================
# ENV
# =============================================================================

class HumanoidEnv:

    NUM_DOFS = 23
    SIM_DT   = 0.002

    BALANCE_KP_ROLL  = 0.9
    BALANCE_KP_PITCH = 1.2

    BALANCE_KD_ROLL  = 0.05
    BALANCE_KD_PITCH = 0.08

    STIFFNESS = np.array([
        150,150,150,300,80,20,
        150,150,150,300,80,20,
        400,400,400,
        80,80,40,60,
        80,80,40,60,
    ], dtype=np.float32)

    DAMPING = np.array([
        2,2,2,4,2,1,
        2,2,2,4,2,1,
        15,15,15,
        2,2,1,1,
        2,2,1,1,
    ], dtype=np.float32)

    TORQUE_LIMITS = np.array([
        88,139,88,139,50,50,
        88,139,88,139,50,50,
        88,50,50,
        25,25,25,25,
        25,25,25,25,
    ], dtype=np.float32)

    IDLE_DOF_POS = np.array([
        -0.02,0.0,-0.02,0.18,-0.065,0.0,
         0.02,0.0,-0.02,0.18,-0.065,0.0,
         0.0,-0.065,0.0,
         0.2, 0.2,0.0,1.28,
         0.2,-0.2,0.0,1.28,
    ], dtype=np.float32)

    def __init__(self):

        # ==========================================================
        # MuJoCo
        # ==========================================================

        print("📁 Cargando escena...")
        self.model = mujoco.MjModel.from_xml_path(PATH_SCENE)
        self.model.opt.timestep = self.SIM_DT

        self.data = mujoco.MjData(self.model)

        # ==========================================================
        # Estado
        # ==========================================================

        self.state = ViewerState()

        # ==========================================================
        # Arm Controller
        # ==========================================================

        self.arm_ctrl = ArmController(self.model, self.data)

        # ==========================================================
        # Viewer
        # ==========================================================

        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=self.key_callback
        )

        self.viewer.cam.distance  = self.model.stat.extent * 1.5
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth   = 180

        print("\n===================================")
        print("🤖 G1 ESTABLE + SALUDO")
        print("===================================")
        print("F1  -> Saludar")
        print("F6  -> Neutral")
        print("ESC -> Salir")
        print("===================================\n")

    # =============================================================================
    # KEYBOARD
    # =============================================================================

    def key_callback(self, keycode):

        if keycode == KEY_ESCAPE:
            self.state.should_exit = True

        elif keycode == KEY_F1:
            self.arm_ctrl.play("wave")
            print("🤚 Saludando...")

        elif keycode == KEY_F6:
            self.arm_ctrl.play("neutral")
            print("😐 Neutral")

    # =============================================================================
    # STATE
    # =============================================================================

    def read_state(self):

        self.dof_pos = self.data.qpos[-self.NUM_DOFS:].astype(np.float32)

        self.dof_vel = self.data.qvel[-self.NUM_DOFS:].astype(np.float32)

        self.quat = self.data.sensor("orientation").data.astype(np.float32)

        self.ang_vel = self.data.sensor("angular-velocity").data.astype(np.float32)

    # =============================================================================
    # BALANCE
    # =============================================================================

    def active_stand_control(self):

        pd_target = self.IDLE_DOF_POS.copy()

        rpy = quat_to_euler(self.quat)

        roll  = rpy[0]
        pitch = rpy[1]

        roll_rate  = self.ang_vel[0]
        pitch_rate = self.ang_vel[1]

        roll_corr = (
            -roll * self.BALANCE_KP_ROLL
            -roll_rate * self.BALANCE_KD_ROLL
        )

        pitch_corr = (
            -pitch * self.BALANCE_KP_PITCH
            -pitch_rate * self.BALANCE_KD_PITCH
        )

        # Ankles
        pd_target[4]  -= pitch_corr
        pd_target[10] -= pitch_corr

        pd_target[5]  += roll_corr
        pd_target[11] += roll_corr

        # Hips compensation
        pd_target[0] -= roll_corr * 0.3
        pd_target[6] -= roll_corr * 0.3

        pd_target[2] += pitch_corr * 0.2
        pd_target[8] += pitch_corr * 0.2

        return pd_target

    # =============================================================================
    # PD CONTROL
    # =============================================================================

    def compute_torque(self, pd_target):

        torque = (
            (pd_target - self.dof_pos) * self.STIFFNESS
            - self.dof_vel * self.DAMPING
        )

        return np.clip(
            torque,
            -self.TORQUE_LIMITS,
            self.TORQUE_LIMITS
        )

    # =============================================================================
    # MAIN LOOP
    # =============================================================================

    def run(self):

        print("🚀 Simulación iniciada")

        try:

            while self.viewer.is_running() and not self.state.should_exit:

                t0 = time.time()

                self.read_state()

                # ======================================================
                # Balance estable
                # ======================================================

                pd_target = self.active_stand_control()

                # ======================================================
                # Arm controller
                # ======================================================

                self.arm_ctrl.update()

                current = self.dof_pos[15:]

                alpha = 0.8

                pd_target[15:] = (

                    (1 - alpha) * current

                    + alpha * self.arm_ctrl.target_pose
                )

                # ======================================================
                # Torque control
                # ======================================================

                torque = self.compute_torque(pd_target)

                for j in range(self.NUM_DOFS):
                    self.data.ctrl[j] = torque[j]

                # ======================================================
                # Step
                # ======================================================

                mujoco.mj_step(self.model, self.data)

                self.viewer.cam.lookat[:] = self.data.qpos[:3]

                self.viewer.sync()

                remaining = self.SIM_DT - (time.time() - t0)

                if remaining > 0:
                    time.sleep(remaining)

        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido")

        finally:

            try:
                if self.viewer.is_running():
                    self.viewer.close()
            except:
                pass

            print("✅ Finalizado")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    env = HumanoidEnv()

    env.run()
