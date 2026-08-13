#!/usr/bin/env python3
"""
play_r1_camina_brazos.py — R1 caminando con la politica NATIVA de Isaac
+ gestos de brazo (saludar, apuntar, etc.) con el ArmController.

DISENO:
  - Piernas + cintura (indices MuJoCo 0-13): las controla la POLITICA.
  - Brazos (indices MuJoCo 14-23): por defecto los controla la politica;
    cuando activas un gesto (F1-F8), el ArmController toma el control de
    los brazos durante la secuencia, con una transicion suave (blend) al
    entrar y salir para que no peguen tirones.

TECLAS:
  Flechas   : vx (arriba/abajo), giro (izq/der)
  Q / E     : vy lateral
  ESPACIO   : parar (comando = 0)
  R         : reset
  F1        : saludar 👋      F2 : apuntar 👉     F3 : cargar 📦
  F4        : cruz ✚          F5 : guardia 🥊     F6 : neutral
  F7        : pausar/reanudar gesto
  ESC       : salir

Requisitos: r1.xml, r1_policy_v2.pt, ArmController.py en la misma carpeta.
"""

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

from amo.control import ArmController
from amo.math_utils import quat_rotate_inverse
from amo.paths import policy, scene

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import time
from collections import deque

import numpy as np
import mujoco
import mujoco.viewer
import torch


# =============================================================================
# RUTAS
# =============================================================================
XML_PATH    = scene("r1")
POLICY_PATH = policy("r1_v2")   # tu politica mejorada

# =============================================================================
# MAPEO DE JOINTS  (array de oro verificado)
# =============================================================================
MUJOCO_FROM_ISAAC = np.array(
    [0, 3, 6, 10, 14, 18, 1, 4, 7, 11, 15, 19, 2, 5, 8, 12, 16, 20, 22, 9, 13, 17, 21, 23]
)
ISAAC_FROM_MUJOCO = np.argsort(MUJOCO_FROM_ISAAC)
NUM_DOFS = 24

# Indices MuJoCo de los brazos (14-23): 5 izq + 5 der
ARM_IDX = np.arange(14, 24)

# =============================================================================
# POSE POR DEFECTO (orden MuJoCo)
# =============================================================================
DEFAULT_MJ = np.array([
    -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,
    -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,
     0.0, 0.0,
     0.18, 0.18, 0.0, 1.5, 0.0,
     0.18,-0.18, 0.0, 1.5, 0.0,
], dtype=np.float32)
DEFAULT_ISAAC = DEFAULT_MJ[ISAAC_FROM_MUJOCO]

# =============================================================================
# GANANCIAS PD (orden MuJoCo)
# =============================================================================
STIFFNESS_MJ = np.array([
    100, 100, 100, 200, 80, 80,
    100, 100, 100, 200, 80, 80,
    250, 250,
    50, 50, 40, 30, 20,
    50, 50, 40, 30, 20,
], dtype=np.float32)
DAMPING_MJ = np.array([
    5, 5, 5, 8, 5, 5,
    5, 5, 5, 8, 5, 5,
    25, 25,
    5, 5, 4, 4, 4,
    5, 5, 4, 4, 4,
], dtype=np.float32)
TORQUE_LIMITS_MJ = np.array([
    88, 139, 88, 139, 50, 50,
    88, 139, 88, 139, 50, 50,
    88, 50,
    25, 25, 25, 25, 25,
    25, 25, 25, 25, 25,
], dtype=np.float32)

# =============================================================================
# CONTROL
# =============================================================================
SIM_DT       = 0.002
DECIMATION   = 10           # control a 50 Hz
ACTION_SCALE = 0.25
HISTORY_LEN  = 5
SCALE_ANG_VEL   = 0.2
SCALE_JOINT_VEL = 0.05

# Blend de brazos: cuanto sube/baja por control-step (50 Hz).
# 0.04 => ~0.5 s de transicion suave al entrar/salir de un gesto.
ARM_BLEND_STEP = 0.04


class R1CaminaBrazos:
    def __init__(self, xml_path=XML_PATH, policy_path=POLICY_PATH, device="cpu"):
        self.device = device
        print(f"Cargando modelo: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)

        print(f"Cargando politica: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        # ArmController (usamos solo su target_pose de 10 valores)
        self.arm_ctrl = ArmController(self.model, self.data)
        # arm_blend = 0 -> brazos los manda la politica
        # arm_blend = 1 -> brazos los manda el ArmController (gesto)
        self.arm_blend = 0.0

        self.command = np.zeros(3, dtype=np.float32)
        self.last_action_isaac = np.zeros(NUM_DOFS, dtype=np.float32)

        self.h_ang_vel = deque([np.zeros(3, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_grav    = deque([np.zeros(3, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_cmd     = deque([np.zeros(3, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_jpos    = deque([np.zeros(NUM_DOFS, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_jvel    = deque([np.zeros(NUM_DOFS, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_act     = deque([np.zeros(NUM_DOFS, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)

        try:
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
            self._has_gyro = True
        except Exception:
            self._has_gyro = False

        self.reset()
        self._setup_viewer()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.74]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:7 + NUM_DOFS] = DEFAULT_MJ
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_action_isaac[:] = 0.0
        for dq in (self.h_ang_vel, self.h_grav, self.h_cmd, self.h_jpos, self.h_jvel, self.h_act):
            for i in range(HISTORY_LEN):
                dq[i][:] = 0.0

    def _setup_viewer(self):
        self.should_exit = False
        KEY = {"F1": 290, "F2": 291, "F3": 292, "F4": 293, "F5": 294, "F6": 295, "F7": 296}

        def key_cb(keycode):
            if keycode == 256:
                self.should_exit = True
            elif keycode == 265:
                self.command[0] += 0.1
            elif keycode == 264:
                self.command[0] -= 0.1
            elif keycode == 263:
                self.command[2] += 0.1
            elif keycode == 262:
                self.command[2] -= 0.1
            elif keycode in (ord('q'), ord('Q')):
                self.command[1] += 0.1
            elif keycode in (ord('e'), ord('E')):
                self.command[1] -= 0.1
            elif keycode == 32:
                self.command[:] = 0.0
            elif keycode in (ord('r'), ord('R')):
                self.reset()
            elif keycode == KEY["F1"]:
                self.arm_ctrl.play("wave");    print("\n👋 Saludando")
            elif keycode == KEY["F2"]:
                self.arm_ctrl.play("point");   print("\n👉 Apuntando")
            elif keycode == KEY["F3"]:
                self.arm_ctrl.play("carry");   print("\n📦 Cargando")
            elif keycode == KEY["F4"]:
                self.arm_ctrl.play("cross");   print("\n✚ Cruz")
            elif keycode == KEY["F5"]:
                self.arm_ctrl.play("guard");   print("\n🥊 Guardia")
            elif keycode == KEY["F6"]:
                self.arm_ctrl.play("neutral"); print("\n😐 Neutral")
            elif keycode == KEY["F7"]:
                self.arm_ctrl.pause_toggle()
            self.command[:] = np.clip(self.command, -1.0, 1.0)

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_cb)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 180

    def _read_state(self):
        qpos_mj = self.data.qpos[7:7 + NUM_DOFS].astype(np.float32)
        qvel_mj = self.data.qvel[6:6 + NUM_DOFS].astype(np.float32)
        quat = self.data.qpos[3:7].astype(np.float32)
        if self._has_gyro:
            ang_vel = self.data.sensor("imu_ang_vel").data.astype(np.float32)
        else:
            ang_vel = self.data.qvel[3:6].astype(np.float32)
        return qpos_mj, qvel_mj, quat, ang_vel

    def _build_obs(self):
        qpos_mj, qvel_mj, quat, ang_vel = self._read_state()
        ang_vel_obs = ang_vel * SCALE_ANG_VEL
        grav_obs = quat_rotate_inverse(quat, np.array([0, 0, -1], np.float32))
        cmd_obs = self.command.copy()
        qpos_obs = qpos_mj.copy()
        qpos_obs[ARM_IDX] = DEFAULT_MJ[ARM_IDX]   # brazos "congelados" en la obs
        jpos_isaac = qpos_obs[ISAAC_FROM_MUJOCO] - DEFAULT_ISAAC
        qvel_obs = qvel_mj.copy()
        qvel_obs[ARM_IDX] = 0.0   # velocidad de brazos = 0 en la obs
        jvel_isaac = qvel_obs[ISAAC_FROM_MUJOCO] * SCALE_JOINT_VEL
        act_obs = self.last_action_isaac.copy()

        self.h_ang_vel.append(ang_vel_obs.astype(np.float32))
        self.h_grav.append(grav_obs.astype(np.float32))
        self.h_cmd.append(cmd_obs.astype(np.float32))
        self.h_jpos.append(jpos_isaac.astype(np.float32))
        self.h_jvel.append(jvel_isaac.astype(np.float32))
        self.h_act.append(act_obs.astype(np.float32))

        def flat(dq):
            return np.concatenate(list(dq))

        return np.concatenate([
            flat(self.h_ang_vel), flat(self.h_grav), flat(self.h_cmd),
            flat(self.h_jpos), flat(self.h_jvel), flat(self.h_act),
        ]).astype(np.float32)

    def _policy_step(self):
        obs = self._build_obs()
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_isaac = self.policy(obs_t).cpu().numpy().squeeze().astype(np.float32)
        self.last_action_isaac = action_isaac.copy()
        target_isaac = DEFAULT_ISAAC + ACTION_SCALE * action_isaac
        target_mj = target_isaac[MUJOCO_FROM_ISAAC]

        # --- brazos: fusionar politica con ArmController segun blend ---
        self.arm_ctrl.update()  # avanza la secuencia si hay gesto activo
        gesto_activo = (self.arm_ctrl._active_seq is not None and self.arm_ctrl._active_seq.active)
        # subir/bajar el blend suavemente
        if gesto_activo:
            self.arm_blend = min(1.0, self.arm_blend + ARM_BLEND_STEP)
        else:
            self.arm_blend = max(0.0, self.arm_blend - ARM_BLEND_STEP)

        if self.arm_blend > 0.0:
            arm_target = self.arm_ctrl.target_pose.astype(np.float32)  # 10 valores
            pol_arm = target_mj[ARM_IDX]
            target_mj[ARM_IDX] = (1.0 - self.arm_blend) * pol_arm + self.arm_blend * arm_target

        return target_mj

    def _apply_pd(self, target_mj):
        qpos_mj = self.data.qpos[7:7 + NUM_DOFS]
        qvel_mj = self.data.qvel[6:6 + NUM_DOFS]
        torque = STIFFNESS_MJ * (target_mj - qpos_mj) - DAMPING_MJ * qvel_mj
        torque = np.clip(torque, -TORQUE_LIMITS_MJ, TORQUE_LIMITS_MJ)
        self.data.ctrl[:NUM_DOFS] = torque

    def run(self):
        print("\n== R1 camina + brazos ==")
        print("Flechas: mover | Q/E: lateral | ESPACIO: parar | R: reset")
        print("F1 saludar | F2 apuntar | F3 cargar | F4 cruz | F5 guardia | F6 neutral | F7 pausa | ESC salir\n")
        target_mj = DEFAULT_MJ.copy()
        step = 0
        try:
            while self.viewer.is_running() and not self.should_exit:
                t0 = time.time()
                if step % DECIMATION == 0:
                    target_mj = self._policy_step()
                self._apply_pd(target_mj)
                mujoco.mj_step(self.model, self.data)
                self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()
                step += 1
                dt = SIM_DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        except KeyboardInterrupt:
            print("\nInterrumpido")
        finally:
            try:
                if self.viewer.is_running():
                    self.viewer.close()
            except Exception:
                pass
            print("\nFin")


if __name__ == "__main__":
    device = "cpu"   # el MLP es minusculo, CPU sobra y evita el tema CUDA sm_120
    print(f"Dispositivo: {device}")
    env = R1CaminaBrazos(device=device)
    env.run()