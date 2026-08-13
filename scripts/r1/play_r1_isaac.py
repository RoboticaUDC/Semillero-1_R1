#!/usr/bin/env python3
"""
play_r1_isaac.py — Corre la politica NATIVA del R1 (entrenada en Isaac Lab)e
dentro de MuJoCo, reemplazando el parche del adapter de G1.

Reconstruye exactamente la observacion que Isaac le daba a la politica:
  obs (405) = [ base_ang_vel(15), projected_gravity(15), velocity_commands(15),
                joint_pos_rel(120), joint_vel_rel(120), last_action(120) ]
  - Todo en ORDEN DE JOINTS DE ISAAC.
  - Cada termino con history de 5 pasos, aplanado [t-4, t-3, t-2, t-1, t].
  - Escalas: base_ang_vel x0.2, joint_vel_rel x0.05, el resto x1.0.

La salida de la politica (24 acciones, orden Isaac) se convierte a target con
  target_isaac = default_isaac + 0.25 * accion_isaac
luego se REORDENA a MuJoCo con el "array de oro" y se aplica como PD.

Uso:
  python play_r1_isaac.py
Teclas:
  flechas  : vx (arriba/abajo), wz/giro (izq/der)
  Q / E    : vy lateral
  ESPACIO  : detener (comando = 0)
  R        : reset a la pose inicial
  ESC      : salir
"""

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

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
POLICY_PATH = policy("r1_v2")

# =============================================================================
# MAPEO DE JOINTS  (el "array de oro" verificado con check_r1.py)
# =============================================================================
# accion_mujoco = accion_isaac[MUJOCO_FROM_ISAAC]
MUJOCO_FROM_ISAAC = np.array(
    [0, 3, 6, 10, 14, 18, 1, 4, 7, 11, 15, 19, 2, 5, 8, 12, 16, 20, 22, 9, 13, 17, 21, 23]
)
# Inverso: vector_isaac = vector_mujoco[ISAAC_FROM_MUJOCO]
ISAAC_FROM_MUJOCO = np.argsort(MUJOCO_FROM_ISAAC)

NUM_DOFS = 24

# =============================================================================
# POSE POR DEFECTO  (en ORDEN MUJOCO, igual que el init_state del CFG)
# =============================================================================
# left leg(6), right leg(6), waist(2), left arm(5), right arm(5)
DEFAULT_MJ = np.array([
    -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,   # pierna izq
    -0.06, 0.0, 0.0, 0.05, -0.04, 0.0,   # pierna der
     0.0, 0.0,                           # cintura (roll, yaw)
     0.18, 0.18, 0.0, 1.5, 0.0,          # brazo izq
     0.18,-0.18, 0.0, 1.5, 0.0,          # brazo der
], dtype=np.float32)
DEFAULT_ISAAC = DEFAULT_MJ[ISAAC_FROM_MUJOCO]

# =============================================================================
# GANANCIAS PD  (en ORDEN MUJOCO, iguales a los actuadores del CFG de Isaac)
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
# PARAMETROS DE CONTROL  (para casar con Isaac: control a 50 Hz)
# =============================================================================
SIM_DT      = 0.002            # paso de fisica de MuJoCo
DECIMATION  = 10               # 0.002 * 10 = 0.02 s  ->  50 Hz de politica
ACTION_SCALE = 0.25            # del JointPositionActionCfg
HISTORY_LEN = 5

# Escalas de observacion (Isaac)
SCALE_ANG_VEL  = 0.2
SCALE_JOINT_VEL = 0.05


# =============================================================================
# ENTORNO
# =============================================================================
class R1IsaacPolicyEnv:
    def __init__(self, xml_path=XML_PATH, policy_path=POLICY_PATH, device="cpu"):
        self.device = device

        print(f"Cargando modelo: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)

        print(f"Cargando politica: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        # Comando [vx, vy, wz]
        self.command = np.zeros(3, dtype=np.float32)

        # last_action en ORDEN ISAAC (salida cruda de la red, paso anterior)
        self.last_action_isaac = np.zeros(NUM_DOFS, dtype=np.float32)

        # Buffers de history (deque por termino, lleno de ceros al inicio)
        # Cada entrada es el snapshot YA ESCALADO de ese paso.
        self.h_ang_vel = deque([np.zeros(3, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_grav    = deque([np.zeros(3, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_cmd     = deque([np.zeros(3, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_jpos    = deque([np.zeros(NUM_DOFS, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_jvel    = deque([np.zeros(NUM_DOFS, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)
        self.h_act     = deque([np.zeros(NUM_DOFS, np.float32)] * HISTORY_LEN, maxlen=HISTORY_LEN)

        # IMU gyro
        try:
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
            self._has_gyro = True
        except Exception:
            self._has_gyro = False

        self.reset()
        self._setup_viewer()

    # -------------------------------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # base flotante a 0.74 con quat identidad
        self.data.qpos[0:3] = [0.0, 0.0, 0.74]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:7 + NUM_DOFS] = DEFAULT_MJ
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.last_action_isaac[:] = 0.0
        for dq in (self.h_ang_vel, self.h_grav, self.h_cmd, self.h_jpos, self.h_jvel, self.h_act):
            for i in range(HISTORY_LEN):
                dq[i][:] = 0.0

    # -------------------------------------------------------------------------
    def _setup_viewer(self):
        self.should_exit = False

        def key_cb(keycode):
            if keycode == 256:  # ESC
                self.should_exit = True
            elif keycode == 265:  # arriba
                self.command[0] += 0.1
            elif keycode == 264:  # abajo
                self.command[0] -= 0.1
            elif keycode == 263:  # izquierda
                self.command[2] += 0.1
            elif keycode == 262:  # derecha
                self.command[2] -= 0.1
            elif keycode in (ord('q'), ord('Q')):
                self.command[1] += 0.1
            elif keycode in (ord('e'), ord('E')):
                self.command[1] -= 0.1
            elif keycode == 32:   # ESPACIO
                self.command[:] = 0.0
            elif keycode in (ord('r'), ord('R')):
                self.reset()
            self.command[:] = np.clip(self.command, -1.0, 1.0)
            print(f"\rComando  vx={self.command[0]:+.2f}  vy={self.command[1]:+.2f}  wz={self.command[2]:+.2f}   ",
                  end="", flush=True)

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=key_cb
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -20
        self.viewer.cam.azimuth = 180

    # -------------------------------------------------------------------------
    def _read_state(self):
        # joints en ORDEN MUJOCO
        qpos_mj = self.data.qpos[7:7 + NUM_DOFS].astype(np.float32)
        qvel_mj = self.data.qvel[6:6 + NUM_DOFS].astype(np.float32)
        quat = self.data.qpos[3:7].astype(np.float32)  # (w,x,y,z)

        if self._has_gyro:
            ang_vel = self.data.sensor("imu_ang_vel").data.astype(np.float32)
        else:
            ang_vel = self.data.qvel[3:6].astype(np.float32)

        return qpos_mj, qvel_mj, quat, ang_vel

    # -------------------------------------------------------------------------
    def _build_obs(self):
        qpos_mj, qvel_mj, quat, ang_vel = self._read_state()

        # --- snapshots del paso actual, ya en ORDEN ISAAC y escalados ---
        ang_vel_obs = ang_vel * SCALE_ANG_VEL                       # (3,)
        grav_obs = quat_rotate_inverse(quat, np.array([0, 0, -1], np.float32))  # (3,)
        cmd_obs = self.command.copy()                              # (3,)

        # joint_pos_rel = (q - default) en orden Isaac
        jpos_isaac = qpos_mj[ISAAC_FROM_MUJOCO] - DEFAULT_ISAAC
        # joint_vel_rel = qd en orden Isaac, escalado
        jvel_isaac = (qvel_mj[ISAAC_FROM_MUJOCO]) * SCALE_JOINT_VEL
        # last_action (orden Isaac, sin escala)
        act_obs = self.last_action_isaac.copy()

        # --- meter al history ---
        self.h_ang_vel.append(ang_vel_obs.astype(np.float32))
        self.h_grav.append(grav_obs.astype(np.float32))
        self.h_cmd.append(cmd_obs.astype(np.float32))
        self.h_jpos.append(jpos_isaac.astype(np.float32))
        self.h_jvel.append(jvel_isaac.astype(np.float32))
        self.h_act.append(act_obs.astype(np.float32))

        # --- aplanar cada termino [t-4, ..., t] y concatenar en el orden del grupo ---
        def flat(dq):
            return np.concatenate(list(dq))  # 5 bloques viejo->nuevo

        obs = np.concatenate([
            flat(self.h_ang_vel),   # 15
            flat(self.h_grav),      # 15
            flat(self.h_cmd),       # 15
            flat(self.h_jpos),      # 120
            flat(self.h_jvel),      # 120
            flat(self.h_act),       # 120
        ]).astype(np.float32)       # total 405

        return obs

    # -------------------------------------------------------------------------
    def _policy_step(self):
        obs = self._build_obs()
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_isaac = self.policy(obs_t).cpu().numpy().squeeze().astype(np.float32)

        # guardar para la obs del proximo paso (cruda, orden Isaac)
        self.last_action_isaac = action_isaac.copy()

        # target en orden Isaac -> orden MuJoCo
        target_isaac = DEFAULT_ISAAC + ACTION_SCALE * action_isaac
        target_mj = target_isaac[MUJOCO_FROM_ISAAC]
        return target_mj

    # -------------------------------------------------------------------------
    def _apply_pd(self, target_mj):
        qpos_mj = self.data.qpos[7:7 + NUM_DOFS]
        qvel_mj = self.data.qvel[6:6 + NUM_DOFS]
        torque = STIFFNESS_MJ * (target_mj - qpos_mj) - DAMPING_MJ * qvel_mj
        torque = np.clip(torque, -TORQUE_LIMITS_MJ, TORQUE_LIMITS_MJ)
        self.data.ctrl[:NUM_DOFS] = torque

    # -------------------------------------------------------------------------
    def run(self):
        print("\n== R1 con politica NATIVA de Isaac ==")
        print("Flechas: vx/giro | Q/E: lateral | ESPACIO: parar | R: reset | ESC: salir\n")
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
    device = "cpu"
    print(f"Dispositivo: {device}")
    env = R1IsaacPolicyEnv(device=device)
    env.run()