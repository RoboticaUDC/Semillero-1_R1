
# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

from amo.math_utils import quatToEuler
from amo.paths import policy, scene
# -----------------------------------------------------------------------------
# Copyright [2025] [Jialong Li, Xuxin Cheng, Tianshu Huang, Xiaolong Wang]
# Licensed under the Apache License, Version 2.0
# -----------------------------------------------------------------------------

import types
import numpy as np
import mujoco, mujoco_viewer
import glfw
from collections import deque
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


# ---------------------------------------------------------------------------
# Índices de joints (qposadr - 7 para acceder como joint relativo)
# pelvis ocupa qpos[0:7] (free joint: x,y,z + quaternion wxyz)
# joints articulados empiezan en qpos[7]
# ---------------------------------------------------------------------------
# qpos[7]  = left_hip_pitch
# qpos[8]  = left_hip_roll
# qpos[9]  = left_hip_yaw
# qpos[10] = left_knee
# qpos[11] = left_ankle_pitch
# qpos[12] = left_ankle_roll
# qpos[13] = right_hip_pitch
# qpos[14] = right_hip_roll
# qpos[15] = right_hip_yaw
# qpos[16] = right_knee
# qpos[17] = right_ankle_pitch
# qpos[18] = right_ankle_roll
# qpos[19] = waist_yaw
# qpos[20] = waist_roll
# qpos[21] = waist_pitch
# qpos[22] = left_shoulder_pitch
# qpos[23] = left_shoulder_roll
# qpos[24] = left_shoulder_yaw
# qpos[25] = left_elbow
# qpos[26] = right_shoulder_pitch
# qpos[27] = right_shoulder_roll
# qpos[28] = right_shoulder_yaw
# qpos[29] = right_elbow

# Pose base para pararse (joints 0..22 relativos, es decir qpos[7..29])
BASE_POSE = np.array([
    # Pierna izquierda (0-5)
    -0.35,  # left_hip_pitch
     0.12,  # left_hip_roll
     0.00,  # left_hip_yaw
     0.50,  # left_knee
    -0.25,  # left_ankle_pitch
    -0.06,  # left_ankle_roll
    # Pierna derecha (6-11)
    -0.35,  # right_hip_pitch
    -0.12,  # right_hip_roll
     0.00,  # right_hip_yaw
     0.50,  # right_knee
    -0.25,  # right_ankle_pitch
     0.06,  # right_ankle_roll
    # Cintura (12-14)
     0.00,  # waist_yaw
     0.00,  # waist_roll
     0.00,  # waist_pitch
    # Brazo izquierdo (15-18)
     0.30,  # left_shoulder_pitch
     0.30,  # left_shoulder_roll
     0.00,  # left_shoulder_yaw
     0.50,  # left_elbow
    # Brazo derecho (19-22)
     0.30,  # right_shoulder_pitch
    -0.30,  # right_shoulder_roll
     0.00,  # right_shoulder_yaw
     0.50,  # right_elbow
], dtype=np.float32)  # 23 valores, uno por actuador


def _key_callback(env, window, key, scancode, action, mods):
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    step = 0.05
    # Velocidad lineal
    if key == glfw.KEY_W:
        env.commands[0] = min(env.commands[0] + step, 0.5)    # adelante
    elif key == glfw.KEY_S:
        env.commands[0] = max(env.commands[0] - step, -0.5)   # atrás
    elif key == glfw.KEY_A:
        env.commands[1] = min(env.commands[1] + step, 0.4)    # lateral izq
    elif key == glfw.KEY_D:
        env.commands[1] = max(env.commands[1] - step, -0.4)   # lateral der
    # Giro
    elif key == glfw.KEY_Q:
        env.commands[2] = min(env.commands[2] + step, 1.0)    # girar izq
    elif key == glfw.KEY_E:
        env.commands[2] = max(env.commands[2] - step, -1.0)   # girar der
    # Altura del torso
    elif key == glfw.KEY_Z:
        env.commands[3] = min(env.commands[3] + step, 0.8)    # subir
    elif key == glfw.KEY_X:
        env.commands[3] = max(env.commands[3] - step, -0.5)   # bajar
    # Pitch del torso
    elif key == glfw.KEY_I:
        env.commands[4] = min(env.commands[4] + step, 1.57)
    elif key == glfw.KEY_K:
        env.commands[4] = max(env.commands[4] - step, -0.52)
    # Roll del torso
    elif key == glfw.KEY_O:
        env.commands[5] = min(env.commands[5] + step, 0.7)
    elif key == glfw.KEY_L:
        env.commands[5] = max(env.commands[5] - step, -0.7)
    # Yaw del torso
    elif key == glfw.KEY_U:
        env.commands[6] = min(env.commands[6] + step, 1.57)
    elif key == glfw.KEY_J:
        env.commands[6] = max(env.commands[6] - step, -1.57)
    # Reset comandos
    elif key == glfw.KEY_SPACE:
        env.commands[:] = 0.0
        print("\n[RESET] Comandos a cero")

    # Mostrar comandos actuales
    print(f"\r[CMD] vx={env.commands[0]:+.2f} vy={env.commands[1]:+.2f} "
          f"vyaw={env.commands[2]:+.2f} h={env.commands[3]:+.2f} "
          f"pitch={env.commands[4]:+.2f} roll={env.commands[5]:+.2f} "
          f"yaw={env.commands[6]:+.2f}   ", end="", flush=True)


class HumanoidEnv:
    def __init__(self, policy_path, adapter_path, device="cuda:0"):

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible.")

        self.device = torch.device(device)

        # ── Modelos ────────────────────────────────────────────────────────
        self.policy  = torch.jit.load(policy_path,  map_location=self.device)
        self.adapter = torch.jit.load(adapter_path, map_location=self.device)
        self.policy.eval()
        self.adapter.eval()
        for p in self.policy.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = False

        # ── MuJoCo ─────────────────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(scene("g1"))
        self.data  = mujoco.MjData(self.model)

        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        glfw.set_key_callback(
            self.viewer.window,
            lambda win, key, sc, act, mods: _key_callback(self, win, key, sc, act, mods)
        )

        # ── Parámetros de observación ───────────────────────────────────────
        # obs_frame = qpos(30) + qvel(29) + adapter_out(15) + last_action(15) + phase(4) = 93
        self.obs_size       = 93
        self.obs_hist_len   = 11   # 11 × 93 = 1023, + 20 comandos = 1043
        self.extra_hist_len = 25   # 25 × 93 = 2325

        self.obs_history   = deque(maxlen=self.obs_hist_len)
        self.extra_history = deque(maxlen=self.extra_hist_len)
        for _ in range(self.obs_hist_len):
            self.obs_history.append(np.zeros(self.obs_size, dtype=np.float32))
        for _ in range(self.extra_hist_len):
            self.extra_history.append(np.zeros(self.obs_size, dtype=np.float32))

        # ── Normalización adapter ───────────────────────────────────────────
        stats = torch.load(policy("adapter_stats"), weights_only=False)
        self.input_mean  = torch.tensor(stats['input_mean'],  device=self.device, dtype=torch.float32)
        self.input_std   = torch.tensor(stats['input_std'],   device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std  = torch.tensor(stats['output_std'],  device=self.device, dtype=torch.float32)

        # ── Estado interno ──────────────────────────────────────────────────
        self.adapter_out_np = np.zeros(15, dtype=np.float32)
        self.last_action    = np.zeros(15, dtype=np.float32)  # últimos torques aplicados

        # Fase de marcha: señal oscilatoria para indicar ciclo de paso
        self.phase     = 0.0
        self.phase_freq = 1.5   # Hz — velocidad del ciclo de marcha

        # Comandos del usuario (20 valores)
        # [0]=vx  [1]=vy  [2]=vyaw  [3]=height  [4]=pitch  [5]=roll  [6]=yaw  [7..19]=0
        self.commands = np.zeros(20, dtype=np.float32)

        # dt de simulación
        self.dt = self.model.opt.timestep  # típicamente 0.002 s

        print(f"\n[INFO] dt={self.dt:.4f}s  obs={self.obs_size}  "
              f"hist={self.obs_hist_len}  extra={self.extra_hist_len}")
        print("[CONTROLES]")
        print("  W/S  = adelante/atrás     A/D = lateral")
        print("  Q/E  = girar izq/der      Z/X = altura torso")
        print("  I/K  = pitch torso        O/L = roll torso")
        print("  U/J  = yaw torso          SPACE = parar\n")

    # ── Reset ───────────────────────────────────────────────────────────────
    def reset_pose(self):
        """Coloca el robot en la pose base estable sobre el suelo."""
        mujoco.mj_resetData(self.model, self.data)

        # Free joint: posición XYZ + cuaternión WXYZ
        self.data.qpos[0] = 0.0    # x
        self.data.qpos[1] = 0.0    # y
        self.data.qpos[2] = 0.793  # z (altura inicial según XML)
        self.data.qpos[3] = 1.0    # qw (vertical)
        self.data.qpos[4] = 0.0    # qx
        self.data.qpos[5] = 0.0    # qy
        self.data.qpos[6] = 0.0    # qz

        # Joints articulados (qpos[7..29] = 23 joints)
        self.data.qpos[7:30] = BASE_POSE

        mujoco.mj_forward(self.model, self.data)

        # Resetear buffers
        for buf in (self.obs_history, self.extra_history):
            for i in range(len(buf)):
                buf[i][:] = 0.0
        self.last_action[:] = 0.0
        self.phase = 0.0

    # ── Construcción del frame de observación ───────────────────────────────
    def get_obs_frame(self):
        """
        Construye el vector de 93 features del frame actual:
          qpos (30) + qvel (29) + adapter_out (15) + last_action (15) + phase (4) = 93
        """
        qpos = self.data.qpos.astype(np.float32)   # 30
        qvel = self.data.qvel.astype(np.float32)   # 29

        # Señal de fase: seno y coseno para cada pierna (ciclo opuesto)
        phase_signal = np.array([
            np.sin(self.phase),
            np.cos(self.phase),
            np.sin(self.phase + np.pi),   # pierna contraria
            np.cos(self.phase + np.pi),
        ], dtype=np.float32)              # 4

        obs_frame = np.concatenate([
            qpos,                  # 30
            qvel,                  # 29
            self.adapter_out_np,   # 15
            self.last_action,      # 15  ← acciones previas
            phase_signal,          # 4
        ])                         # total = 93
        return obs_frame

    # ── Paso de simulación ──────────────────────────────────────────────────
    def step(self):
        qpos_t = torch.tensor(self.data.qpos, device=self.device, dtype=torch.float32)
        qvel_t = torch.tensor(self.data.qvel, device=self.device, dtype=torch.float32)

        # ── Adapter ────────────────────────────────────────────────────────
        # Usa los primeros 6 de qpos y qvel (base del robot)
        adapter_in = torch.cat([qpos_t[:6], qvel_t[:6]]).unsqueeze(0)
        adapter_in = (adapter_in - self.input_mean) / (self.input_std + 1e-8)
        adapter_out = self.adapter(adapter_in)
        adapter_out = adapter_out * self.output_std + self.output_mean
        self.adapter_out_np = adapter_out.squeeze().detach().cpu().numpy()

        # ── Frame de observación ───────────────────────────────────────────
        obs_frame = self.get_obs_frame()
        self.obs_history.append(obs_frame)
        self.extra_history.append(obs_frame)

        # ── Construir tensores para la policy ─────────────────────────────
        obs_hist_flat   = np.array(self.obs_history,   dtype=np.float32).flatten()   # 1023
        extra_hist_flat = np.array(self.extra_history, dtype=np.float32).flatten()   # 2325

        # obs_teacher = historial (1023) + comandos (20) = 1043
        obs_teacher = np.concatenate([obs_hist_flat, self.commands])

        obs_t   = torch.tensor(obs_teacher,    device=self.device, dtype=torch.float32).unsqueeze(0)
        extra_t = torch.tensor(extra_hist_flat, device=self.device, dtype=torch.float32).unsqueeze(0)

        # ── Inferencia ─────────────────────────────────────────────────────
        with torch.no_grad():
            action = self.policy(obs_t, extra_t).squeeze()

        action_np = action.clamp(-40, 40).detach().cpu().numpy()

        # ── Aplicar torques ────────────────────────────────────────────────
        # La policy genera 15 valores (lower body: piernas + cintura)
        # Los 8 restantes del upper body (brazos) se controlan con PD hacia BASE_POSE
        kp_arm, kd_arm = 80.0, 3.0
        q_arms  = self.data.qpos[22:30].astype(np.float32)   # 8 joints de brazos
        qd_arms = self.data.qvel[21:29].astype(np.float32)
        torque_arms = kp_arm * (BASE_POSE[15:] - q_arms) - kd_arm * qd_arms
        torque_arms = np.clip(torque_arms, -20, 20)

        self.data.ctrl[:15] = action_np       # piernas + cintura (policy IA)
        self.data.ctrl[15:] = torque_arms     # brazos (PD estable)

        # ── Guardar acción y avanzar fase ──────────────────────────────────
        self.last_action = action_np.copy()
        self.phase += 2.0 * np.pi * self.phase_freq * self.dt
        if self.phase > 2.0 * np.pi:
            self.phase -= 2.0 * np.pi

        # ── Avanzar simulación ─────────────────────────────────────────────
        mujoco.mj_step(self.model, self.data)
        self.viewer.render()

    # ── Loop principal ──────────────────────────────────────────────────────
    def run(self):
        self.reset_pose()
        while not glfw.window_should_close(self.viewer.window):
            self.step()
        self.viewer.close()


if __name__ == "__main__":
    env = HumanoidEnv(
        policy_path=policy("amo"),
        adapter_path=policy("adapter"),
        device="cuda:0"
    )
    env.run()