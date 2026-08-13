#!/usr/bin/env python3
"""
banda_v2.py — Robot G1 con control de brazos mejorado      Funciona la estabilidad inicial 
======================================================
Igual que banda.py pero con ArmController integrado.

NUEVAS TECLAS DE BRAZOS:
  F1  = Saludar (agitar mano derecha)
  F2  = Apuntar hacia adelante
  F3  = Pose de carga (brazos al frente)
  F4  = Brazos en cruz
  F5  = Pose de guardia
  F6  = Volver a pose neutral
  F7  = Pausar / reanudar secuencia activa
  F8  = Imprimir pose actual de brazos

CONTROL MANUAL FINO DE BRAZOS (como antes):
  Y/H  = Shoulder pitch izquierdo
  [/]  = Shoulder pitch derecho  (nuevo, sin conflicto)
  U/J  = Torso yaw  (sin cambio)
  etc.
"""

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ----------------------------------------------------------

from amo.control import ArmController, ArmSequence
from amo.paths import policy, scene

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import time
from collections import deque

import cv2
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

# ── Importar el controlador de brazos ──────────────────────────────────────


# =============================================================================
# KEYCODES
# =============================================================================
KEY_ESCAPE    = 256
KEY_UP        = 265
KEY_DOWN      = 264
KEY_LEFT      = 263
KEY_RIGHT     = 262
KEY_SPACE     = 32
KEY_ENTER     = 257
KEY_BACKSPACE = 259
KEY_F1        = 290
KEY_F2        = 291
KEY_F3        = 292
KEY_F4        = 293
KEY_F5        = 294
KEY_F6        = 295
KEY_F7        = 296
KEY_F8        = 297

# =============================================================================
# RUTAS
# =============================================================================
PATH_POLICY        = policy("amo")
PATH_ADAPTER       = policy("adapter")
PATH_ADAPTER_STATS = policy("adapter_stats")
PATH_SCENE         = scene("g1")


# =============================================================================
# UTILIDADES
# =============================================================================
def quat_to_euler(quat):
    qw, qx, qy, qz = quat
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr, cosr)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw])


# =============================================================================
# VIEWER STATE
# =============================================================================
class ViewerState:
    def __init__(self):
        self.commands = np.zeros(8, dtype=np.float32)
        self.show_lidar    = False
        self.show_map      = False
        self.print_lidar   = False
        self.clear_map     = False
        self.show_camera   = False
        self.should_exit   = False
        self.paused        = False
        self.movement_started = False
        # Bandas
        self.conveyor_1_speed = 0.0
        self.conveyor_2_speed = 0.0
        # Brazo extra (5 joints del brazo robótico externo, no del G1)
        self.arm_target = np.zeros(5)
        self.arm_step   = 0.1
        self.arm_home   = np.zeros(5)
        self.arm_pick   = np.array([-1.57, 0.3, 1.2, 0.5, 0.0])
        self.arm_place  = np.array([ 1.57, 0.3, 1.2, 0.5, 0.0])
        # Gripper
        self.gripper_active   = False
        self.attached_body_id = -1

    def print_status(self):
        print(f"\r[G1] vx={self.commands[0]:+.2f} vy={self.commands[2]:+.2f} "
              f"yaw={self.commands[1]:+.2f} | "
              f"B1={self.conveyor_1_speed:.1f} B2={self.conveyor_2_speed:.1f} | "
              f"Grip={'ON' if self.gripper_active else 'OFF'}  ",
              end="", flush=True)


def create_key_callback(state: ViewerState, arm_ctrl: ArmController):
    """Crea el callback con soporte para secuencias de brazos F1-F8."""

    def key_callback(keycode):
        # ── Salir ─────────────────────────────────────────────────────────
        if keycode == KEY_ESCAPE:
            state.should_exit = True
            return

        # ── Pausa ─────────────────────────────────────────────────────────
        if keycode == KEY_ENTER:
            state.paused = not state.paused
            state.commands[:3] = 0.0
            print("\n⏸ PAUSADO" if state.paused else "\n▶ REANUDADO")
            return

        # Bloquear movimiento si pausado
        if state.paused and keycode in {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
                                         ord('q'), ord('Q'), ord('e'), ord('E')}:
            print("\n⏸ Pausado — ENTER para reanudar")
            return

        # ── Robot G1: movimiento ───────────────────────────────────────────
        if   keycode == KEY_UP:    state.commands[0] += 0.05
        elif keycode == KEY_DOWN:  state.commands[0] -= 0.05
        elif keycode == KEY_LEFT:  state.commands[1] += 0.1
        elif keycode == KEY_RIGHT: state.commands[1] -= 0.1
        elif keycode in (ord('q'), ord('Q')): state.commands[2] += 0.05
        elif keycode in (ord('e'), ord('E')): state.commands[2] -= 0.05
        elif keycode in (ord('z'), ord('Z')): state.commands[3] += 0.05
        elif keycode in (ord('x'), ord('X')): state.commands[3] -= 0.05

        if keycode in {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT} and not state.paused:
            state.movement_started = True

        # ── Torso ─────────────────────────────────────────────────────────
        elif keycode in (ord('j'), ord('J')): state.commands[4] += 0.1
        elif keycode in (ord('u'), ord('U')): state.commands[4] -= 0.1
        elif keycode in (ord('k'), ord('K')): state.commands[5] += 0.05
        elif keycode in (ord('i'), ord('I')): state.commands[5] -= 0.05
        elif keycode in (ord('l'), ord('L')): state.commands[6] += 0.05
        elif keycode in (ord('o'), ord('O')): state.commands[6] -= 0.1
        elif keycode == KEY_BACKSPACE:
            state.commands[7] = not state.commands[7]
            print(f"\n🦾 Brazos aleatorios: {'ON' if state.commands[7] else 'OFF'}")

        # ── Secuencias de brazos (F1-F8) ──────────────────────────────────
        elif keycode == KEY_F1:
            arm_ctrl.play("wave")
            print("\n🤚 Saludando...")
        elif keycode == KEY_F2:
            arm_ctrl.play("point")
            print("\n👉 Apuntando...")
        elif keycode == KEY_F3:
            arm_ctrl.play("carry")
            print("\n📦 Pose de carga...")
        elif keycode == KEY_F4:
            arm_ctrl.play("cross")
            print("\n✚ Brazos en cruz...")
        elif keycode == KEY_F5:
            arm_ctrl.play("guard")
            print("\n🥊 Pose de guardia...")
        elif keycode == KEY_F6:
            arm_ctrl.play("neutral")
            print("\n😐 Pose neutral...")
        elif keycode == KEY_F7:
            arm_ctrl.pause_toggle()
        elif keycode == KEY_F8:
            arm_ctrl.print_pose()

        # ── Sensores ──────────────────────────────────────────────────────
        elif keycode in (ord('v'), ord('V')):
            state.show_lidar = not state.show_lidar
            print(f"\n📷 LIDAR: {'ON' if state.show_lidar else 'OFF'}")
        elif keycode in (ord('m'), ord('M')):
            state.show_map = not state.show_map
            print(f"\n🗺️ Mapa: {'ON' if state.show_map else 'OFF'}")
        elif keycode in (ord('c'), ord('C')):
            state.clear_map = True
        elif keycode in (ord('b'), ord('B')):
            state.print_lidar = True
        elif keycode in (ord('p'), ord('P')):
            state.show_camera = not state.show_camera

        # ── Bandas ────────────────────────────────────────────────────────
        elif keycode == ord('1'): state.conveyor_1_speed =  0.5; print("\n🔄 Banda 1: ADELANTE")
        elif keycode == ord('2'): state.conveyor_1_speed = -0.5; print("\n🔄 Banda 1: ATRÁS")
        elif keycode == ord('3'): state.conveyor_1_speed =  0.0; print("\n⏹ Banda 1: PARADA")
        elif keycode == ord('4'): state.conveyor_2_speed =  0.5; print("\n🔄 Banda 2: ADELANTE")
        elif keycode == ord('5'): state.conveyor_2_speed = -0.5; print("\n🔄 Banda 2: ATRÁS")
        elif keycode == ord('6'): state.conveyor_2_speed =  0.0; print("\n⏹ Banda 2: PARADA")

        # ── Brazo robótico externo ─────────────────────────────────────────
        elif keycode in (ord('r'), ord('R')): state.arm_target[0] += state.arm_step
        elif keycode in (ord('f'), ord('F')): state.arm_target[0] -= state.arm_step
        elif keycode in (ord('t'), ord('T')): state.arm_target[1] += state.arm_step
        elif keycode in (ord('g'), ord('G')): state.arm_target[1] -= state.arm_step
        elif keycode in (ord('y'), ord('Y')): state.arm_target[2] += state.arm_step
        elif keycode in (ord('h'), ord('H')): state.arm_target[2] -= state.arm_step
        elif keycode in (ord('w'), ord('W')): state.arm_target[3] += state.arm_step
        elif keycode in (ord('s'), ord('S')): state.arm_target[3] -= state.arm_step
        elif keycode in (ord('a'), ord('A')): state.arm_target[4] += state.arm_step
        elif keycode in (ord('d'), ord('D')): state.arm_target[4] -= state.arm_step
        elif keycode in (ord('n'), ord('N')): state.arm_target = state.arm_home.copy()
        elif keycode == ord(','): state.arm_target = state.arm_pick.copy()
        elif keycode == ord('.'): state.arm_target = state.arm_place.copy()

        # ── Gripper ───────────────────────────────────────────────────────
        elif keycode == KEY_SPACE:
            state.gripper_active = not state.gripper_active
            print(f"\n🔧 Gripper: {'ON' if state.gripper_active else 'OFF'}")

        else:
            return

        state.print_status()

    return key_callback


# =============================================================================
# LIDAR 2D (sin cambios respecto a banda.py original)
# =============================================================================
class Lidar2D:
    def __init__(self, model, data, prefix="lidar_", n_rays=32, max_range=10.0):
        self.model = model; self.data = data
        self.max_range = max_range
        self.h_fov = 2 * np.pi
        self.site_ids, self.sensor_names = [], []
        for i in range(n_rays):
            name = f"{prefix}{i}"
            try:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid >= 0:
                    self.site_ids.append(sid); self.sensor_names.append(name)
            except: pass
        self.n_rays = len(self.site_ids)
        self.ranges = np.full(self.n_rays, max_range, dtype=np.float32)
        self.point_cloud = []
        self.angles = np.linspace(0, 2*np.pi, self.n_rays, endpoint=False)
        self._viz = False; self._fig = self._ax = None
        print(f"📷 LIDAR: {self.n_rays} rayos" if self.n_rays else "📷 LIDAR: sin sensores")

    def scan(self):
        self.point_cloud = []; self.ranges.fill(self.max_range)
        for name, sid in zip(self.sensor_names, self.site_ids):
            try:
                d = float(self.data.sensor(name).data[0])
                if not np.isfinite(d) or d <= 0: continue
                self.ranges[self.sensor_names.index(name)] = min(d, self.max_range)
                if d < self.max_range:
                    o = self.data.site_xpos[sid].copy()
                    dr = self.data.site_xmat[sid].reshape(3,3)[:,2]
                    self.point_cloud.append(o + d * dr)
            except: pass

    def get_2d_points(self):
        if not self.point_cloud: return np.empty((0,2))
        return np.array(self.point_cloud)[:,:2]

    @property
    def min_distance(self):
        return float(np.min(self.ranges)) if self.n_rays else self.max_range

    def init_visualization(self):
        plt.ion()
        self._fig, self._ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(6,6))
        self._viz = True

    def update_visualization(self):
        if not self._viz: return
        try:
            self._ax.clear()
            self._ax.scatter(self.angles, np.clip(self.ranges,0,self.max_range), c="red", s=20)
            self._ax.set_rmax(self.max_range)
            self._ax.set_title(f"LIDAR — min: {self.min_distance:.2f}m")
            plt.pause(0.001)
        except: pass

    def close_visualization(self):
        if self._fig: plt.close(self._fig)
        self._fig = self._ax = None; self._viz = False

    @property
    def visualization_enabled(self): return self._viz


# =============================================================================
# MAPA DE OCUPACIÓN (sin cambios)
# =============================================================================
class OccupancyMap:
    def __init__(self, resolution=0.05, size=30.0):
        self.resolution = resolution; self.size = size
        self.origin = size / 2.0
        self.grid_size = int(size / resolution)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.robot_trajectory = []; self.obstacle_points = []
        self._viz = False; self._fig = self._ax = None
        print(f"🗺️ Mapa: {self.grid_size}x{self.grid_size} celdas")

    def world_to_grid(self, x, y):
        return int((x+self.origin)/self.resolution), int((y+self.origin)/self.resolution)

    def is_valid(self, gx, gy):
        return 0 <= gx < self.grid_size and 0 <= gy < self.grid_size

    def update(self, rx, ry, ryaw, lidar):
        self.robot_trajectory.append((rx, ry, ryaw))
        for px, py in lidar.get_2d_points():
            gx, gy = self.world_to_grid(px, py)
            if self.is_valid(gx, gy):
                self.obstacle_points.append((px, py))
                self.log_odds[gx, gy] = np.clip(self.log_odds[gx, gy]+1.0, -10, 10)
                if self.log_odds[gx, gy] > 0.5: self.grid[gx, gy] = 2

    def clear(self):
        self.grid.fill(0); self.log_odds.fill(0)
        self.robot_trajectory.clear(); self.obstacle_points.clear()

    def init_visualization(self):
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(10,10))
        self._viz = True

    def update_visualization(self, rx, ry, ryaw):
        if not self._viz: return
        try:
            self._ax.clear()
            img = np.zeros((self.grid_size, self.grid_size, 3))
            img[self.grid==0]=[.7,.7,.7]; img[self.grid==1]=[1,1,1]; img[self.grid==2]=[.1,.1,.1]
            extent=[-self.origin,self.origin,-self.origin,self.origin]
            self._ax.imshow(img.transpose(1,0,2), extent=extent, origin="lower")
            if self.obstacle_points:
                obs=np.array(self.obstacle_points)
                self._ax.scatter(obs[:,0],obs[:,1],c="red",s=2,alpha=.5)
            self._ax.add_patch(Circle((rx,ry),.3,color="blue",alpha=.9))
            v=8; self._ax.set_xlim(rx-v,rx+v); self._ax.set_ylim(ry-v,ry+v)
            self._ax.set_title("🗺️ Mapa"); plt.pause(0.001)
        except: pass

    def close_visualization(self):
        if self._fig: plt.close(self._fig)
        self._fig = self._ax = None; self._viz = False

    @property
    def visualization_enabled(self): return self._viz


# =============================================================================
# ENTORNO PRINCIPAL
# =============================================================================
class HumanoidEnv:
    NUM_ACTIONS = 15
    NUM_DOFS    = 23
    SIM_DT      = 0.002
    DECIMATION  = 10

    STIFFNESS = np.array([
        150,150,150,300,80,20, 150,150,150,300,80,20,
        400,400,400,
        80,80,40,60, 80,80,40,60,
    ], dtype=np.float32)

    DAMPING = np.array([
        2,2,2,4,2,1, 2,2,2,4,2,1,
        15,15,15,
        2,2,1,1, 2,2,1,1,
    ], dtype=np.float32)

    TORQUE_LIMITS = np.array([
        88,139,88,139,50,50, 88,139,88,139,50,50,
        88,50,50,
        25,25,25,25, 25,25,25,25,
    ], dtype=np.float32)

    DEFAULT_DOF_POS = np.array([
        -0.1,0.0,0.0,0.3,-0.2,0.0,
        -0.1,0.0,0.0,0.3,-0.2,0.0,
         0.0,0.0,0.0,
         0.2, 0.2,0.0,1.28,
         0.2,-0.2,0.0,1.28,
    ], dtype=np.float32)

    IDLE_DOF_POS = np.array([
        -0.10,0.0,-0.10,0.25,-0.20,0.0,
         0.15,0.0,-0.10,0.25,-0.20,0.0,
         0.0,0.0,0.0,
         0.2, 0.2,0.0,1.28,
         0.2,-0.2,0.0,1.28,
    ], dtype=np.float32)

    def __init__(self, policy_jit, device="cuda"):
        self.device     = device
        self.control_dt = self.SIM_DT * self.DECIMATION

        self.action_scale   = 0.25
        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05
        self.n_proprio      = 3+2+2+self.NUM_DOFS*3+2+self.NUM_ACTIONS
        self.n_priv         = 3
        self.history_len    = 10
        self.extra_history_len = 25
        self._n_demo_dof    = 8

        # ── MuJoCo ───────────────────────────────────────────────────────
        print(f"📁 Cargando: {PATH_SCENE}")
        self.model = mujoco.MjModel.from_xml_path(PATH_SCENE)
        self.model.opt.timestep = self.SIM_DT
        self.data  = mujoco.MjData(self.model)
        print(f"   Bodies:{self.model.nbody} Joints:{self.model.njnt} Act:{self.model.nu}")

        # ── IDs de actuadores ─────────────────────────────────────────────
        def _act(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        self.conv1_act_id = _act("conveyor_1_motor")
        self.conv2_act_id = _act("conveyor_2_motor")
        self.arm_act_ids  = [_act(f"arm_act{i+1}") for i in range(5)]

        # IDs de cajas y gripper
        self.box_body_ids = []
        for i in range(1, 5):
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box{i}")
                if bid >= 0: self.box_body_ids.append(bid)
            except: pass
        try:
            self.suction_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "suction_site")
        except:
            self.suction_site_id = -1

        # ── ArmController (NUEVO) ─────────────────────────────────────────
        self.arm_ctrl = ArmController(self.model, self.data)
        print("🦾 ArmController inicializado")
        print("   F1=Saludar  F2=Apuntar  F3=Cargar  F4=Cruz  F5=Guardia  F6=Neutral  F7=Pausa  F8=Info")

        # ── Viewer ────────────────────────────────────────────────────────
        self.state = ViewerState()
        cb = create_key_callback(self.state, self.arm_ctrl)
        print("🖥️ Iniciando viewer...")
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=cb)
        self.viewer.cam.distance  = self.model.stat.extent * 1.5
        self.viewer.cam.elevation = -25
        self.viewer.cam.azimuth   = 180

        # ── Sensores ──────────────────────────────────────────────────────
        self.lidar   = Lidar2D(self.model, self.data)
        self.occ_map = OccupancyMap()
        self._sensor_counter = 0
        self._sensor_freq    = 5

        # Cámara OpenCV
        try:
            self.cv_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
        except:
            self.cv_cam_id = -1
        self.cv_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # ── Policy ────────────────────────────────────────────────────────
        self.policy_jit = policy_jit
        self.adapter = torch.jit.load(PATH_ADAPTER, map_location=device)
        self.adapter.eval()
        for p in self.adapter.parameters(): p.requires_grad = False
        stats = torch.load(PATH_ADAPTER_STATS, weights_only=False)
        def _t(k): return torch.tensor(stats[k], device=device, dtype=torch.float32)
        self.input_mean  = _t("input_mean")
        self.input_std   = _t("input_std")
        self.output_mean = _t("output_mean")
        self.output_std  = _t("output_std")

        # ── Estado del robot ──────────────────────────────────────────────
        self.dof_pos     = np.zeros(self.NUM_DOFS, dtype=np.float32)
        self.dof_vel     = np.zeros(self.NUM_DOFS, dtype=np.float32)
        self.quat        = np.zeros(4, dtype=np.float32)
        self.ang_vel     = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.NUM_DOFS, dtype=np.float32)

        self.target_yaw       = 0.0
        self._in_place_stand  = True
        self.gait_cycle       = np.array([0.25, 0.25])
        self.gait_freq        = 1.3
        self.arm_action       = self.DEFAULT_DOF_POS[15:].copy()
        self.prev_arm_action  = self.DEFAULT_DOF_POS[15:].copy()
        self.arm_blend        = 0.0
        self._toggle_arm      = False

        self.demo_obs_template = np.zeros(8+3+3+3, dtype=np.float32)
        self.demo_obs_template[:8] = self.DEFAULT_DOF_POS[15:]
        self.demo_obs_template[8+6:8+9] = 0.75

        self.proprio_history = deque(maxlen=self.history_len)
        self.extra_history   = deque(maxlen=self.extra_history_len)
        for _ in range(self.history_len):
            self.proprio_history.append(np.zeros(self.n_proprio))
        for _ in range(self.extra_history_len):
            self.extra_history.append(np.zeros(self.n_proprio))

        self._print_instructions()

    def _print_instructions(self):
        sep = "=" * 65
        print(f"\n{sep}")
        print("🤖 G1 v2 | LIDAR | BANDAS | BRAZO | SECUENCIAS DE BRAZOS")
        print(sep)
        print("MOVER    : ↑↓←→  Q/E lateral  Z/X altura  ENTER=pausa")
        print("TORSO    : J/U yaw  K/I pitch  L/O roll")
        print("BRAZOS   : F1=👋Saludar  F2=👉Apuntar  F3=📦Cargar")
        print("           F4=✚Cruz     F5=🥊Guardia  F6=Neutral  F7=⏸")
        print("           F8=Info pose brazos")
        print("BANDAS   : 1/2/3 (B1)  4/5/6 (B2)")
        print("BRAZO EXT: R/F T/G Y/H W/S A/D  N=home ,=pick .=place")
        print("GRIPPER  : ESPACIO")
        print("SENSORES : V=LIDAR  M=Mapa  P=Cámara  B=Print  C=Clear")
        print("SALIR    : ESC")
        print(f"{sep}\n")

    # ── Lectura del estado ────────────────────────────────────────────────
    def _read_state(self):
        self.dof_pos = self.data.qpos[-self.NUM_DOFS:].astype(np.float32)
        self.dof_vel = self.data.qvel[-self.NUM_DOFS:].astype(np.float32)
        self.quat    = self.data.sensor("orientation").data.astype(np.float32)
        self.ang_vel = self.data.sensor("angular-velocity").data.astype(np.float32)

    # ── Adapter ───────────────────────────────────────────────────────────
    def _run_adapter(self):
        cmd = self.state.commands
        raw = np.concatenate([[0.75+cmd[3], cmd[4], cmd[5], cmd[6]], self.dof_pos[15:]])
        inp = torch.tensor(raw, device=self.device, dtype=torch.float32).unsqueeze(0)
        inp = (inp - self.input_mean) / (self.input_std + 1e-8)
        with torch.no_grad():
            out = self.adapter(inp)
        return (out * self.output_std + self.output_mean).cpu().numpy().squeeze()

    # ── Observación ───────────────────────────────────────────────────────
    def _build_obs(self):
        rpy = quat_to_euler(self.quat)
        cmd = self.state.commands
        dyaw = np.remainder(rpy[2]-cmd[1]+np.pi, 2*np.pi) - np.pi
        self._in_place_stand = abs(cmd[0]) < 0.1
        if self._in_place_stand: dyaw = 0.0

        dof_vel_obs = self.dof_vel.copy()
        dof_vel_obs[[4,5,10,11,13,14]] = 0.0
        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)
        adapter_out = self._run_adapter()

        obs_prop = np.concatenate([
            self.ang_vel * self.scales_ang_vel,
            rpy[:2],
            [np.sin(dyaw), np.cos(dyaw)],
            self.dof_pos - self.DEFAULT_DOF_POS,
            dof_vel_obs * self.scales_dof_vel,
            self.last_action,
            gait_obs,
            adapter_out,
        ])

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:8]   = self.dof_pos[15:]
        obs_demo[8]    = cmd[0]
        obs_demo[9]    = cmd[2]
        obs_demo[11]   = cmd[4]
        obs_demo[12]   = cmd[5]
        obs_demo[13]   = cmd[6]
        obs_demo[14:17]= 0.75 + cmd[3]

        obs_priv = np.zeros(self.n_priv, dtype=np.float32)
        obs_hist = np.array(self.proprio_history).flatten()

        self.proprio_history.append(obs_prop)
        self.extra_history.append(obs_prop)

        return np.concatenate([obs_prop, obs_demo, obs_priv, obs_hist])

    # ── Torques PD ────────────────────────────────────────────────────────
    def _compute_torques(self, pd_target):
        t = (pd_target - self.dof_pos) * self.STIFFNESS - self.dof_vel * self.DAMPING
        return np.clip(t, -self.TORQUE_LIMITS, self.TORQUE_LIMITS)

    # ── Sensores ──────────────────────────────────────────────────────────
    def _update_sensors(self):
        rpy = quat_to_euler(self.quat)
        rx, ry = self.data.qpos[0], self.data.qpos[1]
        self.lidar.scan()
        if self.state.clear_map:
            self.state.clear_map = False; self.occ_map.clear()
        self.occ_map.update(rx, ry, rpy[2], self.lidar)
        if self.state.print_lidar:
            self.state.print_lidar = False
            for i,d in enumerate(self.lidar.ranges): print(f"  r[{i:02d}]={d:.2f}m")

        # Lidar viz
        if self.state.show_lidar:
            if not self.lidar.visualization_enabled: self.lidar.init_visualization()
            self.lidar.update_visualization()
        elif self.lidar.visualization_enabled:
            self.lidar.close_visualization()

        # Mapa viz
        if self.state.show_map:
            if not self.occ_map.visualization_enabled: self.occ_map.init_visualization()
            self.occ_map.update_visualization(rx, ry, rpy[2])
        elif self.occ_map.visualization_enabled:
            self.occ_map.close_visualization()

    # ── Cámara OpenCV ─────────────────────────────────────────────────────
    def _render_camera(self):
        if not self.state.show_camera or self.cv_cam_id < 0: return
        try:
            self.cv_renderer.update_scene(self.data, camera=self.cv_cam_id)
            img = cv2.cvtColor(self.cv_renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("head_cam", img); cv2.waitKey(1)
        except: pass

    # ── Bandas ────────────────────────────────────────────────────────────
    def _update_conveyors(self):
        if self.conv1_act_id >= 0: self.data.ctrl[self.conv1_act_id] = self.state.conveyor_1_speed
        if self.conv2_act_id >= 0: self.data.ctrl[self.conv2_act_id] = self.state.conveyor_2_speed

    # ── Brazo robótico externo ────────────────────────────────────────────
    def _update_ext_arm(self):
        for i, aid in enumerate(self.arm_act_ids):
            if aid >= 0: self.data.ctrl[aid] = self.state.arm_target[i]

    # ── Gripper ───────────────────────────────────────────────────────────
    def _update_gripper(self):
        if self.state.gripper_active:
            if self.state.attached_body_id < 0: self._try_attach()
            self._hold_box()
        else:
            if self.state.attached_body_id >= 0: self._detach()

    def _try_attach(self):
        if self.suction_site_id < 0: return
        sp = self.data.site_xpos[self.suction_site_id]
        best, bd = -1, 0.15
        for bid in self.box_body_ids:
            d = np.linalg.norm(sp - self.data.xpos[bid])
            if d < bd: bd, best = d, bid
        if best >= 0:
            self.state.attached_body_id = best
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, best)
            print(f"\n✅ Caja '{name}' agarrada")
        else:
            print("\n⚠️ No hay caja cerca")

    def _hold_box(self):
        if self.suction_site_id < 0 or self.state.attached_body_id < 0: return
        sp  = self.data.site_xpos[self.suction_site_id].copy()
        sm  = self.data.site_xmat[self.suction_site_id].reshape(3,3)
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.state.attached_body_id)
        try:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            qa  = self.model.jnt_qposadr[jid]
            va  = self.model.jnt_dofadr[jid]
            self.data.qpos[qa:qa+3] = sp + sm @ np.array([0,0,-0.1])
            self.data.qvel[va:va+6] = 0
        except: pass

    def _detach(self):
        self.state.attached_body_id = -1

    # ── Loop principal ────────────────────────────────────────────────────
    def run(self):
        pd_target = self.IDLE_DOF_POS.copy()
        print("\n🚀 Simulación iniciada!")
        print("🧍 Robot en reposo — presiona ↑ para caminar")
        print("🦾 Prueba F1 para ver saludar al robot\n")

        try:
            step = 0
            while self.viewer.is_running() and not self.state.should_exit:
                t0 = time.time()
                self._read_state()

                if step % self.DECIMATION == 0:
                    if not self.state.movement_started:
                        pd_target = self.IDLE_DOF_POS.copy()
                    else:
                        obs    = self._build_obs()
                        obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                        extra  = torch.tensor(
                            np.array(self.extra_history).flatten(),
                            dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action = self.policy_jit(obs_t, extra).cpu().numpy().squeeze()
                        action = np.clip(action, -40, 40)

                        self.last_action = np.concatenate([
                            action.copy(),
                            (self.dof_pos - self.DEFAULT_DOF_POS)[15:] / self.action_scale
                        ])

                        # Brazos aleatorios (BACKSPACE)
                        if step % 300 == 0 and step > 0 and self.state.commands[7]:
                            if not self.arm_ctrl._active_seq:   # no interrumpir secuencia
                                self.arm_blend      = 0
                                self.prev_arm_action= self.dof_pos[15:].copy()
                                self.arm_action     = np.random.uniform(
                                    self.arm_ctrl.JOINT_LOWER,
                                    self.arm_ctrl.JOINT_UPPER
                                )
                                self._toggle_arm = True
                        elif not self.state.commands[7] and self._toggle_arm:
                            self._toggle_arm     = False
                            self.arm_blend       = 0
                            self.prev_arm_action = self.dof_pos[15:].copy()
                            self.arm_action      = self.DEFAULT_DOF_POS[15:].copy()

                        pd_target = np.concatenate([action*self.action_scale, np.zeros(8)]) + self.DEFAULT_DOF_POS
                        pd_target[15:] = ((1-self.arm_blend)*self.prev_arm_action
                                          + self.arm_blend*self.arm_action)
                        self.arm_blend = min(1.0, self.arm_blend + 0.01)

                        self.gait_cycle = np.remainder(
                            self.gait_cycle + self.control_dt*self.gait_freq, 1.0
                        )

                    # ── ArmController: actualizar y aplicar (NUEVO) ───────
                    # Solo actúa cuando hay una secuencia activa
                    if self.arm_ctrl._active_seq and self.arm_ctrl._active_seq.active:
                        self.arm_ctrl.update()
                        self.arm_ctrl.apply(self.data)
                        # Sobreescribir pd_target[15:] con la pose de la secuencia
                        pd_target[15:] = self.arm_ctrl.target_pose
                    else:
                        self.arm_ctrl.update()   # actualiza current_pose

                    # Sensores
                    self._sensor_counter += 1
                    if self._sensor_counter >= self._sensor_freq:
                        self._sensor_counter = 0
                        self._update_sensors()

                    self._render_camera()

                # Aplicar torques al G1
                torque = self._compute_torques(pd_target)
                for j in range(self.NUM_DOFS):
                    self.data.ctrl[j] = torque[j]

                self._update_conveyors()
                self._update_ext_arm()
                self._update_gripper()

                mujoco.mj_step(self.model, self.data)
                self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()

                step += 1
                remaining = self.SIM_DT - (time.time() - t0)
                if remaining > 0: time.sleep(remaining)

        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido")
        finally:
            self.lidar.close_visualization()
            self.occ_map.close_visualization()
            cv2.destroyAllWindows()
            try:
                if self.viewer.is_running(): self.viewer.close()
            except: pass
            print("✅ Finalizado")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Dispositivo: {device}")
    policy_jit = torch.jit.load(PATH_POLICY, map_location=device)
    env = HumanoidEnv(policy_jit=policy_jit, device=device)
    env.run()