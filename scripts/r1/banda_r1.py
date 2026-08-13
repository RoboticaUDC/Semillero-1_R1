#!/usr/bin/env python3
""""
banda_r1.py — Robot R1 con control de brazos
=============================================
Adaptación de banda_v2.1.py para el modelo r1.xml.

CAMBIOS RESPECTO AL G1:
  - Sensores IMU: imu_ang_vel, imu_lin_acc (no "orientation"/"angular-velocity")
  - Quaternion leído de qpos[3:7] (floating_base_joint free)
  - DOFs del R1: 24 joints (6L+6R piernas, 2 cintura, 5L+5R brazos)
  - num_actions: 14 (piernas + cintura, sin brazos)
  - Actuadores de banda/brazo externo: opcionales (se ignoran si no existen)
  - default_dof_pos ajustado al R1

JOINTS DEL R1 (orden en qpos tras el free joint):
  [0]  left_hip_pitch_joint
  [1]  left_hip_roll_joint
  [2]  left_hip_yaw_joint
  [3]  left_knee_joint
  [4]  left_ankle_pitch_joint
  [5]  left_ankle_roll_joint
  [6]  right_hip_pitch_joint
  [7]  right_hip_roll_joint
  [8]  right_hip_yaw_joint
  [9]  right_knee_joint
  [10] right_ankle_pitch_joint
  [11] right_ankle_roll_joint
  [12] waist_roll_joint
  [13] waist_yaw_joint
  [14] left_shoulder_pitch_joint
  [15] left_shoulder_roll_joint
  [16] left_shoulder_yaw_joint
  [17] left_elbow_joint
  [18] left_wrist_roll_joint
  [19] right_shoulder_pitch_joint
  [20] right_shoulder_roll_joint
  [21] right_shoulder_yaw_joint
  [22] right_elbow_joint
  [23] right_wrist_roll_joint

TECLAS DE BRAZOS:
  F1  = Saludar (agitar mano derecha)
  F2  = Apuntar hacia adelante
  F3  = Pose de carga (brazos al frente)
  F4  = Brazos en cruz
  F5  = Pose de guardia
  F6  = Volver a pose neutral
  F7  = Pausar / reanudar secuencia activa
  F8  = Imprimir pose actual de brazos
"""

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

from amo.control import ArmController, ArmSequence
from amo.math_utils import quatToEuler
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
RUTA_POLITICA                  = policy("amo")
RUTA_POLITICA_ADAPTADORA       = policy("adapter")
RUTA_POLITICA_ADAPTADORA_ESTADOS = policy("adapter_stats")
ESCENARIO_RUTA                 = scene("r1")


# =============================================================================
# CLASE VIEWER STATE
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
        # Brazo extra (5 joints del brazo robótico externo, no del R1)
        self.arm_target = np.zeros(5)
        self.arm_step   = 0.1
        self.arm_home   = np.zeros(5)
        self.arm_pick   = np.array([-1.57, 0.3, 1.2, 0.5, 0.0])
        self.arm_place  = np.array([ 1.57, 0.3, 1.2, 0.5, 0.0])
        # Gripper
        self.gripper_active   = False
        self.attached_body_id = -1

    def print_status(self):
        print(f"\r[R1] vx={self.commands[0]:+.2f} vy={self.commands[2]:+.2f} "
              f"yaw={self.commands[1]:+.2f} | "
              f"B1={self.conveyor_1_speed:.1f} B2={self.conveyor_2_speed:.1f} | "
              f"Grip={'ON' if self.gripper_active else 'OFF'}  ",
              end="", flush=True)


def create_key_callback(state: ViewerState, arm_ctrl: ArmController):
    def key_callback(keycode):

        if keycode == KEY_ESCAPE:
            state.should_exit = True
            return

        # ── Pausa ─────────────────────────────────────────────────────────
        if keycode == KEY_ENTER:
            state.paused = not state.paused
            state.commands[:3] = 0.0
            print("\n⏸ PAUSADO" if state.paused else "\n▶ REANUDADO")
            return

        if state.paused and keycode in {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
                                         ord('q'), ord('Q'), ord('e'), ord('E')}:
            print("\n⏸ Pausado — ENTER para reanudar")
            return

        # ── Robot R1: movimiento ───────────────────────────────────────────
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
            arm_ctrl.play("wave");    print("\n🤚 Saludando...")
        elif keycode == KEY_F2:
            arm_ctrl.play("point");   print("\n👉 Apuntando...")
        elif keycode == KEY_F3:
            arm_ctrl.play("carry");   print("\n📦 Pose de carga...")
        elif keycode == KEY_F4:
            arm_ctrl.play("cross");   print("\n✚ Brazos en cruz...")
        elif keycode == KEY_F5:
            arm_ctrl.play("guard");   print("\n🥊 Pose de guardia...")
        elif keycode == KEY_F6:
            arm_ctrl.play("neutral"); print("\n😐 Pose neutral...")
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
# LIDAR 2D
# =============================================================================
class Lidar2DRangefinder:
    def __init__(self, model, data, prefix="lidar_", n_rays=32, max_range=10.0):
        self.model = model
        self.data = data
        self.prefix = prefix
        self.n_rays = n_rays
        self.max_range = max_range
        self.h_fov = 2 * np.pi
        
        self.site_ids = []
        self.sensor_names = []
        for i in range(n_rays):
            site_name = f"{prefix}{i}"
            try:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if sid >= 0:
                    self.site_ids.append(sid)
                    self.sensor_names.append(site_name)
            except Exception as e:
                print(e)
        
        self.n_rays = len(self.site_ids)
        self.ranges = np.full(self.n_rays, self.max_range, dtype=np.float32) if self.n_rays > 0 else np.array([])
        self.point_cloud = []
        self.visualization_enabled = False
        self.fig = None
        self.ax = None
        self.angles = np.linspace(0.0, 2.0 * np.pi, self.n_rays, endpoint=False) if self.n_rays > 0 else np.array([])
        
        if self.n_rays > 0:
            print(f"📷 LIDAR 2D: {self.n_rays} rayos, rango {max_range}m")
        else:
            print("⚠️ LIDAR 2D: No se encontraron sensores lidar_* en r1.xml")

    def scan(self):
        if self.n_rays == 0:
            return
        self.point_cloud = []
        self.ranges.fill(self.max_range)
        for i in range(self.n_rays):
            sname = self.sensor_names[i]
            sid = self.site_ids[i]
            if sid < 0:
                continue
            try:
                dist = float(self.data.sensor(sname).data[0])
                if not np.isfinite(dist) or dist <= 0.0:
                    continue
                self.ranges[i] = min(dist, self.max_range)
                if dist >= self.max_range:
                    continue
                origin = self.data.site_xpos[sid].copy()
                R = self.data.site_xmat[sid].reshape(3, 3)
                dir_global = R[:, 2]
                end_point = origin + dist * dir_global
                self.point_cloud.append(end_point)
            except Exception as e:
                print(e)

    def get_2d_points(self):
        if not self.point_cloud:
            return np.array([]).reshape(0, 2)
        pts = np.array(self.point_cloud)
        return pts[:, :2]

    @property
    def min_distance(self):
        return float(np.min(self.ranges)) if len(self.ranges) > 0 else self.max_range

    def print_data(self):
        print("\n" + "=" * 50)
        print(f"📷 LIDAR 2D ({self.n_rays} rayos)")
        print("=" * 50)
        if self.n_rays > 0:
            for i, d in enumerate(self.ranges):
                print(f"  r[{i:02d}] = {d:.2f}m")
            print(f"\n  Min: {self.min_distance:.2f}m | Puntos: {len(self.point_cloud)}")
        print("=" * 50)

    def init_visualization(self):
        if self.n_rays == 0:
            return
        plt.ion()
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        self.fig.canvas.manager.set_window_title('LIDAR 2D')
        self.visualization_enabled = True

    def update_visualization(self):
        if not self.visualization_enabled or self.fig is None or self.n_rays == 0:
            return
        try:
            self.ax.clear()
            r_display = np.clip(self.ranges, 0, self.max_range)
            self.ax.scatter(self.angles, r_display, c='red', s=20, alpha=0.8)
            self.ax.set_rmax(self.max_range)
            self.ax.set_title(f'LIDAR 2D - min: {self.min_distance:.2f}m')
            self.ax.grid(True)
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        except Exception as e:
            print(e)

    def close_visualization(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.visualization_enabled = False


# =============================================================================
# MAPA 2D
# =============================================================================
class OccupancyMap:
    def __init__(self, resolution=0.05, size=30.0):
        self.resolution = resolution
        self.size = size
        self.grid_size = int(size / resolution)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.l_occ = 1.0
        self.l_free = -0.3
        self.l_max = 10.0
        self.l_min = -10.0
        self.occ_threshold = 0.5
        self.free_threshold = -0.5
        self.robot_trajectory = []
        self.max_trajectory = 10000
        self.obstacle_points = []
        self.max_obstacle_points = 50000
        self.origin = size / 2.0
        self.fig = None
        self.ax = None
        self.visualization_enabled = False
        print(f"🗺️ Mapa: {self.grid_size}x{self.grid_size} celdas")

    def world_to_grid(self, x, y):
        gx = int((x + self.origin) / self.resolution)
        gy = int((y + self.origin) / self.resolution)
        return gx, gy

    def is_valid(self, gx, gy):
        return 0 <= gx < self.grid_size and 0 <= gy < self.grid_size

    def update(self, rx, ry, ryaw, lidar):
        self.robot_trajectory.append((rx, ry, ryaw))
        for px, py in lidar.get_2d_points():
            gx, gy = self.world_to_grid(px, py)
            if self.is_valid(gx, gy):
                self.obstacle_points.append((px, py))
                self.log_odds[gy, gx] = np.clip(
                    self.log_odds[gy, gx] + 1.0, -10, 10
                )
            if self.log_odds[gy, gx] > 0.5:
                self.grid[gy, gx] = 2

    def clear(self):
        self.grid.fill(0)
        self.log_odds.fill(0)
        self.robot_trajectory.clear()
        self.obstacle_points.clear()
        print("🧹 Mapa limpiado")

    def get_stats(self):
        total = self.grid_size ** 2
        free = np.sum(self.grid == 1)
        occupied = np.sum(self.grid == 2)
        return {'explored_pct': (free + occupied) / total * 100, 'free': free, 'occupied': occupied}

    def init_visualization(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Mapa 2D')
        self.visualization_enabled = True

    def update_visualization(self, robot_x, robot_y, robot_yaw):
        if not self.visualization_enabled or self.fig is None:
            return
        try:
            self.ax.clear()
            map_img = np.zeros((self.grid_size, self.grid_size, 3))
            map_img[self.grid == 0] = [0.7, 0.7, 0.7]
            map_img[self.grid == 1] = [1.0, 1.0, 1.0]
            map_img[self.grid == 2] = [0.1, 0.1, 0.1]
            extent = [-self.origin, self.origin, -self.origin, self.origin]
            self.ax.imshow(map_img.transpose(1, 0, 2), extent=extent, origin='lower')
            if self.obstacle_points:
                obs_array = np.array(self.obstacle_points)
                self.ax.scatter(obs_array[:, 0], obs_array[:, 1], c='red', s=2, alpha=0.5)
            if len(self.robot_trajectory) > 1:
                traj = np.array(self.robot_trajectory)
                self.ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, alpha=0.6)
            robot_circle = Circle((robot_x, robot_y), 0.3, color='blue', alpha=0.9)
            self.ax.add_patch(robot_circle)
            view_range = 8
            self.ax.set_xlim(robot_x - view_range, robot_x + view_range)
            self.ax.set_ylim(robot_y - view_range, robot_y + view_range)
            self.ax.set_title('🗺️ Mapa de Exploración')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal')
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        except Exception as e:
            print(e)

    def close_visualization(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.visualization_enabled = False


# =============================================================================
# ENTORNO PRINCIPAL
# =============================================================================
class HumanoidEnv:
    """
    Entorno MuJoCo para el robot R1.

    Estructura de DOFs (24 total, todos los joints rotacionales tras
    el free joint del pelvis):
      [0-5]   pierna izquierda  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
      [6-11]  pierna derecha    (idem)
      [12-13] cintura           (waist_roll, waist_yaw)
      [14-18] brazo izquierdo   (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll)
      [19-23] brazo derecho     (idem)

    num_actions = 14  → piernas (12) + cintura (2)
    Los brazos [14:] se controlan por ArmController / brazos aleatorios.
    """

    # ── Índices de DOF ────────────────────────────────────────────────────
    # Piernas: 0-11 | Cintura: 12-13 | Brazos: 14-23
    NUM_DOFS        = 24
    NUM_ACTIONS     = 14   # piernas + cintura
    NUM_ARM_DOFS    = 10   # 5 brazo izq + 5 brazo der

    def __init__(self, policy_jit, robot_type="r1", device="cuda"):
        self.robot_type = robot_type
        self.device = device

        # ── Ganancias PD ──────────────────────────────────────────────────
        # [pierna_izq x6, pierna_der x6, cintura x2, brazo_izq x5, brazo_der x5]
        self.stiffness = np.array([
            # pierna izq (soporte)
            180, 180, 160, 320, 120, 60,

            # pierna der
            180, 180, 160, 320, 120, 60,

            # cintura (MUY importante para estabilidad)
            350, 350,

            # brazo izq (suave para no meter oscilación)
            60, 60, 40, 30, 15,

            # brazo der
            60, 60, 40, 30, 15
        ], dtype=np.float32)
        
        self.damping = np.array([
            # pierna izq (amortiguación fuerte = menos temblor)
            14, 14, 12, 18, 10, 6,

            # pierna der
            14, 14, 12, 18, 10, 6,

            # cintura (clave contra el “shaking”)
            25, 25,

            # brazos (muy amortiguados)
            10, 10, 8, 8, 6,

            10, 10, 8, 8, 6
        ], dtype=np.float32)
        
        self.torque_limits = np.array([
            88, 139, 88, 139, 50, 50,
            88, 139, 88, 139, 50, 50,
            88, 50,
            25, 25, 25, 25, 25,
            25, 25, 25, 25, 25,
        ], dtype=np.float32)

        # ── Pose por defecto del R1 ───────────────────────────────────────
        # Postura neutral erecta; ajusta según necesites
        self.default_dof_pos = np.array([
            # pierna izq
            -0.08, 0.0, 0.0, 0.18, -0.12, 0.0,
        
            # pierna der
            -0.08, 0.0, 0.0, 0.18, -0.12, 0.0,
        
            # cintura (ligera compensación natural)
            0.0, 0.0,
        
            # brazos (colgando, no rígidos)
            0.15, 0.10, 0.0, 1.60, 0.0,
            0.15,-0.10, 0.0, 1.60, 0.0
        ], dtype=np.float32)

        # Límites de articulaciones de brazo (para movimiento aleatorio)
        # Extraídos del XML del R1
        self.arm_dof_lower = np.array([
            -3.1416, -0.22689, -1.9199, -0.97564, -1.9199,   # brazo izq
            -3.1416, -2.47849, -1.9199, -0.97564, -1.9199,   # brazo der
        ])
        self.arm_dof_upper = np.array([
             2.0944,  2.4784,  1.9199,  2.1852,  1.9199,
             2.0944,  0.2268,  1.9199,  2.1852,  1.9199,
        ])

        self.sim_dt         = 0.002
        self.sim_decimation = 1
        self.control_dt     = self.sim_dt * self.sim_decimation
        self.n_proprio      = 3 + 2 + 2 + self.NUM_DOFS*3 + 2 + self.NUM_ACTIONS

        # ── Cargar modelo ─────────────────────────────────────────────────
        print(f"📁 Cargando: {ESCENARIO_RUTA}")
        self.model = mujoco.MjModel.from_xml_path(ESCENARIO_RUTA)
        self.model.opt.timestep = self.sim_dt
        self.data  = mujoco.MjData(self.model)
        print(f"   Bodies: {self.model.nbody} | Joints: {self.model.njnt} | Actuadores: {self.model.nu}")

        # ── IDs de sensores IMU del R1 ────────────────────────────────────
        # El R1 expone: imu_ang_vel (gyro), imu_lin_vel (velocimeter),
        # imu_lin_acc (accelerometer), root_angmom (subtreeangmom)
        # NO hay sensor "orientation" ni "angular-velocity" como en el G1.
        # El quaternion se lee directamente de qpos[3:7] (free joint).
        self._imu_ang_vel_name = "imu_ang_vel"
        self._imu_lin_acc_name = "imu_lin_acc"

        # ── IDs de actuadores opcionales (banda / brazo externo) ──────────
        # Si no existen en el XML del R1 quedan en -1 y se ignoran.
        def _safe_act_id(name):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            return aid  # -1 si no existe

        self.conv1_act_id = _safe_act_id("conveyor_1_motor")
        self.conv2_act_id = _safe_act_id("conveyor_2_motor")
        self.arm_act_ids  = [_safe_act_id(f"arm_act{i+1}") for i in range(5)]

        if self.conv1_act_id < 0:
            print("ℹ️  Actuadores de banda no encontrados en r1.xml (se ignoran)")
        if all(a < 0 for a in self.arm_act_ids):
            print("ℹ️  Actuadores de brazo externo no encontrados en r1.xml (se ignoran)")

        # ── IDs de cajas y ventosa (opcionales) ───────────────────────────
        self.box_body_ids = []
        for i in range(1, 5):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box{i}")
            if bid >= 0:
                self.box_body_ids.append(bid)

        self.suction_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "suction_site"
        )

        # ── ArmController ─────────────────────────────────────────────────
        self.arm_ctrl = ArmController(self.model, self.data)
        print("🦾 ArmController inicializado")
        print("   F1=Saludar  F2=Apuntar  F3=Cargar  F4=Cruz  F5=Guardia  F6=Neutral  F7=Pausa  F8=Info")

        # ── Viewer ────────────────────────────────────────────────────────
        self.state = ViewerState()
        cb = create_key_callback(self.state, self.arm_ctrl)
        print("🖥️ Inicializando viewer...")
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=cb)
        self.viewer.cam.distance  = self.model.stat.extent * 1.5
        self.viewer.cam.elevation = -25
        self.viewer.cam.azimuth   = 180

        # ── LIDAR y mapa ──────────────────────────────────────────────────
        self.lidar    = Lidar2DRangefinder(self.model, self.data)
        self.occ_map  = OccupancyMap()
        self._sensor_counter = 0
        self._sensor_freq    = 5

        # ── Cámara OpenCV ─────────────────────────────────────────────────
        self.cv_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam"
        )
        self.cv_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # ── Policy ────────────────────────────────────────────────────────
        self.policy_jit = policy_jit
        self.adapter    = torch.jit.load(RUTA_POLITICA_ADAPTADORA, map_location=device)
        self.adapter.eval()
        for p in self.adapter.parameters():
            p.requires_grad = False
        stats = torch.load(RUTA_POLITICA_ADAPTADORA_ESTADOS, weights_only=False)
        def _t(k):
            return torch.tensor(stats[k], device=device, dtype=torch.float32)
        self.input_mean  = _t("input_mean")
        self.input_std   = _t("input_std")
        self.output_mean = _t("output_mean")
        self.output_std  = _t("output_std")

        # ── Estado del robot ──────────────────────────────────────────────
        self.dof_pos     = np.zeros(self.NUM_DOFS,   dtype=np.float32)
        self.dof_vel     = np.zeros(self.NUM_DOFS,   dtype=np.float32)
        self.quat        = np.array([1., 0., 0., 0.], dtype=np.float32)  # w,x,y,z
        self.ang_vel     = np.zeros(3,               dtype=np.float32)
        self.last_action = np.zeros(self.NUM_DOFS,   dtype=np.float32)

        self.target_yaw      = 0.0
        self._in_place_stand = True
        self.gait_cycle      = np.array([0.25, 0.25])
        self.gait_freq       = 1.3
        self.arm_action      = self.default_dof_pos[self.NUM_ACTIONS:].copy()
        self.prev_arm_action = self.default_dof_pos[self.NUM_ACTIONS:].copy()
        self.arm_blend       = 0.0
        self._toggle_arm     = False

        self.history_len      = 10
        self.scales_ang_vel   = 0.25
        self.scales_dof_vel   = 0.05
        self.action_scale     = 0.5
        self.n_priv           = 0
        self.extra_history_len = 25

        # Plantilla de obs demo: primeros NUM_ARM_DOFS = pose brazos, luego comandos/altura
        self.demo_obs_template = np.zeros(self.NUM_ARM_DOFS + 3 + 3 + 3, dtype=np.float32)
        self.demo_obs_template[:self.NUM_ARM_DOFS]              = self.default_dof_pos[self.NUM_ACTIONS:]
        self.demo_obs_template[self.NUM_ARM_DOFS + 6:
                               self.NUM_ARM_DOFS + 9]           = 0.75  # altura por defecto

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
        print("🤖 R1 v1 | LIDAR | BANDAS | BRAZO | SECUENCIAS DE BRAZOS")
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
        """
        Lee posición/velocidad de joints y datos IMU del R1.

        El free joint ocupa qpos[0:7] y qvel[0:6]:
          qpos[0:3]  = posición xyz del pelvis
          qpos[3:7]  = quaternion (w, x, y, z) del pelvis  ← orientación
          qpos[7:]   = joints rotacionales (NUM_DOFS)
          qvel[0:3]  = vel lineal pelvis
          qvel[3:6]  = vel angular pelvis
          qvel[6:]   = velocidades de joints (NUM_DOFS)

        Sensores disponibles en r1.xml:
          imu_ang_vel  → gyro en el site "imu" (3D)
          imu_lin_vel  → velocímetro en "imu" (3D)
          imu_lin_acc  → acelerómetro en "imu" (3D)
        """
        # Joints (excluir los 7 elementos del free joint al inicio de qpos)
        self.dof_pos = self.data.qpos[7: 7 + self.NUM_DOFS].astype(np.float32)
        self.dof_vel = self.data.qvel[6: 6 + self.NUM_DOFS].astype(np.float32)

        # Quaternion de orientación: MuJoCo almacena (w, x, y, z) para free joints
        self.quat = self.data.qpos[3:7].astype(np.float32)  # [w, x, y, z]

        # Velocidad angular del IMU (gyro)
        try:
            self.ang_vel = self.data.sensor(self._imu_ang_vel_name).data.astype(np.float32)
        except Exception:
            # Fallback: usar vel angular del free joint
            self.ang_vel = self.data.qvel[3:6].astype(np.float32)

    # ── Adapter ───────────────────────────────────────────────────────────
    def _run_adapter(self):
        cmd = self.state.commands
        # El adapter fue entrenado con G1 (8 DOFs brazo), usamos solo los 8 primeros
        raw = np.concatenate([
            [0.75 + cmd[3], cmd[4], cmd[5], cmd[6]],
            self.dof_pos[self.NUM_ACTIONS: self.NUM_ACTIONS + 8]  # solo 8 DOFs
        ])
        inp = torch.tensor(raw, device=self.device, dtype=torch.float32).unsqueeze(0)
        inp = (inp - self.input_mean) / (self.input_std + 1e-8)
        with torch.no_grad():
            out = self.adapter(inp)
        return (out * self.output_std + self.output_mean).cpu().numpy().squeeze()

    # ── Observación ───────────────────────────────────────────────────────
    def _build_obs(self):
        rpy = quatToEuler(self.quat)
        cmd = self.state.commands
        dyaw = np.remainder(rpy[2] - cmd[1] + np.pi, 2 * np.pi) - np.pi
        self._in_place_stand = abs(cmd[0]) < 0.1
        if self._in_place_stand:
            dyaw = 0.0

        dof_vel_obs = self.dof_vel.copy()
        # Anular velocidades de joints pasivos (igual que G1: índices 4,5,10,11 + cintura)
        passive_idx = [4, 5, 10, 11, 13, 14]
        for idx in passive_idx:
            if idx < len(dof_vel_obs):
                dof_vel_obs[idx] = 0.0

        gait_obs    = np.sin(self.gait_cycle * 2 * np.pi)
        adapter_out = self._run_adapter()

        obs_prop = np.concatenate([
            self.ang_vel * self.scales_ang_vel,
            rpy[:2],
            [np.sin(dyaw), np.cos(dyaw)],
            self.dof_pos - self.default_dof_pos,
            dof_vel_obs * self.scales_dof_vel,
            self.last_action,
            gait_obs,
            adapter_out,
        ])

        # Obs demo (brazos + comandos + altura)
        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self.NUM_ARM_DOFS] = self.dof_pos[self.NUM_ACTIONS:]
        obs_demo[self.NUM_ARM_DOFS]     = cmd[0]
        obs_demo[self.NUM_ARM_DOFS + 1] = cmd[2]
        obs_demo[self.NUM_ARM_DOFS + 3] = cmd[4]
        obs_demo[self.NUM_ARM_DOFS + 4] = cmd[5]
        obs_demo[self.NUM_ARM_DOFS + 5] = cmd[6]
        obs_demo[self.NUM_ARM_DOFS + 6:
                 self.NUM_ARM_DOFS + 9] = np.full(3, 0.75 + cmd[3])

        obs_priv = np.zeros(self.n_priv, dtype=np.float32)
        self.proprio_history.append(obs_prop)
        self.extra_history.append(obs_prop)
        obs_hist = np.array(self.proprio_history).flatten()

        return np.concatenate([obs_prop, obs_demo, obs_priv, obs_hist])

    # ── Torques PD ────────────────────────────────────────────────────────
    def _compute_torques(self, pd_target):
        t = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
        return np.clip(t, -self.torque_limits, self.torque_limits)

    # ── Sensores ──────────────────────────────────────────────────────────
    def _update_sensors(self):
        rpy = quatToEuler(self.quat)
        rx, ry = self.data.qpos[0], self.data.qpos[1]
        self.lidar.scan()
        if self.state.clear_map:
            self.state.clear_map = False
            self.occ_map.clear()
        self.occ_map.update(rx, ry, rpy[2], self.lidar)
        if self.state.print_lidar:
            self.state.print_lidar = False
            for i, d in enumerate(self.lidar.ranges):
                print(f"  r[{i:02d}]={d:.2f}m")

        if self.state.show_lidar:
            if not self.lidar.visualization_enabled:
                self.lidar.init_visualization()
            self.lidar.update_visualization()
        elif self.lidar.visualization_enabled:
            self.lidar.close_visualization()

        if self.state.show_map:
            if not self.occ_map.visualization_enabled:
                self.occ_map.init_visualization()
            self.occ_map.update_visualization(rx, ry, rpy[2])
        elif self.occ_map.visualization_enabled:
            self.occ_map.close_visualization()

    # ── Cámara OpenCV ─────────────────────────────────────────────────────
    def _render_camera(self):
        if not self.state.show_camera or self.cv_cam_id < 0:
            return
        try:
            self.cv_renderer.update_scene(self.data, camera=self.cv_cam_id)
            img = cv2.cvtColor(self.cv_renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("head_cam", img)
            cv2.waitKey(1)
        except Exception as e:
            print(e)

    # ── Bandas (opcionales) ───────────────────────────────────────────────
    def _update_conveyors(self):
        if self.model.nu == 0:
            return
        if self.conv1_act_id >= 0:
            self.data.ctrl[self.conv1_act_id] = self.state.conveyor_1_speed
        if self.conv2_act_id >= 0:
            self.data.ctrl[self.conv2_act_id] = self.state.conveyor_2_speed

    # ── Brazo robótico externo (opcional) ─────────────────────────────────
    def _update_ext_arm(self):
        if self.model.nu == 0:
            return
        for i, aid in enumerate(self.arm_act_ids):
            if aid >= 0:
                self.data.ctrl[aid] = self.state.arm_target[i]

    # ── Gripper ───────────────────────────────────────────────────────────
    def _update_gripper(self):
        if self.state.gripper_active:
            if self.state.attached_body_id < 0:
                self._try_attach()
            self._hold_box()
        else:
            if self.state.attached_body_id >= 0:
                self._detach()

    def _try_attach(self):
        if self.suction_site_id < 0:
            return
        suction_pos = self.data.site_xpos[self.suction_site_id]
        min_dist, closest_box = 0.15, -1
        for box_id in self.box_body_ids:
            dist = np.linalg.norm(suction_pos - self.data.xpos[box_id])
            if dist < min_dist:
                min_dist, closest_box = dist, box_id
        if closest_box >= 0:
            self.state.attached_body_id = closest_box
            box_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, closest_box)
            print(f"   ✅ Caja '{box_name}' agarrada")
        else:
            print("   ⚠️ No hay caja cerca")

    def _detach(self):
        if self.state.attached_body_id >= 0:
            box_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.state.attached_body_id
            )
            print(f"   📦 Caja '{box_name}' soltada")
        self.state.attached_body_id = -1

    def _hold_box(self):
        if self.state.attached_body_id < 0 or self.suction_site_id < 0:
            return
        suction_pos = self.data.site_xpos[self.suction_site_id].copy()
        suction_mat = self.data.site_xmat[self.suction_site_id].reshape(3, 3)
        box_name    = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.state.attached_body_id
        )
        joint_name = f"{box_name}_joint"
        try:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                return
            qpos_adr = self.model.jnt_qposadr[joint_id]
            offset   = suction_mat @ np.array([0, 0, -0.1])
            self.data.qpos[qpos_adr: qpos_adr+3] = suction_pos + offset
            qvel_adr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[qvel_adr: qvel_adr+6] = 0
        except Exception as e:
            print(f"⚠️ Error sujetando caja: {e}")

    # ── Aplicar torques al R1 ─────────────────────────────────────────────
    def _apply_torques(self, pd_target):
        """
        Aplica torques PD a los actuadores del R1.

        El R1 en r1.xml no tiene sección <actuator>; si se añaden
        actuadores position/torque, se usan directamente.
        Si no hay actuadores (nu == 0), se usa mj_applyFT vía qfrc_applied.
        """
        torque = self._compute_torques(pd_target)

        if self.model.nu > 0:
            # Hay actuadores definidos: escribir en ctrl (se ignoran bandas/brazo externo
            # ya manejados antes, que están al final)
            n = min(self.NUM_DOFS, self.model.nu)
            for j in range(n):
                self.data.ctrl[j] = torque[j]
        else:
            # Sin actuadores: inyectar torques directamente en qfrc_applied
            # Los DOFs del robot empiezan en el índice 6 (tras 6 DOFs del free joint)
            self.data.qfrc_applied[6: 6 + self.NUM_DOFS] = torque

    # ── Loop principal ────────────────────────────────────────────────────
    def run(self):
        pd_target = self.default_dof_pos.copy()
        print("\n🚀 Simulación iniciada!")
        print("🧍 Robot en reposo — presiona ↑ para caminar")
        print("🦾 Prueba F1 para ver saludar al robot\n")

        try:
            step = 0
            while self.viewer.is_running() and not self.state.should_exit:
                t0 = time.time()
                self._read_state()

                if step % self.sim_decimation == 0:
                    if not self.state.movement_started:
                        pd_target = self.default_dof_pos.copy()
                    else:
                        obs   = self._build_obs()
                        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                        extra = torch.tensor(
                            np.array(self.extra_history).flatten(),
                            dtype=torch.float32
                        ).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            action = self.policy_jit(obs_t, extra).cpu().numpy().squeeze()
                        action = np.clip(action, -40, 40)

                        self.last_action = np.concatenate([
                            action.copy(),
                            (self.dof_pos - self.default_dof_pos)[self.NUM_ACTIONS:] / self.action_scale
                        ])

                        # Brazos aleatorios (BACKSPACE)
                        if step % 300 == 0 and step > 0 and self.state.commands[7]:
                            if not self.arm_ctrl._active_seq:
                                self.arm_blend       = 0
                                self.prev_arm_action = self.dof_pos[self.NUM_ACTIONS:].copy()
                                self.arm_action      = np.random.uniform(
                                    self.arm_dof_lower, self.arm_dof_upper
                                )
                                self._toggle_arm = True
                        elif not self.state.commands[7] and self._toggle_arm:
                            self._toggle_arm     = False
                            self.arm_blend       = 0
                            self.prev_arm_action = self.dof_pos[self.NUM_ACTIONS:].copy()
                            self.arm_action      = self.default_dof_pos[self.NUM_ACTIONS:].copy()

                        pd_target = (
                            np.concatenate([action * self.action_scale,
                                            np.zeros(self.NUM_ARM_DOFS)])
                            + self.default_dof_pos
                        )
                        pd_target[self.NUM_ACTIONS:] = (
                            (1 - self.arm_blend) * self.prev_arm_action
                            + self.arm_blend     * self.arm_action
                        )
                        self.arm_blend = min(1.0, self.arm_blend + 0.01)

                        self.gait_cycle = np.remainder(
                            self.gait_cycle + self.control_dt * self.gait_freq, 1.0
                        )

                    # ── ArmController ─────────────────────────────────────
                    if (
                        self.arm_ctrl._active_seq is not None and
                        self.arm_ctrl._active_seq.active
                    ):
                        self.arm_ctrl.update()
                        self.arm_ctrl.apply(self.data)
                        pd_target[self.NUM_ACTIONS:] = self.arm_ctrl.target_pose
                    else:
                        self.arm_ctrl.update()

                    # ── Sensores ──────────────────────────────────────────
                    self._sensor_counter += 1
                    if self._sensor_counter >= self._sensor_freq:
                        self._sensor_counter = 0
                        self._update_sensors()

                    self._render_camera()

                # Aplicar torques al R1
                self._apply_torques(pd_target)

                self._update_conveyors()
                self._update_ext_arm()
                self._update_gripper()

                mujoco.mj_step(self.model, self.data)
                # Seguir la posición del pelvis (qpos[0:3])
                self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()

                step += 1
                time_until_next = self.control_dt - (time.time() - t0)
                if time_until_next > 0:
                    time.sleep(time_until_next)

        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido")
        finally:
            self.lidar.close_visualization()
            self.occ_map.close_visualization()
            cv2.destroyAllWindows()
            try:
                if self.viewer.is_running():
                    self.viewer.close()
            except Exception as e:
                print(e)
            print("✅ Finalizado")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Dispositivo: {device}")

    print("📦 Cargando política...")
    policy_jit = torch.jit.load(RUTA_POLITICA, map_location=device)

    print("🤖 Inicializando entorno...")
    env = HumanoidEnv(policy_jit=policy_jit, robot_type="r1", device=device)

    print("▶️ Iniciando simulación...")
    env.run()
