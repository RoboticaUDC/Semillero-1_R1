#!/usr/bin/env python3
"""
Banda.py
SIMULADOR DE ROBOT HUMANOIDE G1 CON LIDAR 2D, MAPEO, CÁMARA,
DOS BANDAS TRANSPORTADORAS Y BRAZO ROBÓTICO CON GRIPPER DE 6 VENTOSAS   no funciona la estabilidad incicial
=============================================================================
Controles:

    === ROBOT G1 ===
    ↑/↓ : Velocidad adelante/atrás
    ←/→ : Girar izquierda/derecha
    Q/E : Velocidad lateral
    Z/X : Subir/bajar altura
    J/U : Torso yaw
    K/I : Torso pitch
    L/O : Torso roll
    T   : Toggle control de brazos aleatorio

    === SENSORES Y MAPA ===
    V   : Toggle visualización LIDAR
    M   : Toggle visualización MAPA
    B   : Imprimir datos del LIDAR
    C   : Limpiar mapa
    P   : Toggle cámara OpenCV

    === BANDA TRANSPORTADORA 1 ===
    1   : Banda 1 ADELANTE
    2   : Banda 1 ATRÁS
    3   : Banda 1 PARAR

    === BANDA TRANSPORTADORA 2 ===
    4   : Banda 2 ADELANTE
    5   : Banda 2 ATRÁS
    6   : Banda 2 PARAR

    === BRAZO ROBÓTICO ===
    Y/H : Joint 1 (base)
    U/J : Joint 2 (shoulder)
    I/K : Joint 3 (elbow)
    O/L : Joint 4 (wrist pitch)
    [/] : Joint 5 (wrist roll)
    
    N   : Posición HOME
    ,   : Posición PICK
    .   : Posición PLACE
    
    ESPACIO : Toggle GRIPPER (ventosas)

    ESC : Salir
=============================================================================
"""

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ----------------------------------------------------------

from amo.paths import policy, scene

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import cv2
import time


# =============================================================================
# KEYCODES
# =============================================================================
KEY_ESCAPE = 256
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263 
KEY_RIGHT = 262
KEY_SPACE = 32

# =============================================================================
# RUTAS
# =============================================================================
RUTA_POLITICA = policy("amo")
RUTA_POLITICA_ADAPTADORA = policy("adapter")
RUTA_POLITICA_ADAPTADORA_ESTADOS = policy("adapter_stats")
ESCENARIO_RUTA = scene("g1")


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qz)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    eulerVec[1] = np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    return eulerVec


# =============================================================================
# CLASE VIEWER STATE
# =============================================================================
class ViewerState:
    def __init__(self):
        self.commands = np.zeros(8, dtype=np.float32)
        self.show_sensor = False
        self.show_map = False
        self.print_sensor_data = False
        self.clear_map = False
        self.show_cv_cam = False
        self.should_exit = False
        
        # Bandas
        self.conveyor_1_speed = 0.0
        self.conveyor_2_speed = 0.0
        
        # Brazo
        self.arm_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.arm_step = 0.1
        self.arm_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.arm_pick = np.array([-1.57, 0.3, 1.2, 0.5, 0.0])
        self.arm_place = np.array([1.57, 0.3, 1.2, 0.5, 0.0])
        
        # Gripper
        self.gripper_active = False
        self.attached_body_id = -1

    def print_status(self):
        print(f"G1: vx={self.commands[0]:.2f} vy={self.commands[2]:.2f} yaw={self.commands[1]:.2f} | "
              f"B1={self.conveyor_1_speed:.1f} B2={self.conveyor_2_speed:.1f} | "
              f"Grip={'ON' if self.gripper_active else 'OFF'}")


def create_key_callback(state: ViewerState):
    def key_callback(keycode):
        # === MOVIMIENTO G1 ===
        if keycode == KEY_DOWN:
            state.commands[0] -= 0.05
        elif keycode == KEY_UP:
            state.commands[0] += 0.05
        elif keycode == KEY_LEFT:
            state.commands[1] += 0.1
        elif keycode == KEY_RIGHT:
            state.commands[1] -= 0.1
        elif keycode == ord('q') or keycode == ord('Q'):
            state.commands[2] += 0.05
        elif keycode == ord('e') or keycode == ord('E'):
            state.commands[2] -= 0.05
        elif keycode == ord('z') or keycode == ord('Z'):
            state.commands[3] += 0.05
        elif keycode == ord('x') or keycode == ord('X'):
            state.commands[3] -= 0.05
        
        # === TORSO G1 ===
        elif keycode == ord('j') or keycode == ord('J'):
            state.commands[4] += 0.1
        elif keycode == ord('u') or keycode == ord('U'):
            state.commands[4] -= 0.1
        elif keycode == ord('k') or keycode == ord('K'):
            state.commands[5] += 0.05
        elif keycode == ord('i') or keycode == ord('I'):
            state.commands[5] -= 0.05
        elif keycode == ord('l') or keycode == ord('L'):
            state.commands[6] += 0.05
        elif keycode == ord('o') or keycode == ord('O'):
            state.commands[6] -= 0.1
        elif keycode == ord('t') or keycode == ord('T'):
            state.commands[7] = not state.commands[7]
            print(f"🦾 Brazos aleatorios G1: {'ON' if state.commands[7] else 'OFF'}")
        
        # === VISUALIZACIÓN ===
        elif keycode == ord('v') or keycode == ord('V'):
            state.show_sensor = not state.show_sensor
            print(f"📷 LIDAR: {'ON' if state.show_sensor else 'OFF'}")
        elif keycode == ord('m') or keycode == ord('M'):
            state.show_map = not state.show_map
            print(f"🗺️ MAPA: {'ON' if state.show_map else 'OFF'}")
        elif keycode == ord('c') or keycode == ord('C'):
            state.clear_map = True
            print("🧹 Limpiando mapa...")
        elif keycode == ord('b') or keycode == ord('B'):
            state.print_sensor_data = True
        elif keycode == ord('p') or keycode == ord('P'):
            state.show_cv_cam = not state.show_cv_cam
            print(f"🎥 Cámara: {'ON' if state.show_cv_cam else 'OFF'}")
        
        # === BANDA 1 ===
        elif keycode == ord('1'):
            state.conveyor_1_speed = 0.5
            print("🔄 Banda 1: ADELANTE")
        elif keycode == ord('2'):
            state.conveyor_1_speed = -0.5
            print("🔄 Banda 1: ATRÁS")
        elif keycode == ord('3'):
            state.conveyor_1_speed = 0.0
            print("⏹️ Banda 1: PARADA")
        
        # === BANDA 2 ===
        elif keycode == ord('4'):
            state.conveyor_2_speed = 0.5
            print("🔄 Banda 2: ADELANTE")
        elif keycode == ord('5'):
            state.conveyor_2_speed = -0.5
            print("🔄 Banda 2: ATRÁS")
        elif keycode == ord('6'):
            state.conveyor_2_speed = 0.0
            print("⏹️ Banda 2: PARADA")
        
        # === BRAZO ROBÓTICO ===
        elif keycode == ord('y') or keycode == ord('Y'):
            state.arm_target[0] += state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        elif keycode == ord('h') or keycode == ord('H'):
            state.arm_target[0] -= state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        elif keycode == ord('u') or keycode == ord('U'):
            state.arm_target[1] += state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        elif keycode == ord('j') or keycode == ord('J'):
            state.arm_target[1] -= state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        elif keycode == ord('i') or keycode == ord('I'):
            state.arm_target[2] += state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        elif keycode == ord('k') or keycode == ord('K'):
            state.arm_target[2] -= state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        elif keycode == ord('o') or keycode == ord('O'):
            state.arm_target[3] += state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        elif keycode == ord('l') or keycode == ord('L'):
            state.arm_target[3] -= state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        elif keycode == ord('['):
            state.arm_target[4] += state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        elif keycode == ord(']'):
            state.arm_target[4] -= state.arm_step
            print(f"🦾 Brazo: {state.arm_target}")
        
        # === POSICIONES PREDEFINIDAS ===
        elif keycode == ord('n') or keycode == ord('N'):
            state.arm_target = state.arm_home.copy()
            print("🏠 Brazo -> HOME")
        elif keycode == ord(','):
            state.arm_target = state.arm_pick.copy()
            print("📥 Brazo -> PICK")
        elif keycode == ord('.'):
            state.arm_target = state.arm_place.copy()
            print("📤 Brazo -> PLACE")
        
        # === GRIPPER ===
        elif keycode == KEY_SPACE:
            state.gripper_active = not state.gripper_active
            print(f"🔧 Gripper 6 ventosas: {'ACTIVADO' if state.gripper_active else 'DESACTIVADO'}")
        
        # === SALIR ===
        elif keycode == KEY_ESCAPE:
            state.should_exit = True
            return
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
            except:
                pass
        
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
            print("⚠️ LIDAR 2D: No se encontraron sensores")

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
            except:
                pass

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
        except:
            pass

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

    def update_from_sensor(self, robot_x, robot_y, robot_yaw, sensor):
        if len(self.robot_trajectory) < self.max_trajectory:
            self.robot_trajectory.append((robot_x, robot_y, robot_yaw))
        robot_gx, robot_gy = self.world_to_grid(robot_x, robot_y)
        points_2d = sensor.get_2d_points()
        if len(points_2d) == 0:
            return
        for point in points_2d:
            px, py = point
            gx, gy = self.world_to_grid(px, py)
            if self.is_valid(gx, gy):
                if len(self.obstacle_points) < self.max_obstacle_points:
                    self.obstacle_points.append((px, py))
                self.log_odds[gx, gy] = np.clip(self.log_odds[gx, gy] + self.l_occ, self.l_min, self.l_max)
                if self.log_odds[gx, gy] > self.occ_threshold:
                    self.grid[gx, gy] = 2

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

    def update_visualization(self, robot_x, robot_y, robot_yaw, sensor_fov=None):
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
        except:
            pass

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
    def __init__(self, policy_jit, robot_type="g1", device="cuda"):
        self.robot_type = robot_type
        self.device = device

        self.stiffness = np.array([
            150, 150, 150, 300, 80, 20,
            150, 150, 150, 300, 80, 20,
            400, 400, 400,
            80, 80, 40, 60,
            80, 80, 40, 60,
        ])
        self.damping = np.array([
            2, 2, 2, 4, 2, 1,
            2, 2, 2, 4, 2, 1,
            15, 15, 15,
            2, 2, 1, 1,
            2, 2, 1, 1,
        ])
        self.num_actions = 15
        self.num_dofs = 23
        self.default_dof_pos = np.array([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            0.0, 0.0, 0.0,
            0.2, 0.2, 0.0, 1.28,
            0.2, -0.2, 0.0, 1.28,
        ])
        self.torque_limits = np.array([
            88, 139, 88, 139, 50, 50,
            88, 139, 88, 139, 50, 50,
            88, 50, 50,
            25, 25, 25, 25,
            25, 25, 25, 25,
        ])
        self.arm_dof_lower_range = -0.4 * np.ones(8)
        self.arm_dof_upper_range = 0.4 * np.ones(8)

        self.sim_dt = 0.002
        self.sim_decimation = 10
        self.control_dt = self.sim_dt * self.sim_decimation

        print(f"📁 Cargando: {ESCENARIO_RUTA}")
        self.model = mujoco.MjModel.from_xml_path(ESCENARIO_RUTA)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        print(f"   Bodies: {self.model.nbody} | Joints: {self.model.njnt} | Actuadores: {self.model.nu}")

        # IDs de actuadores
        self.conv1_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "conveyor_1_motor")
        self.conv2_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "conveyor_2_motor")
        self.arm_act_ids = []
        for i in range(5):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"arm_act{i+1}")
            self.arm_act_ids.append(aid)

        # IDs de cajas
        self.box_body_ids = []
        for i in range(1, 5):
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"box{i}")
                if bid >= 0:
                    self.box_body_ids.append(bid)
            except:
                pass

        # Sitio de la ventosa
        try:
            self.suction_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "suction_site")
        except:
            self.suction_site_id = -1

        self.viewer_state = ViewerState()
        self.key_callback = create_key_callback(self.viewer_state)

        print("🖥️ Inicializando viewer...")
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
        self.viewer.cam.distance = self.model.stat.extent * 1.5
        self.viewer.cam.elevation = -25
        self.viewer.cam.azimuth = 180

        # Cámara OpenCV
        self.cv_cam_name = "head_cam"
        try:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cv_cam_name)
            self.cv_cam_id = cam_id if cam_id >= 0 else -1
        except:
            self.cv_cam_id = -1
        self.cv_renderer = mujoco.Renderer(self.model, height=480, width=640)

        # LIDAR y mapa
        self.rgbd_sensor = Lidar2DRangefinder(self.model, self.data, prefix="lidar_", n_rays=32, max_range=10.0)
        self.occupancy_map = OccupancyMap(resolution=0.05, size=30.0)
        self.sensor_update_freq = 3
        self.sensor_step_counter = 0

        # Estado robot
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.25
        self.arm_action = self.default_dof_pos[15:]
        self.prev_arm_action = self.default_dof_pos[15:]
        self.arm_blend = 0.0
        self.toggle_arm = False
        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05
        self.nj = 23
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8

        self.dof_pos = np.zeros(self.nj, dtype=np.float32)
        self.dof_vel = np.zeros(self.nj, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.nj)

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3,))
        self.demo_obs_template[:self._n_demo_dof] = self.default_dof_pos[15:]
        self.demo_obs_template[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75

        self.target_yaw = 0.0
        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for _ in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

        self.policy_jit = policy_jit

        # Adapter
        self.adapter = torch.jit.load(RUTA_POLITICA_ADAPTADORA, map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False

        norm_stats = torch.load(RUTA_POLITICA_ADAPTADORA_ESTADOS, weights_only=False)
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)

        self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        self._print_instructions()

    def _print_instructions(self):
        print("\n" + "=" * 70)
        print("🤖 ROBOT G1 + LIDAR + BANDAS + BRAZO CON 6 VENTOSAS")
        print("=" * 70)
        print("MOVIMIENTO G1: ↑↓←→ | Q/E lateral | Z/X altura")
        print("BANDAS: 1/2/3 (B1) | 4/5/6 (B2)")
        print("BRAZO: Y/H J1 | U/J J2 | I/K J3 | O/L J4 | [/] J5")
        print("POSICIONES: N=HOME | ,=PICK | .=PLACE")
        print("GRIPPER: ESPACIO")
        print("SENSORES: V=LIDAR | M=MAPA | P=Cámara | B=Print | C=Clear")
        print("ESC: Salir")
        print("=" * 70 + "\n")

    def extract_data(self):
        self.dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        self.dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)

    def get_observation(self):
        rpy = quatToEuler(self.quat)
        self.target_yaw = self.viewer_state.commands[1]
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0

        obs_dof_vel = self.dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0
        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])
        self.adapter_input[0] = 0.75 + self.viewer_state.commands[3]
        self.adapter_input[1] = self.viewer_state.commands[4]
        self.adapter_input[2] = self.viewer_state.commands[5]
        self.adapter_input[3] = self.viewer_state.commands[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)
        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        obs_prop = np.concatenate([
            self.ang_vel * self.scales_ang_vel,
            rpy[:2],
            (np.sin(dyaw), np.cos(dyaw)),
            (self.dof_pos - self.default_dof_pos),
            self.dof_vel * self.scales_dof_vel,
            self.last_action,
            gait_obs,
            self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv,))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]
        obs_demo[self._n_demo_dof] = self.viewer_state.commands[0]
        obs_demo[self._n_demo_dof + 1] = self.viewer_state.commands[2]
        self._in_place_stand_flag = np.abs(self.viewer_state.commands[0]) < 0.1
        obs_demo[self._n_demo_dof + 3] = self.viewer_state.commands[4]
        obs_demo[self._n_demo_dof + 4] = self.viewer_state.commands[5]
        obs_demo[self._n_demo_dof + 5] = self.viewer_state.commands[6]
        obs_demo[self._n_demo_dof + 6:self._n_demo_dof + 9] = 0.75 + self.viewer_state.commands[3]

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)

        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))

    def update_sensor_and_map(self):
        robot_pos = self.data.qpos[:3]
        rpy = quatToEuler(self.quat)
        robot_yaw = rpy[2]

        self.rgbd_sensor.scan()

        if self.viewer_state.clear_map:
            self.viewer_state.clear_map = False
            self.occupancy_map.clear()

        self.occupancy_map.update_from_sensor(robot_pos[0], robot_pos[1], robot_yaw, self.rgbd_sensor)

        if self.viewer_state.print_sensor_data:
            self.viewer_state.print_sensor_data = False
            self.rgbd_sensor.print_data()

        if self.viewer_state.show_sensor:
            if not self.rgbd_sensor.visualization_enabled:
                self.rgbd_sensor.init_visualization()
            self.rgbd_sensor.update_visualization()
        else:
            if self.rgbd_sensor.visualization_enabled:
                self.rgbd_sensor.close_visualization()

        if self.viewer_state.show_map:
            if not self.occupancy_map.visualization_enabled:
                self.occupancy_map.init_visualization()
            self.occupancy_map.update_visualization(robot_pos[0], robot_pos[1], robot_yaw, self.rgbd_sensor.h_fov)
        else:
            if self.occupancy_map.visualization_enabled:
                self.occupancy_map.close_visualization()

    def render_cv_camera(self):
        if not self.viewer_state.show_cv_cam or self.cv_cam_id < 0:
            return
        try:
            self.cv_renderer.update_scene(self.data, camera=self.cv_cam_id)
            img = self.cv_renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.cv_cam_name, img_bgr)
            cv2.waitKey(1)
        except:
            pass

    def update_conveyors(self):
        if self.conv1_act_id >= 0:
            self.data.ctrl[self.conv1_act_id] = self.viewer_state.conveyor_1_speed
        if self.conv2_act_id >= 0:
            self.data.ctrl[self.conv2_act_id] = self.viewer_state.conveyor_2_speed

    def update_robot_arm(self):
        for i, act_id in enumerate(self.arm_act_ids):
            if act_id >= 0:
                self.data.ctrl[act_id] = self.viewer_state.arm_target[i]

    def try_attach_box(self):
        if self.suction_site_id < 0:
            return
        suction_pos = self.data.site_xpos[self.suction_site_id]
        min_dist = 0.15
        closest_box = -1
        for box_id in self.box_body_ids:
            box_pos = self.data.xpos[box_id]
            dist = np.linalg.norm(suction_pos - box_pos)
            if dist < min_dist:
                min_dist = dist
                closest_box = box_id
        if closest_box >= 0:
            self.viewer_state.attached_body_id = closest_box
            box_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, closest_box)
            print(f"   ✅ Caja '{box_name}' agarrada con 6 ventosas!")
        else:
            print("   ⚠️ No hay caja cerca")

    def detach_box(self):
        if self.viewer_state.attached_body_id >= 0:
            box_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.viewer_state.attached_body_id)
            print(f"   📦 Caja '{box_name}' soltada")
        self.viewer_state.attached_body_id = -1

    def update_gripper(self):
        if self.viewer_state.gripper_active and self.viewer_state.attached_body_id >= 0:
            suction_pos = self.data.site_xpos[self.suction_site_id].copy()
            suction_mat = self.data.site_xmat[self.suction_site_id].reshape(3, 3)
            box_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.viewer_state.attached_body_id)
            joint_name = f"{box_name}_joint"
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_adr = self.model.jnt_qposadr[joint_id]
                offset = suction_mat @ np.array([0, 0, -0.1])
                self.data.qpos[qpos_adr:qpos_adr+3] = suction_pos + offset
                qvel_adr = self.model.jnt_dofadr[joint_id]
                self.data.qvel[qvel_adr:qvel_adr+6] = 0
            except:
                pass

    def run(self):
        pd_target = self.default_dof_pos.copy()
        print("\n🚀 Simulación iniciada!")

        try:
            i = 0
            while self.viewer.is_running() and not self.viewer_state.should_exit:
                step_start = time.time()

                self.extract_data()

                if i % self.sim_decimation == 0:
                    obs = self.get_observation()
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        extra_hist = torch.tensor(
                            np.array(self.extra_history_buf).flatten().copy(),
                            dtype=torch.float
                        ).view(1, -1).to(self.device)
                        raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()

                    raw_action = np.clip(raw_action, -40., 40.)
                    self.last_action = np.concatenate([
                        raw_action.copy(),
                        (self.dof_pos - self.default_dof_pos)[15:] / self.action_scale
                    ])
                    scaled_actions = raw_action * self.action_scale

                    if i % 300 == 0 and i > 0 and self.viewer_state.commands[7]:
                        self.arm_blend = 0
                        self.prev_arm_action = self.dof_pos[15:].copy()
                        self.arm_action = (
                            np.random.uniform(0, 1, 8) *
                            (self.arm_dof_upper_range - self.arm_dof_lower_range) +
                            self.arm_dof_lower_range
                        )
                        self.toggle_arm = True
                    elif not self.viewer_state.commands[7]:
                        if self.toggle_arm:
                            self.toggle_arm = False
                            self.arm_blend = 0
                            self.prev_arm_action = self.dof_pos[15:].copy()
                            self.arm_action = self.default_dof_pos[15:]

                    pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + self.default_dof_pos
                    pd_target[15:] = (1 - self.arm_blend) * self.prev_arm_action + self.arm_blend * self.arm_action
                    self.arm_blend = min(1.0, self.arm_blend + 0.01)

                    self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)

                    self.sensor_step_counter += 1
                    if self.sensor_step_counter >= self.sensor_update_freq:
                        self.sensor_step_counter = 0
                        self.update_sensor_and_map()

                    self.render_cv_camera()

                # Control del robot G1
                torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                for j in range(self.num_dofs):
                    self.data.ctrl[j] = torque[j]

                # Control bandas y brazo
                self.update_conveyors()
                self.update_robot_arm()

                # Gripper
                if self.viewer_state.gripper_active:
                    if self.viewer_state.attached_body_id < 0:
                        self.try_attach_box()
                    self.update_gripper()
                else:
                    if self.viewer_state.attached_body_id >= 0:
                        self.detach_box()

                mujoco.mj_step(self.model, self.data)
                self.viewer.cam.lookat[:] = self.data.qpos[:3]
                self.viewer.sync()

                i += 1
                time_until_next_step = self.sim_dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido")
        finally:
            self.rgbd_sensor.close_visualization()
            self.occupancy_map.close_visualization()
            cv2.destroyAllWindows()
            try:
                if self.viewer.is_running():
                    self.viewer.close()
            except:
                pass
            print("✅ Simulación finalizada")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Dispositivo: {device}")

    print("📦 Cargando política...")
    policy_jit = torch.jit.load(RUTA_POLITICA, map_location=device)

    print("🤖 Inicializando entorno...")
    env = HumanoidEnv(policy_jit=policy_jit, robot_type="g1", device=device)

    print("▶️ Iniciando simulación...")
    env.run()

