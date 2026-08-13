# --- rutas del repo, sin depender del CWD ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from amo.paths import ASSETS, THIRD_PARTY, POLICIES
# -------------------------------------------

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg

USD_PATH = str(ASSETS / "r1_description" / "r1.usd")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))
cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.74)),
    actuators={},
)
robot = Articulation(cfg.replace(prim_path="/World/Robot"))
sim.reset()

print("\n" + "="*60)
print(f"NUM DOFS: {robot.num_joints}")
print("="*60)
print("ORDEN DE JOINTS EN ISAAC:")
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")
print("="*60)
print("\nORDEN EN TU MUJOCO (qpos[7:]):")
mujoco_order = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint",
    "left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint",
    "right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_roll_joint","waist_yaw_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint",
    "left_elbow_joint","left_wrist_roll_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
    "right_elbow_joint","right_wrist_roll_joint",
]
isaac_names = list(robot.joint_names)
print("\nMAPEO  isaac_idx -> mujoco_idx:")
mapping = []
for mj_i, name in enumerate(mujoco_order):
    is_i = isaac_names.index(name) if name in isaac_names else -1
    mapping.append(is_i)
    flag = "" if is_i == mj_i else "  <-- DIFERENTE"
    print(f"  mujoco[{mj_i:2d}] {name:32s} = isaac[{is_i}]{flag}")
print("\nmujoco_from_isaac =", [isaac_names.index(n) for n in mujoco_order])
print("="*60)
simulation_app.close()
