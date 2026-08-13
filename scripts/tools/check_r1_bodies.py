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
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cpu"))
cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "r1_description" / "r1.usd")),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.74)),
    actuators={},
)
robot = Articulation(cfg.replace(prim_path="/World/Robot"))
sim.reset()
print("\n" + "="*50)
print("CUERPOS (BODIES) DEL R1:")
for i, n in enumerate(robot.body_names):
    print(f"  [{i:2d}] {n}")
print("="*50)
app.close()
