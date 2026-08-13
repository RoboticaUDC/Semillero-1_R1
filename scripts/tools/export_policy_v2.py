#!/usr/bin/env python3
"""Exporta el actor de un checkpoint de rsl_rl a TorchScript.

El .pt resultante es lo que cargan play_r1_isaac.py y play_r1_camina_brazos.py.
La arquitectura esta fija (405 -> 512 -> 256 -> 128 -> 24); si entrenas con otra,
mira primero `inspeccionar_checkpoint.py` y ajusta ARQUITECTURA.

    python scripts/tools/export_policy_v2.py
    python scripts/tools/export_policy_v2.py --ckpt ... --out policies/r1_policy_v3.pt
"""

# --- rutas del repo, sin depender del CWD ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from amo.paths import POLICIES, THIRD_PARTY
# -------------------------------------------

import argparse

import torch
import torch.nn as nn

LOGS = THIRD_PARTY / "IsaacLab" / "logs" / "rsl_rl"

OBS_DIM = 405     # 6 terminos de observacion x history de 5 pasos
ACT_DIM = 24      # 24 joints del R1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", type=Path,
                    default=LOGS / "2026-06-23_17-40-45" / "model_5400.pt")
parser.add_argument("--out", type=Path, default=POLICIES / "r1_policy_v2.pt")
args = parser.parse_args()

if not args.ckpt.exists():
    disponibles = sorted(p.name for p in LOGS.glob("*")) if LOGS.exists() else []
    raise SystemExit(
        f"No existe el checkpoint: {args.ckpt}\n"
        f"Runs disponibles en {LOGS}: {disponibles or '(ninguno)'}"
    )

loaded = torch.load(args.ckpt, map_location="cpu", weights_only=False)
actor = loaded["actor_state_dict"]

mlp = nn.Sequential(
    nn.Linear(OBS_DIM, 512), nn.ELU(),
    nn.Linear(512, 256), nn.ELU(),
    nn.Linear(256, 128), nn.ELU(),
    nn.Linear(128, ACT_DIM),
)
# rsl_rl guarda las capas como "mlp.N"; nn.Sequential las quiere como "N"
mlp.load_state_dict({
    f"{i}.{sufijo}": actor[f"mlp.{i}.{sufijo}"]
    for i in (0, 2, 4, 6)
    for sufijo in ("weight", "bias")
})
mlp.eval()

args.out.parent.mkdir(parents=True, exist_ok=True)
torch.jit.script(mlp).save(args.out)
print("OK exportado a:", args.out)

salida = torch.jit.load(args.out)(torch.zeros(1, OBS_DIM))
print(f"Verificacion shape: {tuple(salida.shape)} (debe ser (1, {ACT_DIM}))")
