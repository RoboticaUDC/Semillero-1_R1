#!/usr/bin/env python3
"""Imprime las capas del actor de un checkpoint de rsl_rl.

Sirve para saber que arquitectura hay que reconstruir antes de exportar con
`export_policy_v2.py` (dimensiones de entrada/salida y numero de capas).

Antes se llamaba export_r1_policy.py, pero no exportaba nada: solo inspecciona.

    python scripts/tools/inspeccionar_checkpoint.py
    python scripts/tools/inspeccionar_checkpoint.py --ckpt /ruta/model_8200.pt
"""

# --- rutas del repo, sin depender del CWD ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from amo.paths import THIRD_PARTY
# -------------------------------------------

import argparse

LOGS = THIRD_PARTY / "IsaacLab" / "logs" / "rsl_rl"
CKPT_POR_DEFECTO = LOGS / "2026-06-23_17-40-45" / "model_5400.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", type=Path, default=CKPT_POR_DEFECTO,
                    help=f"checkpoint de rsl_rl (por defecto: {CKPT_POR_DEFECTO})")
args = parser.parse_args()

if not args.ckpt.exists():
    disponibles = sorted(p.name for p in LOGS.glob("*")) if LOGS.exists() else []
    raise SystemExit(
        f"No existe el checkpoint: {args.ckpt}\n"
        f"Runs disponibles en {LOGS}: {disponibles or '(ninguno)'}"
    )

import torch

loaded = torch.load(args.ckpt, map_location="cpu", weights_only=False)
actor = loaded["actor_state_dict"]

print(f"=== CAPAS DEL ACTOR ({args.ckpt.name}) ===")
for k, v in actor.items():
    print(f"  {k}: {tuple(v.shape)}")
print("=== FIN ===")
