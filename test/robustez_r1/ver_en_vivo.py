#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanzador EN VIVO: abre el visor MuJoCo REAL en tu pantalla para reproducir,
uno por uno, los escenarios de la prueba de robustez.

A diferencia de captura_grafica.py (que corre headless y guarda GIFs), aqui SI
se abre la ventana del visor, para que veas el comportamiento en directo.

Escenarios:
  estabilidad        banda_estabilidad_r1 normal (PD, se queda de pie)
  camina             play_r1_camina_brazos normal (usa flechas para caminar)
  isaac              play_r1_isaac normal
  banda_r1           banda_r1 normal (AMO+adaptador; flechas para caminar)
  banda_r1_cae       banda_r1 con comando de caminar inyectado -> SE CAE solo
  estabilidad_roto   banda_estabilidad_r1 con XML de malla inexistente -> CRASH
  camina_roto        play_r1_camina_brazos con politica .pt truncada  -> CRASH
  banda_r1_roto      banda_r1 con stats de normalizacion sin input_std -> CRASH

Uso (en la maquina con pantalla; el display suele ser :0 o :1):
    DISPLAY=:1 conda run -n r1mujoco \
        python test/robustez_r1/ver_en_vivo.py banda_r1_cae

Cierra la ventana (ESC o la X) para terminar cada escenario.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
FIXTURES = AQUI / "fixtures"
sys.path.insert(0, str(RAIZ))


def _mod(nombre, rel):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def estabilidad():
    _mod("be", "scripts/r1/banda_estabilidad_r1.py").R1StableEnv().run()


def camina():
    m = _mod("cb", "scripts/r1/play_r1_camina_brazos.py")
    m.R1CaminaBrazos(device="cpu").run()


def isaac():
    m = _mod("iz", "scripts/r1/play_r1_isaac.py")
    m.R1IsaacPolicyEnv(device="cpu").run()


def banda_r1():
    import torch
    m = _mod("br", "scripts/r1/banda_r1.py")
    pj = torch.jit.load(m.RUTA_POLITICA, map_location="cpu")
    m.HumanoidEnv(policy_jit=pj, robot_type="r1", device="cpu").run()


def banda_r1_cae():
    """Inyecta un comando de caminar y deja que se caiga solo, en vivo."""
    import threading
    import time

    import torch
    m = _mod("br", "scripts/r1/banda_r1.py")
    pj = torch.jit.load(m.RUTA_POLITICA, map_location="cpu")
    env = m.HumanoidEnv(policy_jit=pj, robot_type="r1", device="cpu")

    def _empujar():
        time.sleep(1.5)                     # deja que abra la ventana
        env.state.commands[0] = 0.9         # caminar al frente al limite
        print(">>> comando de caminar inyectado: vx=0.9 (mira como se cae)")

    threading.Thread(target=_empujar, daemon=True).start()
    env.run()


def estabilidad_roto():
    xml = str(FIXTURES / "escenas/r1/02_malla_inexistente.xml")
    _mod("be", "scripts/r1/banda_estabilidad_r1.py").R1StableEnv(xml_path=xml).run()


def camina_roto():
    pt = str(FIXTURES / "politicas/r1/03_truncado.pt")
    m = _mod("cb", "scripts/r1/play_r1_camina_brazos.py")
    m.R1CaminaBrazos(policy_path=pt, device="cpu").run()


def banda_r1_roto():
    import torch
    m = _mod("br", "scripts/r1/banda_r1.py")
    m.RUTA_POLITICA_ADAPTADORA_ESTADOS = str(
        FIXTURES / "politicas/stats_01_falta_input_std.pt")
    pj = torch.jit.load(m.RUTA_POLITICA, map_location="cpu")
    m.HumanoidEnv(policy_jit=pj, robot_type="r1", device="cpu").run()


# --------------------------------------------------------------------------
# G1
# --------------------------------------------------------------------------
def g1_camina():
    """play_amo_stable: marcha programada. Usa W/S/A/D/Q/E para moverlo."""
    m = _mod("pas", "scripts/g1/play_amo_stable.py")
    m.HumanoidEnv().run()


def g1_banda():
    """banda_v2_1: OJO, hoy crashea por un bug real (pose de brazos 8 vs 10)."""
    m = _mod("bv", "scripts/g1/banda_v2_1.py")
    m.HumanoidEnv().run()


def g1_banda_roto():
    m = _mod("bv", "scripts/g1/banda_v2_1.py")
    m.PATH_SCENE = str(FIXTURES / "escenas/g1/01_xml_mal_cerrado.xml")
    m.HumanoidEnv().run()


def g1_play_amo():
    """play_amo: politica AMO real. Requiere GPU con CUDA; en CPU no arranca."""
    import torch
    m = _mod("pa", "scripts/g1/play_amo.py")
    m.HumanoidEnv(policy_path=str(RAIZ / "policies/amo_jit.pt"),
                  adapter_path=str(RAIZ / "policies/adapter_jit.pt"),
                  device="cuda:0").run()


ESCENARIOS = {
    # R1
    "estabilidad": estabilidad, "camina": camina, "isaac": isaac,
    "banda_r1": banda_r1, "banda_r1_cae": banda_r1_cae,
    "estabilidad_roto": estabilidad_roto, "camina_roto": camina_roto,
    "banda_r1_roto": banda_r1_roto,
    # G1
    "g1_camina": g1_camina, "g1_banda": g1_banda,
    "g1_banda_roto": g1_banda_roto, "g1_play_amo": g1_play_amo,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ESCENARIOS:
        print("Escenarios disponibles:")
        for k in ESCENARIOS:
            print("   ", k)
        print(f"\nEjemplo:\n    DISPLAY=:1 conda run -n r1mujoco "
              f"python {Path(__file__).relative_to(RAIZ)} banda_r1_cae")
        sys.exit(1)
    ESCENARIOS[sys.argv[1]]()
