#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captura de evidencia GRAFICA del G1 (equivalente a captura_grafica.py del R1).

Los tres scripts del G1 arrancan distinto, asi que cada uno se maneja aparte:

  play_amo_stable.py  -> marcha PROGRAMADA (sin red). Usa `mujoco_viewer`; aqui
                         se sustituye por un stub offscreen. Camina de verdad.
  banda_v2_1.py       -> controlador de equilibrio. Usa `mujoco.viewer`.
  play_amo.py         -> politica AMO real, pero EXIGE CUDA. En una maquina sin
                         GPU se documenta el crash del guard de CUDA.

Salida: test/robustez_r1/evidencia/grafica_g1/  (PNGs, GIFs, metricas, informe).

Uso:
    DISPLAY=:1 MUJOCO_GL=glfw conda run -n r1mujoco \
        python test/robustez_r1/captura_grafica_g1.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
FIXTURES = AQUI / "fixtures"
SALIDA = AQUI / "evidencia" / "grafica_g1"
SALIDA.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RAIZ))

import mujoco
import mujoco.viewer

UMBRAL_CAIDA_M = 0.45
RES = (360, 480)
CADA = 10
INFORME: list[dict] = []


# ---------------------------------------------------------------------------
# Renderer offscreen compartido: acumula frames + altura de la base.
# ---------------------------------------------------------------------------
class Grabador:
    def __init__(self, model):
        self._r = mujoco.Renderer(model, RES[0], RES[1])
        self._c = mujoco.MjvCamera()
        self._c.distance = model.stat.extent * 1.5
        self._c.elevation = -20.0
        self._c.azimuth = 135.0
        self.frames: list[np.ndarray] = []
        self.altura: list[float] = []
        self._n = 0

    def tick(self, data):
        self._n += 1
        try:
            self.altura.append(float(data.qpos[2]))
        except Exception:
            pass
        if self._n % CADA == 0:
            try:
                self._c.lookat[:] = data.qpos[:3]
                self._r.update_scene(data, camera=self._c)
                self.frames.append(self._r.render().copy())
            except Exception:
                pass

    def cerrar(self):
        try:
            self._r.close()
        except Exception:
            pass


def _cam_stub():
    c = types.SimpleNamespace(lookat=np.zeros(3), distance=3.0,
                              elevation=-20.0, azimuth=180.0)
    return c


def _cargar(nombre, rel):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _guardar(clave, grab):
    d = SALIDA / clave
    d.mkdir(parents=True, exist_ok=True)
    info = {"frames": len(grab.frames) if grab else 0}
    if grab and grab.frames:
        imgs = [Image.fromarray(f) for f in grab.frames]
        imgs[0].save(d / "primer_frame.png")
        imgs[-1].save(d / "ultimo_frame.png")
        imgs[0].save(d / "animacion.gif", save_all=True,
                     append_images=imgs[1:], duration=80, loop=0)
        info["gif"] = str((d / "animacion.gif").relative_to(AQUI))
    if grab and grab.altura:
        z = np.asarray(grab.altura)
        info.update({"z_inicial_m": round(float(z[0]), 3),
                     "z_final_m": round(float(z[-1]), 3),
                     "z_min_m": round(float(z.min()), 3),
                     "se_cayo": bool(z.min() < UMBRAL_CAIDA_M),
                     "pasos": int(z.size)})
        (d / "altura_pelvis.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _reg(res):
    INFORME.append(res)
    et = {"ejecuto": " ok ", "crasheo": "BOOM", "crasheo_esperado": "crash-esp",
          "no_ejecutable": "n/a "}.get(res.get("clase"), "?")
    print(f"[{et}] {res['corrida']:22s} -> {res.get('clase')} "
          f"{res.get('veredicto', res.get('error',''))[:70]}")


# ---------------------------------------------------------------------------
# 1. play_amo_stable — marcha programada (mujoco_viewer stub)
# ---------------------------------------------------------------------------
def corr_play_amo_stable(clave, titulo, xml_roto=None, caminar=True, pasos=400):
    import glfw
    grab_box = {}

    class StubViewer:
        def __init__(self, model, data, *a, **k):
            self.model, self.data, self.window = model, data, object()
            grab_box["g"] = Grabador(model)

        def render(self):
            grab_box["g"].tick(self.data)

        def close(self):
            grab_box["g"].cerrar()

    fake = types.ModuleType("mujoco_viewer")
    fake.MujocoViewer = StubViewer
    sys.modules["mujoco_viewer"] = fake
    glfw.set_key_callback = lambda *a, **k: None
    cnt = {"n": 0}

    def _wsc(_w):
        cnt["n"] += 1
        return cnt["n"] > pasos
    glfw.window_should_close = _wsc

    res = {"corrida": clave, "titulo": titulo,
           "modo": "estres" if caminar else "valida"}
    try:
        mod = _cargar("play_amo_stable", "scripts/g1/play_amo_stable.py")
        if xml_roto:
            # fuerza la escena rota reemplazando scene() dentro del modulo
            mod.scene = lambda *_a, **_k: xml_roto
        if caminar:
            mod.commands["vx"] = 0.5
        env = mod.HumanoidEnv()
        env.run()
        info = _guardar(clave, grab_box.get("g"))
        res.update({"clase": "ejecuto", **info})
        res["veredicto"] = ("SE CAYO / inestable" if info.get("se_cayo")
                            else "camina/se mantiene de pie")
    except BaseException as e:  # noqa: BLE001
        _tb(clave, e)
        info = _guardar(clave, grab_box.get("g"))
        res.update({"clase": ("crasheo_esperado" if xml_roto else "crasheo"),
                    "error": f"{type(e).__name__}: {e}", **info})
    finally:
        sys.modules.pop("mujoco_viewer", None)
    _reg(res)


# ---------------------------------------------------------------------------
# 2. banda_v2_1 — equilibrio (mujoco.viewer.launch_passive stub)
# ---------------------------------------------------------------------------
def corr_banda_v2_1(clave, titulo, xml_roto=None, pasos=400):
    grab_box = {}

    class StubPassive:
        def __init__(self, model, data, **k):
            self.model, self.data, self.cam = model, data, _cam_stub()
            self._n = 0
            grab_box["g"] = Grabador(model)

        def is_running(self):
            self._n += 1
            return self._n <= pasos

        def sync(self):
            grab_box["g"].tick(self.data)

        def close(self):
            grab_box["g"].cerrar()

    mujoco.viewer.launch_passive = lambda model, data, **k: StubPassive(model, data, **k)

    res = {"corrida": clave, "titulo": titulo,
           "modo": "fixture_roto" if xml_roto else "valida"}
    try:
        mod = _cargar("banda_v2_1", "scripts/g1/banda_v2_1.py")
        if xml_roto:
            mod.PATH_SCENE = xml_roto
        env = mod.HumanoidEnv()
        env.run()
        info = _guardar(clave, grab_box.get("g"))
        res.update({"clase": "ejecuto", **info})
        res["veredicto"] = ("SE CAYO / inestable" if info.get("se_cayo")
                            else "se mantiene de pie")
    except BaseException as e:  # noqa: BLE001
        _tb(clave, e)
        info = _guardar(clave, grab_box.get("g"))
        res.update({"clase": ("crasheo_esperado" if xml_roto else "crasheo"),
                    "error": f"{type(e).__name__}: {e}", **info})
    _reg(res)


# ---------------------------------------------------------------------------
# 3. play_amo — exige CUDA
# ---------------------------------------------------------------------------
def corr_play_amo(clave, titulo):
    import glfw
    res = {"corrida": clave, "titulo": titulo, "modo": "requisito_gpu"}
    # inyecta un mujoco_viewer y glfw de pega para pasar los imports y llegar
    # al verdadero bloqueo de este script: el requisito de CUDA.
    fake = types.ModuleType("mujoco_viewer")
    fake.MujocoViewer = object
    sys.modules["mujoco_viewer"] = fake
    glfw.set_key_callback = lambda *a, **k: None
    try:
        import torch
        mod = _cargar("play_amo", "scripts/g1/play_amo.py")
        env = mod.HumanoidEnv(policy_path=str(RAIZ / "policies/amo_jit.pt"),
                              adapter_path=str(RAIZ / "policies/adapter_jit.pt"),
                              device="cuda:0")
        res.update({"clase": "ejecuto"})
    except BaseException as e:  # noqa: BLE001
        _tb(clave, e)
        res.update({"clase": "no_ejecutable", "error": f"{type(e).__name__}: {e}"})
    finally:
        sys.modules.pop("mujoco_viewer", None)
    _reg(res)


def _tb(clave, e):
    (SALIDA / f"{clave}_TRACEBACK.txt").write_text(
        traceback.format_exc(), encoding="utf-8")


# ---------------------------------------------------------------------------
def main():
    XML_G1_ROTO = str(FIXTURES / "escenas/g1/01_xml_mal_cerrado.xml")

    print("=== play_amo_stable (marcha programada) ===")
    corr_play_amo_stable("g1_stable_valida",
                         "play_amo_stable quieto (sin comando)", caminar=False)
    corr_play_amo_stable("g1_stable_camina",
                         "play_amo_stable caminando vx=0.5", caminar=True)
    corr_play_amo_stable("g1_stable_xml_roto",
                         "play_amo_stable con XML de escena roto",
                         xml_roto=XML_G1_ROTO, caminar=False, pasos=40)

    print("\n=== banda_v2_1 (equilibrio) ===")
    corr_banda_v2_1("g1_banda_valida", "banda_v2_1 equilibrio, entrada valida")
    corr_banda_v2_1("g1_banda_xml_roto", "banda_v2_1 con XML de escena roto",
                    xml_roto=XML_G1_ROTO, pasos=40)

    print("\n=== play_amo (requiere CUDA) ===")
    corr_play_amo("g1_play_amo_cuda", "play_amo: politica AMO real (exige GPU)")

    _informe()


def _informe():
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    (SALIDA / "resultados_g1.json").write_text(
        json.dumps({"generado": ts, "corridas": INFORME},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    L = ["# Evidencia grafica — scripts G1 en runtime", "",
         f"- **Generado:** {ts}", f"- **Corridas:** {len(INFORME)}", "",
         "Cada corrida ejecuta el script REAL del G1 en headless.", "",
         "## Runtime", "",
         "| Corrida | Modo | Frames | z ini | z min | z fin | Veredicto/Estado | GIF |",
         "|---|---|---|---|---|---|---|---|"]
    for r in INFORME:
        if r.get("clase") in ("ejecuto",):
            L.append(f"| `{r['corrida']}` | {r.get('modo','-')} | {r.get('frames','-')} "
                     f"| {r.get('z_inicial_m','-')} | {r.get('z_min_m','-')} "
                     f"| {r.get('z_final_m','-')} | {r.get('veredicto','-')} "
                     f"| {r.get('gif','-')} |")
    L += ["", "## Crashes / requisitos", "",
          "| Corrida | Estado | Error |", "|---|---|---|"]
    for r in INFORME:
        if r.get("clase") in ("crasheo", "crasheo_esperado", "no_ejecutable"):
            err = (r.get("error", "")).replace("|", "\\|").replace("\n", " ")
            if len(err) > 110:
                err = err[:107] + "..."
            L.append(f"| `{r['corrida']}` | {r['clase']} | {err} |")
    (SALIDA / "informe_g1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nEvidencia G1 en: {SALIDA}\n   informe: {SALIDA/'informe_g1.md'}")


if __name__ == "__main__":
    main()
