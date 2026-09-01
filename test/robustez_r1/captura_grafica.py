#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captura de evidencia GRAFICA: corre los scripts REALES de R1 en modo headless
y guarda lo que se ve, sin depender de que tengas una ventana abierta.

Como funciona
-------------
Los scripts (banda_estabilidad_r1, play_r1_camina_brazos, play_r1_isaac,
banda_r1) abren un visor con `mujoco.viewer.launch_passive`, que necesita una
ventana. Aqui parcheamos esa llamada por un visor "stub" que, en vez de abrir
ventana, renderiza la MISMA escena a memoria y guarda frames a PNG + GIF. El
resto del script (fisica, politica, control PD, gestos) corre TAL CUAL.

Para cada script se hace:
  1) Corrida VALIDA (runtime): entradas buenas, N pasos. Se guarda un GIF y la
     altura de la pelvis paso a paso -> evidencia de si el robot se mantiene o
     se cae ("no camina bien").
  2) Corrida con FIXTURE ROTO: se apunta el script a un XML / politica corrupta
     y se captura el crash con su traceback -> evidencia de que revienta al
     cargar.

Salida: test/robustez_r1/evidencia/grafica/  (PNGs, GIFs, metricas, informe.md)

Uso (necesita el display :1 para el contexto GL, pero NO abre ventanas):
    DISPLAY=:1 MUJOCO_GL=glfw conda run -n r1mujoco \
        python test/robustez_r1/captura_grafica.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
FIXTURES = AQUI / "fixtures"
SALIDA = AQUI / "evidencia" / "grafica"
SALIDA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(RAIZ))

import mujoco
import mujoco.viewer
import torch
from amo.paths import policy, scene

# altura de pelvis por debajo de la cual consideramos que el robot se cayo
UMBRAL_CAIDA_M = 0.45
PASOS = 600                    # pasos de fisica por corrida valida (~1.2 s a 500 Hz)
CADA = 10                      # guarda 1 frame cada CADA pasos
RES = (360, 480)               # alto, ancho del render

INFORME: list[dict] = []


# ---------------------------------------------------------------------------
# Visor stub: reemplaza la ventana por un renderer offscreen que graba frames.
# ---------------------------------------------------------------------------
class _Cam:
    def __init__(self):
        self.lookat = np.zeros(3)
        self.distance = 3.0
        self.elevation = -20.0
        self.azimuth = 180.0


class VisorCaptura:
    """Se hace pasar por el objeto de mujoco.viewer.launch_passive."""

    def __init__(self, model, data, max_pasos=PASOS, **_):
        self.model, self.data = model, data
        self.cam = _Cam()
        self._n = 0
        self._max = max_pasos
        self._rend = mujoco.Renderer(model, RES[0], RES[1])
        self._camv = mujoco.MjvCamera()
        self._camv.distance = model.stat.extent * 1.6
        self._camv.elevation = -20.0
        self._camv.azimuth = 135.0
        self.frames: list[np.ndarray] = []
        self.altura_pelvis: list[float] = []

    def is_running(self):
        self._n += 1
        # registra altura de la base en cada paso
        try:
            self.altura_pelvis.append(float(self.data.qpos[2]))
        except Exception:
            pass
        return self._n <= self._max

    def sync(self):
        if self._n % CADA == 0:
            try:
                self._camv.lookat[:] = self.data.qpos[:3]
                self._rend.update_scene(self.data, camera=self._camv)
                self.frames.append(self._rend.render().copy())
            except Exception:
                pass

    def close(self):
        try:
            self._rend.close()
        except Exception:
            pass


def _instalar_stub(max_pasos=PASOS):
    """Devuelve la lista compartida donde el stub guardara su instancia."""
    caja = {}

    def _fabrica(model, data, **kw):
        v = VisorCaptura(model, data, max_pasos=max_pasos, **kw)
        caja["visor"] = v
        return v

    mujoco.viewer.launch_passive = _fabrica
    return caja


def _cargar_modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _guardar(nombre_corrida, visor):
    """Guarda GIF + primer/ultimo PNG + metricas de altura. Devuelve dict."""
    d = SALIDA / nombre_corrida
    d.mkdir(parents=True, exist_ok=True)
    info = {"frames": len(visor.frames) if visor else 0}

    if visor and visor.frames:
        imgs = [Image.fromarray(f) for f in visor.frames]
        imgs[0].save(d / "primer_frame.png")
        imgs[-1].save(d / "ultimo_frame.png")
        imgs[0].save(d / "animacion.gif", save_all=True,
                     append_images=imgs[1:], duration=80, loop=0)
        info["gif"] = str((d / "animacion.gif").relative_to(AQUI))

    if visor and visor.altura_pelvis:
        z = np.asarray(visor.altura_pelvis)
        cayo = bool(z.min() < UMBRAL_CAIDA_M)
        info.update({
            "z_inicial_m": round(float(z[0]), 3),
            "z_final_m": round(float(z[-1]), 3),
            "z_min_m": round(float(z.min()), 3),
            "se_cayo": cayo,
            "pasos": int(z.size),
        })
        (d / "altura_pelvis.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def corrida_valida(clave, titulo, construir, max_pasos=PASOS,
                   preparar=None, modo="valida"):
    """construir() -> instancia del env ya lista para .run().
    preparar(env) permite inyectar un comando (p.ej. caminar) antes de run()."""
    print(f"\n=== [{modo.upper()}] {clave}: {titulo} ===")
    caja = _instalar_stub(max_pasos)
    resultado = {"corrida": clave, "modo": modo, "titulo": titulo}
    try:
        env = construir()
        if preparar is not None:
            preparar(env)
        env.run()
        visor = caja.get("visor")
        info = _guardar(clave, visor)
        resultado.update({"clase": "ejecuto", **info})
        veredicto = ("SE CAYO / inestable" if info.get("se_cayo")
                     else "se mantuvo de pie")
        resultado["veredicto"] = veredicto
        print(f"    -> {info.get('frames',0)} frames, "
              f"z_min={info.get('z_min_m')} m -> {veredicto}")
    except BaseException as e:  # noqa: BLE001  (queremos capturar TODO)
        tb = traceback.format_exc()
        (SALIDA / f"{clave}_TRACEBACK.txt").write_text(tb, encoding="utf-8")
        resultado.update({"clase": "crasheo",
                          "error": f"{type(e).__name__}: {e}",
                          "traceback": f"{clave}_TRACEBACK.txt"})
        # aun asi intenta guardar los frames que alcanzo a capturar
        info = _guardar(clave, caja.get("visor"))
        resultado.update(info)
        print(f"    -> CRASH: {type(e).__name__}: {e}")
    INFORME.append(resultado)
    return resultado


def corrida_rota(clave, titulo, construir, defecto):
    """Igual, pero esperamos que reviente al cargar el fixture roto."""
    print(f"\n=== [ROTA]   {clave}: {titulo} ===")
    caja = _instalar_stub(60)
    resultado = {"corrida": clave, "modo": "fixture_roto",
                 "titulo": titulo, "defecto": defecto}
    try:
        env = construir()
        env.run()
        info = _guardar(clave, caja.get("visor"))
        resultado.update({"clase": "ejecuto_inesperado", **info})
        print(f"    -> cargo sin reventar (inesperado)")
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        (SALIDA / f"{clave}_TRACEBACK.txt").write_text(tb, encoding="utf-8")
        resultado.update({"clase": "crasheo_esperado",
                          "error": f"{type(e).__name__}: {e}",
                          "traceback": f"{clave}_TRACEBACK.txt"})
        print(f"    -> CRASH esperado: {type(e).__name__}: {str(e)[:90]}")
    INFORME.append(resultado)
    return resultado


# ---------------------------------------------------------------------------
# Definicion de las corridas por script
# ---------------------------------------------------------------------------
def main():
    XML_R1 = scene("r1")
    XML_ROTO_MALLA = str(FIXTURES / "escenas/r1/02_malla_inexistente.xml")
    XML_ROTO_CERRADO = str(FIXTURES / "escenas/r1/01_xml_mal_cerrado.xml")
    PT_ROTO = str(FIXTURES / "politicas/r1/03_truncado.pt")

    # --- 1. banda_estabilidad_r1 (solo PD, sin politica) ------------------
    be = _cargar_modulo("banda_estabilidad_r1",
                        RAIZ / "scripts/r1/banda_estabilidad_r1.py")
    corrida_valida("estabilidad_valida",
                   "banda_estabilidad_r1: controlador PD, entrada valida",
                   lambda: be.R1StableEnv())
    corrida_rota("estabilidad_xml_roto",
                 "banda_estabilidad_r1 con XML de malla inexistente",
                 lambda: be.R1StableEnv(xml_path=XML_ROTO_MALLA),
                 "escena con malla STL que no existe")

    # --- 2. play_r1_camina_brazos -----------------------------------------
    cb = _cargar_modulo("play_r1_camina_brazos",
                        RAIZ / "scripts/r1/play_r1_camina_brazos.py")
    corrida_valida("camina_valida",
                   "play_r1_camina_brazos: politica r1_v2, entrada valida",
                   lambda: cb.R1CaminaBrazos(device="cpu"))
    corrida_valida("camina_estres",
                   "play_r1_camina_brazos: comando de caminar vx=0.9 (limite)",
                   lambda: cb.R1CaminaBrazos(device="cpu"),
                   max_pasos=1800, modo="estres",
                   preparar=lambda e: e.command.__setitem__(slice(None),
                                                             np.array([0.9, 0.0, 0.0], np.float32)))
    corrida_rota("camina_politica_rota",
                 "play_r1_camina_brazos con politica .pt truncada",
                 lambda: cb.R1CaminaBrazos(policy_path=PT_ROTO, device="cpu"),
                 "politica TorchScript truncada a la mitad")

    # --- 3. play_r1_isaac -------------------------------------------------
    iz = _cargar_modulo("play_r1_isaac",
                        RAIZ / "scripts/r1/play_r1_isaac.py")
    corrida_valida("isaac_valida",
                   "play_r1_isaac: politica r1_v2, entrada valida",
                   lambda: iz.R1IsaacPolicyEnv(device="cpu"))
    corrida_valida("isaac_estres",
                   "play_r1_isaac: comando de caminar vx=0.9 (limite)",
                   lambda: iz.R1IsaacPolicyEnv(device="cpu"),
                   max_pasos=1800, modo="estres",
                   preparar=lambda e: e.command.__setitem__(slice(None),
                                                            np.array([0.9, 0.0, 0.0], np.float32)))
    corrida_rota("isaac_xml_roto",
                 "play_r1_isaac con XML mal cerrado",
                 lambda: iz.R1IsaacPolicyEnv(xml_path=XML_ROTO_CERRADO, device="cpu"),
                 "XML de escena sin etiqueta de cierre")

    # --- 4. banda_r1 (politica AMO + adaptador) ---------------------------
    try:
        br = _cargar_modulo("banda_r1", RAIZ / "scripts/r1/banda_r1.py")
        policy_jit = torch.jit.load(br.RUTA_POLITICA, map_location="cpu")
        corrida_valida("banda_r1_valida",
                       "banda_r1: AMO+adaptador sobre R1, entrada valida",
                       lambda: br.HumanoidEnv(policy_jit=policy_jit,
                                              robot_type="r1", device="cpu"))
        corrida_valida("banda_r1_estres",
                       "banda_r1: comando de caminar vx=0.9 (limite)",
                       lambda: br.HumanoidEnv(policy_jit=policy_jit,
                                              robot_type="r1", device="cpu"),
                       max_pasos=1800, modo="estres",
                       preparar=lambda e: e.state.commands.__setitem__(0, 0.9))
        # fixture roto: apuntar los stats del adaptador a uno sin la clave
        stats_roto = str(FIXTURES / "politicas/stats_01_falta_input_std.pt")
        orig = br.RUTA_POLITICA_ADAPTADORA_ESTADOS
        br.RUTA_POLITICA_ADAPTADORA_ESTADOS = stats_roto
        corrida_rota("banda_r1_stats_rotos",
                     "banda_r1 con stats de normalizacion sin 'input_std'",
                     lambda: br.HumanoidEnv(policy_jit=policy_jit,
                                            robot_type="r1", device="cpu"),
                     "adapter_norm_stats sin la clave input_std")
        br.RUTA_POLITICA_ADAPTADORA_ESTADOS = orig
    except BaseException as e:  # noqa: BLE001
        INFORME.append({"corrida": "banda_r1", "modo": "-",
                        "clase": "no_ejecutable",
                        "error": f"{type(e).__name__}: {e}"})
        print(f"\n[banda_r1] no ejecutable: {type(e).__name__}: {e}")

    _escribir_informe()


def _escribir_informe():
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    (SALIDA / "resultados_grafica.json").write_text(
        json.dumps({"generado": ts, "corridas": INFORME},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    L = ["# Evidencia grafica — scripts R1 en runtime", "",
         f"- **Generado:** {ts}",
         f"- **Corridas:** {len(INFORME)}",
         "",
         "Cada corrida ejecuta el script REAL en headless (visor sustituido por "
         "un renderer offscreen). Las validas guardan un GIF y la altura de la "
         "pelvis; las de fixture roto capturan el crash.", "",
         "## Corridas de runtime (valida = quieto, estres = caminando)", "",
         "| Corrida | Modo | Frames | z ini | z min | z fin | Veredicto | GIF |",
         "|---|---|---|---|---|---|---|---|"]
    for r in INFORME:
        if r.get("modo") in ("valida", "estres"):
            L.append(f"| `{r['corrida']}` | {r['modo']} | {r.get('frames','-')} "
                     f"| {r.get('z_inicial_m','-')} | {r.get('z_min_m','-')} "
                     f"| {r.get('z_final_m','-')} | {r.get('veredicto', r.get('clase','-'))} "
                     f"| {r.get('gif','-')} |")
    L += ["", "## Corridas con fixture roto (crash esperado)", "",
          "| Corrida | Defecto | Resultado | Error |",
          "|---|---|---|---|"]
    for r in INFORME:
        if r.get("modo") in ("fixture_roto", "-"):
            err = (r.get("error", "")).replace("|", "\\|")
            if len(err) > 100:
                err = err[:97] + "..."
            L.append(f"| `{r['corrida']}` | {r.get('defecto','-')} "
                     f"| {r.get('clase','-')} | {err} |")
    (SALIDA / "informe_grafica.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nEvidencia grafica en: {SALIDA}")
    print(f"   informe: {SALIDA/'informe_grafica.md'}")


if __name__ == "__main__":
    main()
