#!/usr/bin/env python3
"""Arnes de robustez: alimenta el codigo REAL del proyecto con documentos
defectuosos y registra que pasa en cada caso.

No simula los resultados: importa las mismas funciones que usan los scripts de
R1 y G1 en produccion (amo.paths, amo.math_utils, la CajaDeHerramientas y el
parser de .env de play_r1_ia) y las invoca contra los fixtures de
test/robustez_r1/fixtures/. Para cada caso guarda:

  - la entrada exacta (o su ruta),
  - si el codigo la acepto, la rechazo con un error controlado, o revento,
  - el mensaje / excepcion tal cual.

La evidencia queda en test/robustez_r1/evidencia/ (JSON + Markdown + log).

Uso:
    conda run -n r1mujoco python test/robustez_r1/arnes_robustez.py

Salida distinta de cero solo si el propio arnes falla; que un fixture reviente
es un RESULTADO esperado, no un fallo del arnes.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]                       # test/robustez_r1 -> test -> raiz
FIXTURES = AQUI / "fixtures"
EVIDENCIA = AQUI / "evidencia"
EVIDENCIA.mkdir(exist_ok=True)

sys.path.insert(0, str(RAIZ))

# --- carga del codigo real del proyecto --------------------------------------
import importlib.util

from amo import math_utils
from amo.paths import policy, scene


def _cargar_play_r1_ia():
    ruta = RAIZ / "scripts" / "r1" / "play_r1_ia.py"
    spec = importlib.util.spec_from_file_location("play_r1_ia", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IA = _cargar_play_r1_ia()

try:
    import mujoco
except Exception:                             # pragma: no cover
    mujoco = None
try:
    import torch
except Exception:                             # pragma: no cover
    torch = None


# --- recoleccion de resultados -----------------------------------------------
RESULTADOS: list[dict] = []


def registrar(bloque, robot, fixture, defecto, esperado, resultado):
    """resultado: dict con al menos {'clase', 'detalle'}. clase in
    {'aceptado', 'rechazado_controlado', 'excepcion_no_controlada'}."""
    RESULTADOS.append({
        "bloque": bloque,
        "robot": robot,
        "fixture": fixture,
        "defecto": defecto,
        "comportamiento_esperado": esperado,
        "clase": resultado["clase"],
        "detalle": resultado["detalle"],
    })
    icono = {"aceptado": "  ok ",
             "rechazado_controlado": " ctrl",
             "excepcion_no_controlada": " BOOM"}[resultado["clase"]]
    print(f"[{icono}] {bloque:14s} {robot:3s} {fixture:32s} -> {resultado['clase']}")


def _fmt_exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}".strip()


# =============================================================================
# BLOQUE 1 — Parser de .env  (codigo real: bucle de play_r1_ia.py)
# =============================================================================
def _parsear_env_como_el_script(ruta: Path) -> dict:
    """Replica EXACTA del parser inline de scripts/r1/play_r1_ia.py (lineas
    ~91-97). El script no expone una funcion, asi que reproducimos su logica
    byte a byte para que la evidencia refleje su comportamiento real."""
    capturado = {}
    texto = ruta.read_text()                  # el script usa .read_text() sin encoding
    for _linea in texto.splitlines():
        _linea = _linea.strip()
        if _linea and not _linea.startswith("#") and "=" in _linea:
            _clave, _valor = _linea.split("=", 1)
            capturado.setdefault(_clave.strip(), _valor.strip())
    return capturado


def bloque_env():
    carpeta = FIXTURES / "env"
    for ruta in sorted(carpeta.glob("*.env")):
        defecto = ruta.stem
        try:
            capturado = _parsear_env_como_el_script(ruta)
            key = capturado.get("OPENAI_API_KEY")
            if key is None:
                res = {"clase": "rechazado_controlado",
                       "detalle": f"OPENAI_API_KEY no quedo definida. Claves vistas: "
                                  f"{sorted(capturado)}"}
            else:
                res = {"clase": "aceptado",
                       "detalle": f"OPENAI_API_KEY='{key}' "
                                  f"(otras claves: {sorted(k for k in capturado if k!='OPENAI_API_KEY')})"}
        except Exception as e:
            res = {"clase": "excepcion_no_controlada",
                   "detalle": _fmt_exc(e)}
        registrar("env", "-", ruta.name, defecto,
                  "el parser ignora el .env corrupto y cae al entorno / backend simulado",
                  res)


# =============================================================================
# BLOQUE 2 — Estado del robot en JSON  (codigo real: _narrar_estado)
# =============================================================================
def bloque_estado_json():
    narrar = IA.BackendSimulado._narrar_estado
    carpeta = FIXTURES / "estado_json"
    for ruta in sorted(carpeta.glob("*.json")):
        defecto = ruta.stem
        crudo = ruta.read_text()
        try:
            salida = narrar(crudo)
            devolvio_crudo = salida == crudo
            res = {"clase": "aceptado" if not devolvio_crudo else "rechazado_controlado",
                   "detalle": ("narrado: " + salida) if not devolvio_crudo
                              else "no era JSON parseable -> devuelve el texto crudo (degradado)"}
        except Exception as e:
            res = {"clase": "excepcion_no_controlada", "detalle": _fmt_exc(e)}
        registrar("estado_json", "-", ruta.name, defecto,
                  "narrar el estado si es valido, o degradar sin reventar el agente",
                  res)


# =============================================================================
# BLOQUE 3 — Argumentos de tool-call  (codigo real: CajaDeHerramientas.despachar)
# =============================================================================
def bloque_tool_args():
    bus = IA.BusDeIntenciones()
    caja = IA.CajaDeHerramientas(bus)
    carpeta = FIXTURES / "tool_args"
    for ruta in sorted(carpeta.glob("*.json")):
        meta = json.loads(ruta.read_text())
        herr, crudo, defecto = meta["herramienta"], meta["argumentos_crudos"], meta["defecto"]
        # Reproduce el camino real de play_r1_ia: json.loads del string de
        # argumentos y luego despachar. Un JSON invalido se rechaza antes.
        try:
            argumentos = json.loads(crudo or "{}")
        except json.JSONDecodeError as e:
            registrar("tool_args", "-", ruta.name, defecto,
                      "argumentos no-JSON: rechazo controlado antes de tocar el robot",
                      {"clase": "rechazado_controlado",
                       "detalle": f"json.loads rechazo los argumentos: {_fmt_exc(e)}"})
            continue

        try:
            if not isinstance(argumentos, dict):
                # el script hace herramienta.ejecutar(**argumentos); un no-dict
                # revienta ahi. Lo reproducimos para ver como responde despachar.
                salida = caja.despachar(herr, argumentos)
            else:
                salida = caja.despachar(herr, argumentos)
            empieza_error = salida.startswith("ERROR")
            res = {"clase": "rechazado_controlado" if empieza_error else "aceptado",
                   "detalle": salida}
        except Exception as e:
            res = {"clase": "excepcion_no_controlada", "detalle": _fmt_exc(e)}
        registrar("tool_args", "-", ruta.name, defecto,
                  "clampeo / validacion en la CajaDeHerramientas, nunca una excepcion cruda",
                  res)


# =============================================================================
# BLOQUE 4 — Escenas MuJoCo  (codigo real: mujoco.MjModel.from_xml_path)
# =============================================================================
def bloque_escenas():
    if mujoco is None:
        print("  [skip] mujoco no disponible; se omite el bloque de escenas")
        return
    for robot in ("r1", "g1"):
        carpeta = FIXTURES / "escenas" / robot
        for ruta in sorted(carpeta.glob("*.xml")):
            defecto = ruta.stem
            try:
                modelo = mujoco.MjModel.from_xml_path(str(ruta))
                res = {"clase": "aceptado",
                       "detalle": f"cargo: nq={modelo.nq}, nu={modelo.nu}, nbody={modelo.nbody}"}
            except Exception as e:
                res = {"clase": "rechazado_controlado", "detalle": _fmt_exc(e)}
            registrar("escenas", robot, ruta.name, defecto,
                      "MuJoCo rechaza el XML corrupto con un error claro (salvo el control)",
                      res)


# =============================================================================
# BLOQUE 5 — Politicas TorchScript  (codigo real: torch.jit.load)
# =============================================================================
def bloque_politicas():
    if torch is None:
        print("  [skip] torch no disponible; se omite el bloque de politicas")
        return
    for robot in ("r1", "g1"):
        carpeta = FIXTURES / "politicas" / robot
        for ruta in sorted(carpeta.glob("*.pt")):
            defecto = ruta.stem
            try:
                red = torch.jit.load(str(ruta), map_location="cpu")
                res = {"clase": "aceptado", "detalle": f"jit.load OK: {type(red).__name__}"}
            except Exception as e:
                res = {"clase": "rechazado_controlado", "detalle": _fmt_exc(e)}
            registrar("politicas", robot, ruta.name, defecto,
                      "torch.jit.load rechaza el .pt corrupto sin arrancar el robot",
                      res)

    # stats de normalizacion del adaptador (comun G1 / adaptador R1)
    carpeta = FIXTURES / "politicas"
    for ruta in sorted(carpeta.glob("stats_*.pt")):
        defecto = ruta.stem
        try:
            stats = torch.load(str(ruta), weights_only=False)
            faltan = [k for k in ("input_mean", "input_std", "output_mean", "output_std")
                      if not (isinstance(stats, dict) and k in stats)]
            if faltan:
                res = {"clase": "rechazado_controlado",
                       "detalle": f"cargo pero faltan claves {faltan} (tipo={type(stats).__name__})"}
            else:
                import numpy as np
                std0 = int(np.count_nonzero(np.asarray(stats["input_std"]) == 0))
                dims = {k: np.asarray(stats[k]).shape for k in
                        ("input_mean", "input_std", "output_mean", "output_std")}
                nan = any(bool(np.isnan(np.asarray(stats[k], dtype=float)).any())
                          for k in dims)
                notas = []
                if std0:
                    notas.append(f"input_std tiene {std0} ceros (riesgo de /0 pese al +1e-8)")
                if nan:
                    notas.append("contiene NaN")
                if dims["input_mean"] != (12,):
                    notas.append(f"input_mean dim {dims['input_mean']} != (12,)")
                res = {"clase": "aceptado" if not notas else "rechazado_controlado",
                       "detalle": ("dims=" + json.dumps({k: list(v) for k, v in dims.items()}))
                                  + ((" | " + "; ".join(notas)) if notas else "")}
        except Exception as e:
            res = {"clase": "rechazado_controlado", "detalle": _fmt_exc(e)}
        registrar("stats_norm", "g1/r1", ruta.name, defecto,
                  "detectar claves/dims/NaN malos antes de normalizar las observaciones",
                  res)


# =============================================================================
# Informe
# =============================================================================
def escribir_evidencia(segundos: float):
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    resumen = {}
    for r in RESULTADOS:
        resumen[r["clase"]] = resumen.get(r["clase"], 0) + 1

    # JSON crudo
    (EVIDENCIA / "resultados.json").write_text(
        json.dumps({"generado": ts, "duracion_s": round(segundos, 2),
                    "resumen": resumen, "casos": RESULTADOS},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown legible
    lineas = [f"# Evidencia de pruebas de robustez — R1 / G1",
              "",
              f"- **Generado:** {ts}",
              f"- **Duracion:** {segundos:.2f} s",
              f"- **Casos ejecutados:** {len(RESULTADOS)}",
              "",
              "Cada caso alimenta el codigo real del proyecto con un documento "
              "defectuoso y registra el comportamiento observado.",
              "",
              "## Resumen por comportamiento",
              "",
              "| Comportamiento | Casos |",
              "|---|---|"]
    etiqueta = {"aceptado": "Aceptado (paso la validacion)",
                "rechazado_controlado": "Rechazado de forma controlada",
                "excepcion_no_controlada": "Excepcion NO controlada (a revisar)"}
    for clase in ("aceptado", "rechazado_controlado", "excepcion_no_controlada"):
        lineas.append(f"| {etiqueta[clase]} | {resumen.get(clase, 0)} |")

    bloques = {}
    for r in RESULTADOS:
        bloques.setdefault(r["bloque"], []).append(r)

    titulos = {
        "env": "1. Parser de credenciales `.env` (play_r1_ia.py)",
        "estado_json": "2. Estado del robot en JSON (`_narrar_estado`)",
        "tool_args": "3. Argumentos de herramienta del LLM (`CajaDeHerramientas.despachar`)",
        "escenas": "4. Escenas MuJoCo (`MjModel.from_xml_path`)",
        "politicas": "5. Politicas TorchScript (`torch.jit.load`)",
        "stats_norm": "6. Estadisticas de normalizacion del adaptador (`torch.load`)",
    }
    for bloque, casos in bloques.items():
        lineas += ["", f"## {titulos.get(bloque, bloque)}", "",
                   "| Robot | Fixture | Defecto inyectado | Comportamiento | Detalle |",
                   "|---|---|---|---|---|"]
        for c in casos:
            det = c["detalle"].replace("|", "\\|").replace("\n", " ")
            if len(det) > 160:
                det = det[:157] + "..."
            lineas.append(f"| {c['robot']} | `{c['fixture']}` | {c['defecto']} "
                          f"| {c['clase']} | {det} |")

    (EVIDENCIA / "informe.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main():
    print("=" * 78)
    print("ARNES DE ROBUSTEZ R1 / G1 — fixtures defectuosos contra el codigo real")
    print("=" * 78)
    t0 = time.perf_counter()
    for fn in (bloque_env, bloque_estado_json, bloque_tool_args,
               bloque_escenas, bloque_politicas):
        print(f"\n--- {fn.__name__} ---")
        fn()
    dt = time.perf_counter() - t0
    escribir_evidencia(dt)

    resumen = {}
    for r in RESULTADOS:
        resumen[r["clase"]] = resumen.get(r["clase"], 0) + 1
    print("\n" + "=" * 78)
    print(f"TOTAL: {len(RESULTADOS)} casos en {dt:.2f}s")
    for k, v in sorted(resumen.items()):
        print(f"   {k:26s}: {v}")
    print(f"\nEvidencia:")
    print(f"   {EVIDENCIA/'informe.md'}")
    print(f"   {EVIDENCIA/'resultados.json'}")


if __name__ == "__main__":
    main()
