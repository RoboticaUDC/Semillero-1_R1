#!/usr/bin/env python3
"""
play_r1_ia.py — R1 en MuJoCo controlado por lenguaje natural (LLM con API key).

QUE HACE
========
Levanta el R1 con la politica nativa y los gestos de brazo (exactamente el
mismo comportamiento que `play_r1_camina_brazos.py`, del que hereda sin
modificarlo) y le anade encima un agente conversacional:

    tu escribes  ->  "camina despacio hacia adelante 3 segundos y saluda"
    el LLM       ->  decide que herramientas llamar
    el robot     ->  las ejecuta a 50 Hz sin bloquear la simulacion

Tambien contesta preguntas sobre su propio estado ("donde estas?",
"que tan rapido vas?", "que estas haciendo?").

ARQUITECTURA (lo importante)
============================
El bucle de MuJoCo corre a 500 Hz con un presupuesto de 2 ms por paso. Una
llamada a un LLM tarda entre 0.5 y 3 s. Si se llamara desde el bucle, el
control PD se congelaria y el robot se cae. Por eso:

    [hilo consola]  lee stdin -> llama al LLM -> ejecuta herramientas
                                       |
                                       v
                              BusDeIntenciones (queue.Queue)
                                       |
                                       v
    [hilo principal]  bucle MuJoCo 500 Hz: drena la cola, nunca bloquea

El hilo del robot jamas toca la red. El hilo del LLM jamas toca `data.qpos`
directamente: lee un snapshot que publica el bucle a 10 Hz.

EL BACKEND DE LLM ES INTERCAMBIABLE
===================================
Todo lo especifico de un proveedor esta aislado en una subclase de
`BackendLLM`. Hoy vienen cuatro:

  * `simulado`  — sin API key, sin red, reglas por palabras clave.
                  Sirve para demostrar el sistema completo funcionando YA.
  * `openai`    — GPT via el SDK oficial (`pip install openai`).
  * `anthropic` — Claude via el SDK oficial (`pip install anthropic`).
  * `plantilla` — esqueleto marcado con TODO para enchufar CUALQUIER otra API.
                  Son 3 metodos. Ver la clase `BackendPlantilla`.

La key sale del `.env` de la raiz del repo (ver mas abajo), que esta en
.gitignore; las variables ya exportadas en el entorno mandan sobre el .env.

USO
===
    conda activate r1mujoco

    # 1. Demo sin API key (funciona tal cual, ahora mismo):
    python scripts/r1/play_r1_ia.py

    # 2. Con ChatGPT:
    pip install openai
    export OPENAI_API_KEY="sk-proj-..."
    python scripts/r1/play_r1_ia.py --backend openai

    # 3. Con Claude:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    python scripts/r1/play_r1_ia.py --backend anthropic

    # 4. Probar solo el cableado del LLM, sin levantar MuJoCo:
    python scripts/r1/play_r1_ia.py --backend openai --sin-sim

TECLADO
=======
Sigue funcionando TODO lo de `play_r1_camina_brazos.py` (flechas, Q/E,
ESPACIO, R, F1-F7, ESC). El teclado tiene prioridad: ESPACIO cancela
cualquier cosa que haya mandado el LLM. Es el freno de emergencia.

NO MODIFICA NINGUN ARCHIVO EXISTENTE.
"""

# --- hace importable el paquete `amo` sin instalar nada ---
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# ----------------------------------------------------------

# --- carga la API key desde el .env de la raiz del repo (si existe) ---
# El .env esta en .gitignore: la key nunca viaja en el codigo ni en git.
# Las variables ya exportadas en el entorno tienen prioridad.
import os as _os

_RUTA_ENV = Path(__file__).resolve().parents[2] / ".env"
if _RUTA_ENV.exists():
    for _linea in _RUTA_ENV.read_text().splitlines():
        _linea = _linea.strip()
        if _linea and not _linea.startswith("#") and "=" in _linea:
            _clave, _valor = _linea.split("=", 1)
            _os.environ.setdefault(_clave.strip(), _valor.strip())
# ----------------------------------------------------------------------

import argparse
import importlib.util
import json
import os
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from amo.math_utils import quat_to_euler


# =============================================================================
# CARGA DEL ENTORNO BASE
# =============================================================================
# `play_r1_camina_brazos.py` es un script suelto, no un modulo del paquete, asi
# que lo cargamos por ruta. Lo importamos en vez de copiar su codigo para que
# cualquier arreglo que le hagas alli se refleje aqui automaticamente.

_RUTA_BASE = Path(__file__).resolve().parent / "play_r1_camina_brazos.py"


def _cargar_entorno_base():
    if not _RUTA_BASE.exists():
        raise FileNotFoundError(
            f"No encuentro {_RUTA_BASE}.\n"
            "play_r1_ia.py hereda de play_r1_camina_brazos.py y espera "
            "encontrarlo en la misma carpeta."
        )
    spec = importlib.util.spec_from_file_location("play_r1_camina_brazos", _RUTA_BASE)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


# =============================================================================
# INTENCIONES
# =============================================================================
# Lo que viaja del hilo del LLM al hilo del robot. Objetos inmutables y tontos:
# nada de logica, nada de referencias a MuJoCo.


@dataclass(frozen=True)
class Mover:
    """Comando de velocidad con fecha de caducidad.

    `hasta` es un instante de time.monotonic(). El bucle lo compara cada paso
    de control y pone el comando a cero cuando vence. Sin esto, un
    "camina adelante" haria caminar al robot para siempre.
    """

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    hasta: float = 0.0


@dataclass(frozen=True)
class Parar:
    pass


@dataclass(frozen=True)
class Gesto:
    nombre: str


@dataclass(frozen=True)
class Reiniciar:
    pass


# =============================================================================
# BUS DE INTENCIONES
# =============================================================================


class BusDeIntenciones:
    """Canal thread-safe entre el agente y el bucle de simulacion.

    Dos direcciones:
      - intenciones: agente -> robot (cola FIFO, no bloqueante)
      - snapshot:    robot  -> agente (ultimo estado conocido, con lock)
    """

    def __init__(self):
        self._cola: queue.Queue = queue.Queue(maxsize=64)
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] = {
            "listo": False,
            "posicion_m": [0.0, 0.0, 0.0],
            "orientacion_grados": 0.0,
            "velocidad_ms": 0.0,
            "comando_actual": [0.0, 0.0, 0.0],
            "gesto_activo": None,
            "altura_pelvis_m": 0.0,
            "caido": False,
            "tiempo_sim_s": 0.0,
        }

    # -- agente -> robot ----------------------------------------------------

    def enviar(self, intencion) -> bool:
        try:
            self._cola.put_nowait(intencion)
            return True
        except queue.Full:
            print("\n[BUS] Cola llena, intencion descartada.", flush=True)
            return False

    def drenar(self) -> list:
        pendientes = []
        while True:
            try:
                pendientes.append(self._cola.get_nowait())
            except queue.Empty:
                return pendientes

    # -- robot -> agente ----------------------------------------------------

    def publicar_snapshot(self, datos: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = datos

    def leer_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)


# =============================================================================
# HERRAMIENTAS
# =============================================================================
# Definidas en un formato neutro (nombre / descripcion / esquema JSON) para
# que cada backend las traduzca a su propio dialecto. Anthropic las quiere
# como `input_schema`, otros proveedores como `parameters`; el esquema JSON
# de dentro es identico en la practica.

GESTOS_VALIDOS = ["wave", "point", "carry", "cross", "guard", "neutral"]

DESCRIPCION_GESTOS = (
    "wave = saludar agitando la mano derecha; "
    "point = apuntar al frente con el brazo derecho; "
    "carry = extender ambos brazos como para cargar algo; "
    "cross = brazos en cruz horizontal; "
    "guard = pose de guardia tipo boxeo; "
    "neutral = volver a la pose de reposo"
)


@dataclass
class Herramienta:
    nombre: str
    descripcion: str
    esquema: dict[str, Any]
    ejecutar: Callable[..., str]


class CajaDeHerramientas:
    """Las capacidades reales del robot, expuestas al LLM.

    Regla de diseno: el catalogo es CERRADO. El LLM elige de una lista de
    acciones ya verificadas; nunca genera angulos de joints. Un LLM
    produciendo 10 floats de radianes da poses imposibles, autocolisiones y
    tirones que desestabilizan el balance.
    """

    # Limites reales de la politica: `command` se clipea a +-1.0 en
    # play_r1_camina_brazos.py y fue entrenada en ese rango.
    VEL_MAX = 1.0
    DURACION_MAX_S = 30.0

    def __init__(self, bus: BusDeIntenciones):
        self.bus = bus
        self._herramientas = self._construir()
        self.indice = {h.nombre: h for h in self._herramientas}

    def listar(self) -> list[Herramienta]:
        return list(self._herramientas)

    def despachar(self, nombre: str, argumentos: dict[str, Any]) -> str:
        herramienta = self.indice.get(nombre)
        if herramienta is None:
            return f"ERROR: no existe la herramienta '{nombre}'."
        try:
            return herramienta.ejecutar(**(argumentos or {}))
        except TypeError as e:
            return f"ERROR: argumentos invalidos para '{nombre}': {e}"
        except Exception as e:  # una herramienta rota no debe tumbar el agente
            return f"ERROR ejecutando '{nombre}': {e}"

    # -- implementaciones ---------------------------------------------------

    def _mover(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0,
               duracion_s: float = 2.0) -> str:
        vx = float(np.clip(vx, -self.VEL_MAX, self.VEL_MAX))
        vy = float(np.clip(vy, -self.VEL_MAX, self.VEL_MAX))
        wz = float(np.clip(wz, -self.VEL_MAX, self.VEL_MAX))
        duracion_s = float(np.clip(duracion_s, 0.1, self.DURACION_MAX_S))

        self.bus.enviar(Mover(vx=vx, vy=vy, wz=wz,
                              hasta=time.monotonic() + duracion_s))
        return (f"Moviendome durante {duracion_s:.1f}s con "
                f"vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}. "
                f"Me detendre solo al terminar.")

    def _parar(self) -> str:
        self.bus.enviar(Parar())
        return "Detenido. Comando de velocidad a cero."

    def _gesto(self, nombre: str) -> str:
        nombre = str(nombre).strip().lower()
        if nombre not in GESTOS_VALIDOS:
            return (f"ERROR: '{nombre}' no es un gesto valido. "
                    f"Opciones: {', '.join(GESTOS_VALIDOS)}.")
        self.bus.enviar(Gesto(nombre))
        return f"Ejecutando el gesto '{nombre}'. Tarda unos segundos."

    def _consultar_estado(self) -> str:
        s = self.bus.leer_snapshot()
        if not s.get("listo"):
            return "La simulacion todavia no esta lista."
        return json.dumps(s, ensure_ascii=False)

    def _reiniciar(self) -> str:
        self.bus.enviar(Reiniciar())
        return "Simulacion reiniciada: robot de vuelta en el origen, de pie."

    def _construir(self) -> list[Herramienta]:
        return [
            Herramienta(
                nombre="mover",
                descripcion=(
                    "Hace caminar al robot durante un tiempo determinado. El robot "
                    "se detiene solo cuando pasa la duracion, no hace falta llamar "
                    "a parar despues. Las velocidades son adimensionales en el "
                    "rango [-1, 1]: 0.3 es lento, 0.5 normal, 1.0 el maximo."
                ),
                esquema={
                    "type": "object",
                    "properties": {
                        "vx": {
                            "type": "number",
                            "description": "Velocidad adelante (positivo) o atras (negativo). Rango [-1, 1].",
                        },
                        "vy": {
                            "type": "number",
                            "description": "Velocidad lateral: izquierda (positivo) o derecha (negativo). Rango [-1, 1].",
                        },
                        "wz": {
                            "type": "number",
                            "description": "Velocidad de giro: izquierda/antihorario (positivo) o derecha/horario (negativo). Rango [-1, 1].",
                        },
                        "duracion_s": {
                            "type": "number",
                            "description": "Segundos que mantiene el movimiento antes de detenerse solo. Maximo 30.",
                        },
                    },
                    "required": ["duracion_s"],
                },
                ejecutar=self._mover,
            ),
            Herramienta(
                nombre="parar",
                descripcion="Detiene inmediatamente cualquier movimiento. El robot se queda de pie, quieto.",
                esquema={"type": "object", "properties": {}},
                ejecutar=self._parar,
            ),
            Herramienta(
                nombre="gesto",
                descripcion=f"Ejecuta un gesto con los brazos. {DESCRIPCION_GESTOS}.",
                esquema={
                    "type": "object",
                    "properties": {
                        "nombre": {
                            "type": "string",
                            "enum": GESTOS_VALIDOS,
                            "description": "Nombre del gesto a ejecutar.",
                        }
                    },
                    "required": ["nombre"],
                },
                ejecutar=self._gesto,
            ),
            Herramienta(
                nombre="consultar_estado",
                descripcion=(
                    "Devuelve el estado actual del robot en JSON: posicion en metros, "
                    "orientacion en grados, velocidad, comando activo, gesto en curso, "
                    "altura de la pelvis y si se ha caido. Usalo antes de contestar "
                    "cualquier pregunta sobre donde esta o que esta haciendo."
                ),
                esquema={"type": "object", "properties": {}},
                ejecutar=self._consultar_estado,
            ),
            Herramienta(
                nombre="reiniciar",
                descripcion="Reinicia la simulacion: devuelve el robot al origen, de pie. Usalo si se ha caido.",
                esquema={"type": "object", "properties": {}},
                ejecutar=self._reiniciar,
            ),
        ]


# =============================================================================
# BACKENDS DE LLM
# =============================================================================

INSTRUCCIONES_SISTEMA = """\
Eres el cerebro de un robot humanoide Unitree R1 que corre en una simulacion \
MuJoCo. Hablas espanol.

Como te comportas:
- Eres el robot. Habla en primera persona: "voy a caminar", no "el robot caminara".
- Sé breve. Una o dos frases. Estas hablando en voz alta, no escribiendo un informe.
- Para actuar, usa las herramientas. No describas movimientos que no has ejecutado.
- Antes de responder cualquier pregunta sobre tu posicion, velocidad o lo que \
estas haciendo, llama a `consultar_estado`. No inventes ni supongas.
- Puedes encadenar varias herramientas en un turno si el usuario pide varias cosas.

Tus limites fisicos, respetalos:
- Solo caminas y haces los gestos del catalogo. No tienes manos que agarren, \
no puedes subir escaleras, no ves con camara.
- Las velocidades van en [-1, 1]. Por encima de 0.6 el robot se vuelve inestable; \
usa 0.3-0.5 salvo que te pidan ir rapido.
- Si te piden algo que no puedes hacer, dilo claramente en una frase y ofrece \
lo mas parecido que si puedas hacer.
"""


@dataclass
class RespuestaLLM:
    """Lo que devuelve un backend tras procesar un turno completo."""

    texto: str
    herramientas_usadas: list[str] = field(default_factory=list)


class BackendLLM(ABC):
    """Interfaz que debe cumplir cualquier proveedor de LLM.

    Para enchufar una API nueva solo hay que implementar `responder`. Ver
    `BackendPlantilla` mas abajo, que es el esqueleto con los TODO.
    """

    nombre = "base"

    def __init__(self, caja: CajaDeHerramientas, modelo: str | None = None):
        self.caja = caja
        self.modelo = modelo

    @abstractmethod
    def responder(self, texto_usuario: str) -> RespuestaLLM:
        """Procesa un turno: llama al modelo, ejecuta herramientas, devuelve texto."""

    def reiniciar_conversacion(self) -> None:
        """Olvida el historial. Por defecto no hace nada."""


# -----------------------------------------------------------------------------
# Backend 1: SIMULADO (sin API key, sin red)
# -----------------------------------------------------------------------------


class BackendSimulado(BackendLLM):
    """Interprete por palabras clave. No es un LLM y no lo pretende.

    Existe por dos razones:
      1. Puedes demostrar el sistema completo funcionando sin gastar un token
         ni tener la API key todavia.
      2. Ejercita exactamente el mismo camino de codigo que un backend real
         (misma caja de herramientas, mismo bus, mismo hilo), asi que cuando
         enchufes la API de verdad ya sabes que la mitad del robot funciona.
    """

    nombre = "simulado"

    PATRONES_GESTO = {
        "wave": r"salud|hola|saluda|adios|despid",
        "point": r"apunt|senal|señal|indica",
        "carry": r"carg|sostien|lleva|agarr",
        "cross": r"cruz|brazos en cruz|abre los brazos",
        "guard": r"guardia|defens|box|pelea",
        "neutral": r"neutral|descans|relaj|baja los brazos|firme",
    }

    def responder(self, texto_usuario: str) -> RespuestaLLM:
        t = texto_usuario.lower().strip()
        usadas: list[str] = []
        partes: list[str] = []

        # --- parar tiene prioridad absoluta ---
        if re.search(r"\bpara\b|\bparate\b|detente|deten|alto|quieto|stop", t):
            partes.append(self.caja.despachar("parar", {}))
            usadas.append("parar")
            return RespuestaLLM(" ".join(partes), usadas)

        if re.search(r"reinicia|resetea|levantate|de pie otra vez", t):
            partes.append(self.caja.despachar("reiniciar", {}))
            usadas.append("reiniciar")
            return RespuestaLLM(" ".join(partes), usadas)

        # --- preguntas de estado ---
        if re.search(r"donde estas|que haces|estado|como estas|posicion|"
                     r"que tan rapido|velocidad|te caiste|estas bien", t):
            crudo = self.caja.despachar("consultar_estado", {})
            usadas.append("consultar_estado")
            return RespuestaLLM(self._narrar_estado(crudo), usadas)

        # --- movimiento ---
        vx = vy = wz = 0.0
        if re.search(r"adelante|avanza|camina|sigue|ve al frente|recto", t):
            vx = 0.4
        if re.search(r"atras|retrocede|para atras|reversa", t):
            vx = -0.35
        if re.search(r"izquierda", t):
            wz = 0.4
        if re.search(r"derecha", t):
            wz = -0.4
        if re.search(r"lateral|de lado|costado", t):
            vy, wz = (0.35, 0.0) if "izquierda" in t else (-0.35, 0.0)
        if re.search(r"gira|voltea|date la vuelta|rota", t) and wz == 0.0:
            wz = 0.5

        # modificadores de velocidad
        if re.search(r"despacio|lento|suave|con calma", t):
            vx, vy, wz = vx * 0.55, vy * 0.55, wz * 0.7
        if re.search(r"rapido|deprisa|corre|veloz", t):
            vx, vy, wz = vx * 1.5, vy * 1.5, wz * 1.3

        duracion = self._extraer_duracion(t)

        if vx or vy or wz:
            partes.append(self.caja.despachar(
                "mover", {"vx": vx, "vy": vy, "wz": wz, "duracion_s": duracion}
            ))
            usadas.append("mover")

        # --- gestos ---
        for gesto, patron in self.PATRONES_GESTO.items():
            if re.search(patron, t):
                partes.append(self.caja.despachar("gesto", {"nombre": gesto}))
                usadas.append("gesto")
                break

        if not partes:
            return RespuestaLLM(
                "No entendi. Prueba con: 'camina adelante 3 segundos', "
                "'gira a la izquierda', 'saludame', 'donde estas?', 'para'. "
                "(Backend simulado: solo palabras clave. Con una API real "
                "entenderia lenguaje libre.)",
                usadas,
            )

        return RespuestaLLM(" ".join(partes), usadas)

    @staticmethod
    def _extraer_duracion(t: str) -> float:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:s\b|seg|segundo)", t)
        if m:
            return float(m.group(1).replace(",", "."))
        palabras = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
                    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
                    "nueve": 9, "diez": 10}
        m = re.search(r"\b(" + "|".join(palabras) + r")\s+segundo", t)
        if m:
            return float(palabras[m.group(1)])
        return 2.0

    @staticmethod
    def _narrar_estado(crudo: str) -> str:
        try:
            s = json.loads(crudo)
        except (json.JSONDecodeError, TypeError):
            return crudo
        x, y, _ = s["posicion_m"]
        txt = (f"Estoy en x={x:.2f} m, y={y:.2f} m, mirando hacia "
               f"{s['orientacion_grados']:.0f} grados, a {s['velocidad_ms']:.2f} m/s.")
        if s.get("gesto_activo"):
            txt += f" Ahora mismo estoy haciendo el gesto '{s['gesto_activo']}'."
        if s.get("caido"):
            txt += " Me he caido, pideme que reinicie."
        return txt


# -----------------------------------------------------------------------------
# Backend 2: ANTHROPIC (Claude)
# -----------------------------------------------------------------------------


class BackendAnthropic(BackendLLM):
    """Claude via el SDK oficial, con tool use y memoria de conversacion.

    Bucle manual en vez del tool_runner del SDK a proposito: las herramientas
    estan definidas en un formato neutro compartido con los demas backends,
    para que cambiar de proveedor no obligue a reescribirlas.
    """

    nombre = "anthropic"
    MODELO_POR_DEFECTO = "claude-opus-5"
    MAX_VUELTAS = 6  # tope de ciclos de herramientas por turno

    def __init__(self, caja: CajaDeHerramientas, modelo: str | None = None):
        super().__init__(caja, modelo or self.MODELO_POR_DEFECTO)
        try:
            import anthropic
        except ImportError as e:
            raise SystemExit(
                "Falta el SDK de Anthropic. Instalalo con:\n"
                "    pip install anthropic\n"
                "O usa el backend sin dependencias:\n"
                "    python scripts/r1/play_r1_ia.py --backend simulado"
            ) from e

        self._anthropic = anthropic
        # El cliente resuelve las credenciales del entorno (ANTHROPIC_API_KEY).
        # Nunca escribas la key en el codigo.
        self.cliente = anthropic.Anthropic()
        self.historial: list[dict[str, Any]] = []
        self.herramientas_api = [
            {
                "name": h.nombre,
                "description": h.descripcion,
                "input_schema": h.esquema,
            }
            for h in self.caja.listar()
        ]

    def reiniciar_conversacion(self) -> None:
        self.historial = []

    def responder(self, texto_usuario: str) -> RespuestaLLM:
        self.historial.append({"role": "user", "content": texto_usuario})
        usadas: list[str] = []
        textos: list[str] = []

        for _ in range(self.MAX_VUELTAS):
            try:
                respuesta = self.cliente.messages.create(
                    model=self.modelo,
                    max_tokens=2048,
                    system=INSTRUCCIONES_SISTEMA,
                    tools=self.herramientas_api,
                    messages=self.historial,
                    # Efecto bajo: un robot que responde a comandos necesita
                    # latencia corta, no razonamiento profundo. El pensamiento
                    # se queda activo (desactivarlo del todo tiene efectos raros
                    # con tool use).
                    output_config={"effort": "low"},
                )
            except self._anthropic.APIStatusError as e:
                return RespuestaLLM(f"[API {e.status_code}] {e.message}", usadas)
            except self._anthropic.APIConnectionError:
                return RespuestaLLM("[Sin conexion con la API. Revisa la red.]", usadas)

            if respuesta.stop_reason == "refusal":
                return RespuestaLLM("[El modelo declino responder a eso.]", usadas)

            textos += [b.text for b in respuesta.content if b.type == "text"]
            self.historial.append({"role": "assistant", "content": respuesta.content})

            bloques_tool = [b for b in respuesta.content if b.type == "tool_use"]
            if not bloques_tool:
                break

            resultados = []
            for bloque in bloques_tool:
                print(f"  [tool] {bloque.name}({json.dumps(bloque.input, ensure_ascii=False)})",
                      flush=True)
                salida = self.caja.despachar(bloque.name, bloque.input)
                usadas.append(bloque.name)
                resultados.append({
                    "type": "tool_result",
                    "tool_use_id": bloque.id,
                    "content": salida,
                    "is_error": salida.startswith("ERROR"),
                })
            self.historial.append({"role": "user", "content": resultados})

        self._recortar_historial()
        return RespuestaLLM(" ".join(t.strip() for t in textos if t.strip())
                            or "(sin respuesta de texto)", usadas)

    def _recortar_historial(self, maximo: int = 40) -> None:
        """Evita que la conversacion crezca sin limite en sesiones largas."""
        if len(self.historial) > maximo:
            # No cortamos a mitad de un par tool_use/tool_result.
            corte = len(self.historial) - maximo
            while corte < len(self.historial):
                msg = self.historial[corte]
                contenido = msg.get("content")
                es_tool_result = (
                    isinstance(contenido, list)
                    and contenido
                    and isinstance(contenido[0], dict)
                    and contenido[0].get("type") == "tool_result"
                )
                if msg["role"] == "user" and not es_tool_result:
                    break
                corte += 1
            self.historial = self.historial[corte:]


# -----------------------------------------------------------------------------
# Backend 3: OPENAI (ChatGPT)
# -----------------------------------------------------------------------------


class BackendOpenAI(BackendLLM):
    """GPT via el SDK oficial de OpenAI, con tool calling y memoria.

    Mismo bucle manual que `BackendAnthropic` y misma caja de herramientas
    neutra; lo unico que cambia es el dialecto:

        Anthropic : herramienta = {"name", "description", "input_schema"}
                    resultado   = bloque tool_result dentro de un mensaje user
        OpenAI    : herramienta = {"type": "function", "function": {...}}
                    resultado   = mensaje aparte con role "tool"

    Y que aqui los argumentos llegan como STRING JSON, no como dict: hay que
    parsearlos, y hacerlo con red debajo porque el modelo puede mandar JSON
    invalido y eso no debe tumbar al robot.
    """

    nombre = "openai"
    MODELO_POR_DEFECTO = "gpt-5-mini"
    MAX_VUELTAS = 6  # tope de ciclos de herramientas por turno

    def __init__(self, caja: CajaDeHerramientas, modelo: str | None = None):
        super().__init__(caja, modelo or self.MODELO_POR_DEFECTO)
        try:
            import openai
        except ImportError as e:
            raise SystemExit(
                "Falta el SDK de OpenAI. Instalalo con:\n"
                "    pip install openai\n"
                "O usa el backend sin dependencias:\n"
                "    python scripts/r1/play_r1_ia.py --backend simulado"
            ) from e

        self._openai = openai
        # El cliente resuelve las credenciales del entorno (OPENAI_API_KEY).
        # Nunca escribas la key en el codigo.
        self.cliente = openai.OpenAI()
        self.historial: list[dict[str, Any]] = []
        self.herramientas_api = [
            {
                "type": "function",
                "function": {
                    "name": h.nombre,
                    "description": h.descripcion,
                    "parameters": h.esquema,
                },
            }
            for h in self.caja.listar()
        ]
        # Los gpt-5 aceptan reasoning_effort; los anteriores lo rechazan. Un
        # robot que despacha herramientas quiere latencia, no reflexion:
        # medido con este catalogo, "minimal" tarda la mitad que "low" y elige
        # las mismas herramientas. Si alguna vez se equivoca al encadenar
        # varias, subelo a "low".
        self._esfuerzo = "minimal" if str(self.modelo).startswith("gpt-5") else None

    def reiniciar_conversacion(self) -> None:
        self.historial = []

    def _llamar(self):
        extra = {"reasoning_effort": self._esfuerzo} if self._esfuerzo else {}
        mensajes = [{"role": "system", "content": INSTRUCCIONES_SISTEMA}] + self.historial
        try:
            return self.cliente.chat.completions.create(
                model=self.modelo,
                messages=mensajes,
                tools=self.herramientas_api,
                tool_choice="auto",
                **extra,
            )
        except self._openai.BadRequestError:
            if not extra:
                raise
            # El modelo no conoce reasoning_effort: lo dejamos de mandar.
            self._esfuerzo = None
            return self.cliente.chat.completions.create(
                model=self.modelo,
                messages=mensajes,
                tools=self.herramientas_api,
                tool_choice="auto",
            )

    def responder(self, texto_usuario: str) -> RespuestaLLM:
        self.historial.append({"role": "user", "content": texto_usuario})
        usadas: list[str] = []
        textos: list[str] = []

        for _ in range(self.MAX_VUELTAS):
            try:
                respuesta = self._llamar()
            except self._openai.APIStatusError as e:
                return RespuestaLLM(f"[API {e.status_code}] {e.message}", usadas)
            except self._openai.APIConnectionError:
                return RespuestaLLM("[Sin conexion con la API. Revisa la red.]", usadas)

            mensaje = respuesta.choices[0].message
            if mensaje.content:
                textos.append(mensaje.content)

            llamadas = mensaje.tool_calls or []
            entrada: dict[str, Any] = {"role": "assistant", "content": mensaje.content}
            if llamadas:
                entrada["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.function.name,
                                     "arguments": c.function.arguments},
                    }
                    for c in llamadas
                ]
            self.historial.append(entrada)

            if not llamadas:
                break

            for llamada in llamadas:
                nombre = llamada.function.name
                try:
                    argumentos = json.loads(llamada.function.arguments or "{}")
                except json.JSONDecodeError:
                    salida = (f"ERROR: los argumentos de '{nombre}' no son JSON "
                              "valido. Vuelve a llamarla bien formada.")
                else:
                    print(f"  [tool] {nombre}({json.dumps(argumentos, ensure_ascii=False)})",
                          flush=True)
                    salida = self.caja.despachar(nombre, argumentos)
                    usadas.append(nombre)
                self.historial.append({"role": "tool",
                                       "tool_call_id": llamada.id,
                                       "content": salida})

        self._recortar_historial()
        return RespuestaLLM(" ".join(t.strip() for t in textos if t.strip())
                            or "(sin respuesta de texto)", usadas)

    def _recortar_historial(self, maximo: int = 40) -> None:
        """Evita que la conversacion crezca sin limite en sesiones largas.

        El corte tiene que caer en un mensaje 'user': si empezara en un 'tool'
        (o en el 'assistant' que lo pidio), la API rechaza el historial por
        tener resultados de herramientas huerfanos.
        """
        if len(self.historial) <= maximo:
            return
        corte = len(self.historial) - maximo
        while corte < len(self.historial):
            if self.historial[corte].get("role") == "user":
                break
            corte += 1
        self.historial = self.historial[corte:]


# -----------------------------------------------------------------------------
# Backend 4: PLANTILLA — enchufa aqui la API que elijas
# -----------------------------------------------------------------------------


class BackendPlantilla(BackendLLM):
    """Esqueleto para CUALQUIER otra API de LLM.

    Cuando me digas que proveedor usamos, esto es lo unico que hay que
    rellenar. El resto del archivo (robot, hilos, herramientas, cola,
    seguridad) no se toca.

    Son tres cosas:

      1. TRADUCIR las herramientas al formato del proveedor. Casi todos usan
         el mismo JSON Schema; solo cambia el nombre del campo que lo envuelve:
             Anthropic : {"name", "description", "input_schema"}
             OpenAI    : {"type": "function",
                          "function": {"name", "description", "parameters"}}
             Gemini    : {"name", "description", "parameters"}
         El contenido de `h.esquema` sirve tal cual en los tres casos.

      2. LLAMAR al modelo con el historial + las herramientas.

      3. BUCLE de herramientas: mientras el modelo pida llamadas, ejecutarlas
         con `self.caja.despachar(nombre, argumentos)` y devolver el resultado
         al modelo. Terminar cuando conteste solo con texto.

    Mira `BackendAnthropic` arriba como referencia completa de los tres pasos.
    """

    nombre = "plantilla"

    def __init__(self, caja: CajaDeHerramientas, modelo: str | None = None):
        super().__init__(caja, modelo)
        self.historial: list[dict[str, Any]] = []

        # TODO(1): crea aqui el cliente del proveedor.
        #   from proveedor import Cliente
        #   self.cliente = Cliente(api_key=os.environ["MI_API_KEY"])
        #
        # NUNCA escribas la API key en el codigo. Leela del entorno.
        self.cliente = None

        # TODO(2): traduce el catalogo neutro al formato del proveedor.
        self.herramientas_api = [
            {
                "name": h.nombre,
                "description": h.descripcion,
                "parameters": h.esquema,   # <- renombra este campo si hace falta
            }
            for h in self.caja.listar()
        ]

    def reiniciar_conversacion(self) -> None:
        self.historial = []

    def responder(self, texto_usuario: str) -> RespuestaLLM:
        raise NotImplementedError(
            "BackendPlantilla todavia no esta conectado a ninguna API.\n"
            "Rellena los TODO de esta clase, o usa:\n"
            "    --backend simulado    (sin API key)\n"
            "    --backend anthropic   (Claude)"
        )

        # --- Esqueleto del bucle, para cuando lo implementes -----------------
        # self.historial.append({"role": "user", "content": texto_usuario})
        # usadas = []
        # for _ in range(6):
        #     r = self.cliente.chat(model=self.modelo,
        #                           system=INSTRUCCIONES_SISTEMA,
        #                           messages=self.historial,
        #                           tools=self.herramientas_api)
        #     self.historial.append(r.mensaje)
        #     llamadas = r.llamadas_a_herramientas
        #     if not llamadas:
        #         return RespuestaLLM(r.texto, usadas)
        #     for c in llamadas:
        #         salida = self.caja.despachar(c.nombre, c.argumentos)
        #         usadas.append(c.nombre)
        #         self.historial.append({"role": "tool",
        #                                "tool_call_id": c.id,
        #                                "content": salida})
        # return RespuestaLLM("Demasiadas vueltas de herramientas.", usadas)


BACKENDS: dict[str, type[BackendLLM]] = {
    "simulado": BackendSimulado,
    "anthropic": BackendAnthropic,
    "openai": BackendOpenAI,
    "plantilla": BackendPlantilla,
}


# =============================================================================
# ENTORNO: R1 + cola de intenciones
# =============================================================================


def construir_clase_entorno(base):
    """Crea la subclase del entorno. Se hace en funcion porque la clase base
    se carga en tiempo de ejecucion desde play_r1_camina_brazos.py."""

    class R1ConIA(base):
        """El R1 de siempre, mas un canal de intenciones desde el agente.

        Solo se sobrescribe `_policy_step`, que corre a 50 Hz. Toda la
        simulacion, la politica y los gestos son exactamente los de la clase
        padre, sin cambios.
        """

        PASOS_ENTRE_SNAPSHOTS = 5   # 50 Hz / 5 = 10 Hz de telemetria
        ALTURA_CAIDO_M = 0.45

        def __init__(self, bus: BusDeIntenciones, **kwargs):
            self.bus = bus                    # antes de super(): __init__ llama a reset()
            self._vence_comando = 0.0
            self._contador_snapshot = 0
            super().__init__(**kwargs)

        # -- lo unico que cambia del bucle -----------------------------------

        def _policy_step(self):
            self._procesar_intenciones()
            self._publicar_telemetria()
            return super()._policy_step()

        def _procesar_intenciones(self):
            for intencion in self.bus.drenar():
                if isinstance(intencion, Mover):
                    self.command[0] = intencion.vx
                    self.command[1] = intencion.vy
                    self.command[2] = intencion.wz
                    np.clip(self.command, -1.0, 1.0, out=self.command)
                    self._vence_comando = intencion.hasta

                elif isinstance(intencion, Parar):
                    self.command[:] = 0.0
                    self._vence_comando = 0.0

                elif isinstance(intencion, Gesto):
                    # play() se llama desde ESTE hilo, no desde el del agente:
                    # lee self.current_pose y arranca un time.time(), asi que
                    # hacerlo desde fuera del bucle seria una carrera.
                    self.arm_ctrl.play(intencion.nombre)

                elif isinstance(intencion, Reiniciar):
                    self.command[:] = 0.0
                    self._vence_comando = 0.0
                    self.reset()

            # Caducidad: sin esto un "camina adelante" no termina nunca.
            if self._vence_comando and time.monotonic() >= self._vence_comando:
                self.command[:] = 0.0
                self._vence_comando = 0.0

        def _publicar_telemetria(self):
            self._contador_snapshot += 1
            if self._contador_snapshot < self.PASOS_ENTRE_SNAPSHOTS:
                return
            self._contador_snapshot = 0

            pos = self.data.qpos[0:3]
            quat = self.data.qpos[3:7]
            vel = self.data.qvel[0:3]
            _, _, yaw = quat_to_euler(quat)

            seq = self.arm_ctrl._active_seq
            gesto_activo = seq.name if (seq is not None and seq.active) else None

            self.bus.publicar_snapshot({
                "listo": True,
                "posicion_m": [round(float(v), 3) for v in pos],
                "orientacion_grados": round(float(np.degrees(yaw)), 1),
                "velocidad_ms": round(float(np.linalg.norm(vel[:2])), 3),
                "comando_actual": [round(float(c), 2) for c in self.command],
                "gesto_activo": gesto_activo,
                "altura_pelvis_m": round(float(pos[2]), 3),
                "caido": bool(pos[2] < self.ALTURA_CAIDO_M),
                "tiempo_sim_s": round(float(self.data.time), 1),
            })

    return R1ConIA


# =============================================================================
# CONSOLA
# =============================================================================


BANNER = """
================================================================
  R1 + IA  —  control por lenguaje natural
================================================================
  Backend LLM : {backend}
  Modelo      : {modelo}

  Escribe una orden y pulsa Enter. Ejemplos:
     camina hacia adelante 3 segundos
     gira a la izquierda despacio
     saludame
     donde estas?
     para

  Comandos locales:  /estado   /limpiar   /salir
  El teclado del visor sigue funcionando (ESPACIO = freno).
================================================================
"""


class Consola(threading.Thread):
    """Hilo que lee stdin, se lo pasa al LLM e imprime la respuesta.

    Va en su propio hilo justamente para poder bloquear tranquilamente:
    en la red, en input(), en lo que sea. El bucle de MuJoCo ni se entera.
    """

    def __init__(self, backend: BackendLLM, bus: BusDeIntenciones, parar_evt: threading.Event):
        super().__init__(name="consola-ia", daemon=True)
        self.backend = backend
        self.bus = bus
        self.parar_evt = parar_evt

    def run(self):
        print(BANNER.format(backend=self.backend.nombre,
                            modelo=self.backend.modelo or "-"), flush=True)
        while not self.parar_evt.is_set():
            try:
                texto = input("tu > ").strip()
            except (EOFError, KeyboardInterrupt):
                self.parar_evt.set()
                return

            if not texto:
                continue

            if texto in ("/salir", "/exit", "/quit"):
                self.parar_evt.set()
                return
            if texto == "/limpiar":
                self.backend.reiniciar_conversacion()
                print("robot > Conversacion olvidada.\n", flush=True)
                continue
            if texto == "/estado":
                print(json.dumps(self.bus.leer_snapshot(), indent=2, ensure_ascii=False),
                      flush=True)
                continue

            t0 = time.monotonic()
            try:
                respuesta = self.backend.responder(texto)
            except Exception as e:
                print(f"robot > [fallo del backend: {type(e).__name__}: {e}]\n", flush=True)
                continue
            dt = time.monotonic() - t0

            print(f"robot > {respuesta.texto}")
            print(f"         ({dt:.2f}s"
                  + (f", herramientas: {', '.join(respuesta.herramientas_usadas)}"
                     if respuesta.herramientas_usadas else "")
                  + ")\n", flush=True)


# =============================================================================
# MAIN
# =============================================================================


def parsear_argumentos():
    p = argparse.ArgumentParser(
        description="R1 en MuJoCo controlado por lenguaje natural.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--backend", default=os.environ.get("AMO_LLM_BACKEND", "simulado"),
        choices=sorted(BACKENDS),
        help="Proveedor de LLM. 'simulado' no necesita API key (por defecto).",
    )
    p.add_argument(
        "--modelo", default=os.environ.get("AMO_LLM_MODELO"),
        help="Id del modelo. Si se omite, cada backend usa el suyo por defecto.",
    )
    p.add_argument(
        "--sin-sim", action="store_true",
        help="No levanta MuJoCo. Solo el agente, para probar el cableado de la API.",
    )
    p.add_argument("--device", default="cpu", help="cpu o cuda (la red es minuscula, cpu sobra).")
    return p.parse_args()


def main():
    args = parsear_argumentos()

    bus = BusDeIntenciones()
    caja = CajaDeHerramientas(bus)

    print(f"Inicializando backend '{args.backend}'...", flush=True)
    backend = BACKENDS[args.backend](caja, modelo=args.modelo)

    parar_evt = threading.Event()

    # ---- modo solo-chat: probar la API sin arrancar la simulacion ----------
    if args.sin_sim:
        print("\n[MODO SIN SIMULACION] Las herramientas se ejecutan y encolan de\n"
              "verdad, pero no hay robot que las consuma. Sirve para verificar\n"
              "que la API responde y que el LLM elige bien las herramientas.\n")
        bus.publicar_snapshot({
            "listo": True, "posicion_m": [0.0, 0.0, 0.74],
            "orientacion_grados": 0.0, "velocidad_ms": 0.0,
            "comando_actual": [0.0, 0.0, 0.0], "gesto_activo": None,
            "altura_pelvis_m": 0.74, "caido": False, "tiempo_sim_s": 0.0,
            "nota": "simulacion no activa (--sin-sim)",
        })
        Consola(backend, bus, parar_evt).run()   # en el hilo principal
        return

    # ---- modo normal: simulacion + agente ---------------------------------
    print("Cargando entorno base (play_r1_camina_brazos.py)...", flush=True)
    base = _cargar_entorno_base()
    R1ConIA = construir_clase_entorno(base.R1CaminaBrazos)

    entorno = R1ConIA(bus, device=args.device)

    consola = Consola(backend, bus, parar_evt)
    consola.start()

    try:
        entorno.run()          # bucle de MuJoCo, bloquea hasta que cierras el visor
    finally:
        parar_evt.set()
        print("\nAgente detenido. Pulsa Enter si la consola sigue abierta.", flush=True)


if __name__ == "__main__":
    main()
