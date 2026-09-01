#!/usr/bin/env python3
"""
play_r1_voz.py — R1 con MANOS, controlado por VOZ y por consola, con
respuestas habladas.

QUE HACE
========
Toma TODO lo de `play_r1_ia.py` (robot + Claude + herramientas + consola), le
anade una capa de voz completamente offline y le pone las MANOS Revo2 del
modelo `r1_manos.xml`, las mismas de `scripts/teleop/teleop_dedos.py`:

    tu hablas   ->  microfono -> Vosk (reconocimiento en espanol)
    el LLM      ->  decide que herramientas llamar (igual que por texto)
    el robot    ->  ejecuta la accion en MuJoCo (caminar, gesto, dedos)
    la voz      ->  Piper sintetiza la respuesta y suena por los altavoces

    tu mano     ->  camara -> MediaPipe -> curl por dedo -> dedos del robot
                    (opcional, con --dedos-camara)
    tu brazo    ->  camara -> MediaPipe -> IK -> hombros, codos y cintura
                    (opcional, con --brazos-camara)

La consola sigue funcionando en paralelo: puedes hablarle O escribirle, y
siempre contesta por texto Y por voz. Un comando hablado ("cierra la mano
derecha") ejecuta la accion exactamente igual que uno escrito.

MANOS
=====
Tres formas de mover los 22 joints de dedos, en este orden de prioridad:

  1. CAMARA  (--dedos-camara): mientras se te vea una mano, los dedos del
     robot copian tu cierre dedo a dedo. Se enciende y se apaga por voz
     ("copia mis manos" / "deja de copiar").
  2. GESTOS  de brazo: cada gesto lleva su pose de dedos coordinada
     (saludar abre la mano, apuntar extiende el indice, guardia cierra los
     punos...). Al terminar el gesto los dedos vuelven a su pose anterior.
  3. VOZ     directa: "cierra la mano derecha", "haz una pinza", "pulgar
     arriba". Catalogo cerrado de poses, igual que los gestos de brazo.

La politica de caminar NO controla los dedos: son DOFs aparte, con su propio
PD. Por eso las manos no interfieren con el equilibrio.

BRAZOS Y CINTURA
================
Con `--brazos-camara` el robot copia ademas tus brazos: MediaPipe Holistic
mide a donde apuntan tu brazo y tu antebrazo, una IK amortiguada saca los 4
joints de cada hombro/codo, y el resultado se MEZCLA con lo que pide la
politica de caminar (entra y sale en ~0.5 s, nunca de golpe). El giro de tu
torso va al waist_yaw, recortado a +-0.35 rad para no tumbar al robot; se
puede quitar con `--sin-cintura`.

Estos si son DOFs de la politica, pero la observacion ya congela los brazos
(el padre pone DEFAULT_MJ en la obs), asi que la politica ni se entera, igual
que con los gestos.

ARQUITECTURA
============
    [hilo oido]     microfono 16 kHz -> Vosk -> texto final
                          |                          (se silencia mientras
                          v                           el robot habla, para
    [BackendCompartido]  un lock: consola y voz       no escucharse a si
                          comparten el mismo LLM       mismo)
                          |
                          v
    [hilo consola]  input() de siempre  ->  mismo backend
                          |
                          v  BusDeIntenciones (thread-safe)
    [hilo robot]    MuJoCo 500 Hz: politica + gestos + PD de dedos
                          ^          ^
                          |          |
    [hilo camara]   MediaPipe   [hilo IK] 30 Hz: direcciones -> joints
                    (Hands o Holistic)   (copia propia del modelo, nunca
                                          toca la data de la simulacion)

REQUISITOS (ya instalados si seguiste el setup)
===============================================
    pip install vosk sounddevice piper-tts
    conda install -c conda-forge portaudio
    modelos en <raiz>/modelos_voz/:
        vosk-model-small-es-0.42/          (reconocimiento, ~40 MB)
        es_ES-davefx-medium.onnx (+.json)  (sintesis, ~60 MB)
    para --dedos-camara: mediapipe + opencv (ya en requirements-r1.txt)

USO
===
    conda activate r1mujoco

    # robot con manos + voz + consola:
    python scripts/r1/play_r1_voz.py

    # ademas, que los dedos copien tu mano por la camara:
    python scripts/r1/play_r1_voz.py --dedos-camara --camara 0 --ver-camara

    # con DroidCam por USB (adb forward tcp:4747 tcp:4747):
    python scripts/r1/play_r1_voz.py --dedos-camara

    # tren superior completo: brazos + cintura + dedos copiando los tuyos:
    python scripts/r1/play_r1_voz.py --brazos-camara --dedos-camara --ver-camara

    # brazos si, cintura no (el robot no gira el torso):
    python scripts/r1/play_r1_voz.py --brazos-camara --sin-cintura

    # el robot de antes, sin manos (r1.xml):
    python scripts/r1/play_r1_voz.py --sin-manos

    # probar solo la voz, sin levantar MuJoCo:
    python scripts/r1/play_r1_voz.py --sin-sim

    # solo escuchar cuando la frase empiece por una palabra clave:
    python scripts/r1/play_r1_voz.py --palabra-clave robot

    # sin sintesis de voz (solo escucha) / sin microfono (solo habla):
    python scripts/r1/play_r1_voz.py --sin-tts
    python scripts/r1/play_r1_voz.py --sin-mic

NO MODIFICA NINGUN ARCHIVO EXISTENTE: play_r1_ia.py y
play_r1_camina_brazos.py se cargan tal cual y se extienden por herencia.
"""

import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RAIZ))

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "glfw")   # igual que los demas scripts del R1

import numpy as np
import mujoco

from amo.control.arm_ik import ControladorBrazosCamara
from amo.control.hand_controller import (
    DESCRIPCION_POSES,
    POSES,
    POSE_REPOSO,
    normalizar_lado,
    ControladorManos,
)
from amo.paths import scene
from amo.vision.cuerpo_camara import SeguidorTrenSuperior
from amo.vision.manos_camara import FUENTE_POR_DEFECTO, SeguidorDeManos

# =============================================================================
# CARGA DE play_r1_ia.py (reutilizamos backends, bus, herramientas y consola)
# =============================================================================

_RUTA_IA = Path(__file__).resolve().parent / "play_r1_ia.py"


def _cargar_modulo_ia():
    if not _RUTA_IA.exists():
        raise FileNotFoundError(
            f"No encuentro {_RUTA_IA}. play_r1_voz.py hereda de play_r1_ia.py."
        )
    spec = importlib.util.spec_from_file_location("play_r1_ia", _RUTA_IA)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = modulo
    spec.loader.exec_module(modulo)
    return modulo


ia = _cargar_modulo_ia()

# Rutas por defecto de los modelos de voz
_DIR_MODELOS = _RAIZ / "modelos_voz"
_MODELO_VOSK = _DIR_MODELOS / "vosk-model-small-es-0.42"
_MODELO_PIPER = _DIR_MODELOS / "es_ES-davefx-medium.onnx"


def _detectar_headset_usb():
    """Busca un headset USB conectado.

    Devuelve (indice_microfono, dispositivo_alsa_salida). Si hay un headset
    USB (microfono + audifonos), lo preferimos sobre los jacks de la placa:
    si el usuario lo conecto, es porque quiere usarlo.
    """
    try:
        import sounddevice as sd
        mic, salida = None, None
        for i, d in enumerate(sd.query_devices()):
            if "usb" not in d["name"].lower():
                continue
            m = re.search(r"\(hw:(\d+),(\d+)\)", d["name"])
            alsa = f"plughw:{m.group(1)},{m.group(2)}" if m else None
            if mic is None and d["max_input_channels"] > 0:
                mic = i
            if salida is None and alsa and d["max_output_channels"] > 0:
                salida = alsa
        return mic, salida
    except Exception:
        return None, None


# =============================================================================
# VOZ (texto -> audio)
# =============================================================================


class Voz:
    """Sintetiza y reproduce las respuestas del robot.

    Motor primario: Piper (voz neuronal espanola, offline).
    Respaldo: espeak-ng si Piper o su modelo no estan.

    `hablando` es un Event que el hilo del oido consulta para silenciar el
    microfono mientras suena el altavoz: sin esto el robot se escucha a si
    mismo y entra en bucle.
    """

    def __init__(self, activa: bool = True, dispositivo_alsa: str | None = None):
        self.activa = activa
        self.hablando = threading.Event()
        self._lock = threading.Lock()   # una frase a la vez
        self._motor = None
        self._alsa = dispositivo_alsa   # ej. "plughw:3,0" para unos audifonos USB
        if not activa:
            return

        # piper vive en el bin del entorno de python, que puede no estar en PATH
        piper_env = Path(sys.executable).parent / "piper"
        piper = str(piper_env) if piper_env.exists() else shutil.which("piper")
        if piper and _MODELO_PIPER.exists():
            self._motor = "piper"
            self._piper_bin = piper
        elif shutil.which("espeak-ng"):
            self._motor = "espeak-ng"
        else:
            print("[VOZ] Sin motor de voz (ni piper ni espeak-ng). "
                  "Respuestas solo por texto.", flush=True)
            self.activa = False

        self._reproductor = shutil.which("aplay") or shutil.which("paplay")
        if self._motor == "piper" and not self._reproductor:
            print("[VOZ] Sin reproductor de audio (aplay/paplay). "
                  "Respuestas solo por texto.", flush=True)
            self.activa = False

        if self.activa:
            print(f"[VOZ] Motor de sintesis: {self._motor}", flush=True)

    @staticmethod
    def _limpiar(texto: str) -> str:
        """Quita marcas que no deben leerse en voz alta."""
        texto = re.sub(r"[\*_`#\[\]{}<>|]", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    def decir(self, texto: str) -> None:
        """Bloquea hasta terminar de hablar. Silencia el microfono mientras."""
        if not self.activa:
            return
        texto = self._limpiar(texto)
        if not texto:
            return
        with self._lock:
            self.hablando.set()
            try:
                if self._motor == "piper":
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                        subprocess.run(
                            [self._piper_bin, "--model", str(_MODELO_PIPER),
                             "--output_file", f.name],
                            input=texto.encode(), capture_output=True, timeout=60,
                        )
                        orden = [self._reproductor, "-q"]
                        if self._alsa and self._reproductor.endswith("aplay"):
                            orden += ["-D", self._alsa]
                        subprocess.run(orden + [f.name],
                                       capture_output=True, timeout=120)
                else:  # espeak-ng reproduce directamente
                    subprocess.run(["espeak-ng", "-v", "es", "-s", "165", texto],
                                   capture_output=True, timeout=120)
            except Exception as e:
                print(f"[VOZ] Fallo al hablar: {e}", flush=True)
            finally:
                self.hablando.clear()


# =============================================================================
# BACKEND COMPARTIDO (consola + voz -> un solo Claude, con voz en la salida)
# =============================================================================


class BackendCompartido:
    """Envuelve el backend real para que consola y microfono lo compartan.

    - Un lock serializa las llamadas (el historial del backend no es
      thread-safe y ademas no queremos dos ordenes pisandose).
    - Toda respuesta se dice en voz alta, venga de donde venga.
    """

    def __init__(self, backend, voz: Voz):
        self._backend = backend
        self._voz = voz
        self._lock = threading.Lock()
        self.nombre = backend.nombre
        self.modelo = backend.modelo

    def responder(self, texto_usuario: str):
        with self._lock:
            respuesta = self._backend.responder(texto_usuario)
        self._voz.decir(respuesta.texto)
        return respuesta

    def reiniciar_conversacion(self):
        self._backend.reiniciar_conversacion()


# =============================================================================
# OIDO (microfono -> texto)
# =============================================================================


class Oido(threading.Thread):
    """Escucha el microfono en streaming y manda cada frase al backend.

    Vosk trabaja a 16 kHz mono. Cuando cierra una frase (silencio), el texto
    va al mismo backend que usa la consola. Mientras el robot habla, los
    frames se descartan y el reconocedor se resetea para no oirse a si mismo.
    """

    def __init__(self, backend: BackendCompartido, voz: Voz,
                 parar_evt: threading.Event, palabra_clave: str | None = None,
                 dispositivo: int | None = None):
        super().__init__(name="oido-voz", daemon=True)
        self.backend = backend
        self.voz = voz
        self.parar_evt = parar_evt
        self.palabra_clave = palabra_clave.lower().strip() if palabra_clave else None
        self.dispositivo = dispositivo
        self._pico_ref = 1000.0   # pico reciente para la ganancia automatica

    def run(self):
        try:
            import sounddevice as sd
            import vosk
        except ImportError as e:
            print(f"[OIDO] Falta una dependencia ({e}). Microfono desactivado.\n"
                  "       pip install vosk sounddevice", flush=True)
            return

        if not _MODELO_VOSK.exists():
            print(f"[OIDO] No encuentro el modelo Vosk en {_MODELO_VOSK}.\n"
                  "       Descargalo de https://alphacephei.com/vosk/models "
                  "(vosk-model-small-es-0.42) y descomprimelo ahi.", flush=True)
            return

        vosk.SetLogLevel(-1)
        modelo = vosk.Model(str(_MODELO_VOSK))
        rec = vosk.KaldiRecognizer(modelo, 16000)

        import numpy as np
        from scipy.signal import resample_poly

        import queue as _queue
        cola_audio: _queue.Queue = _queue.Queue(maxsize=32)

        def _callback(indata, frames, tiempo, status):
            # Corre en el hilo de PortAudio: solo encolar, jamas bloquear.
            try:
                cola_audio.put_nowait(bytes(indata))
            except _queue.Full:
                pass

        # Vosk quiere 16 kHz pero cada microfono soporta lo que le da la gana
        # (algunos hasta mienten en su default_samplerate). Sondeamos hasta
        # encontrar una frecuencia que el hardware acepte de verdad.
        try:
            info = sd.query_devices(self.dispositivo, "input")
            sr_defecto = int(info["default_samplerate"])
        except Exception:
            sr_defecto = 48000

        sr_nativa = None
        candidatas = [16000, sr_defecto, 48000, 44100, 32000, 22050, 8000]
        for sr in dict.fromkeys(candidatas):  # dedup conservando orden
            try:
                sd.check_input_settings(device=self.dispositivo, samplerate=sr,
                                        channels=1, dtype="int16")
                sr_nativa = sr
                break
            except Exception:
                continue
        if sr_nativa is None:
            print("[OIDO] El microfono no acepta ninguna frecuencia conocida. "
                  "Microfono desactivado.", flush=True)
            return

        try:
            stream = sd.RawInputStream(
                samplerate=sr_nativa, blocksize=sr_nativa // 4,  # bloques de 250 ms
                dtype="int16", channels=1,
                device=self.dispositivo, callback=_callback,
            )
        except Exception as e:
            print(f"[OIDO] No pude abrir el microfono: {e}\n"
                  "       Prueba `python -m sounddevice` para listar "
                  "dispositivos y pasa --mic-device N.", flush=True)
            return

        if self.palabra_clave:
            print(f"[OIDO] Escuchando. Di '{self.palabra_clave} ...' "
                  "antes de cada orden.", flush=True)
        else:
            print("[OIDO] Escuchando. Habla con normalidad "
                  "(ej: 'camina hacia adelante tres segundos').", flush=True)

        with stream:
            while not self.parar_evt.is_set():
                try:
                    datos = cola_audio.get(timeout=0.25)
                except _queue.Empty:
                    continue

                # Mientras el robot habla, tiramos el audio y reseteamos:
                # lo que entra por el micro es su propia voz.
                if self.voz.hablando.is_set():
                    rec.Reset()
                    continue

                audio = np.frombuffer(datos, dtype=np.int16).astype(np.float32)

                # Ganancia automatica: muchos microfonos USB entregan una
                # senal debil (pico <15% del rango) que Vosk no entiende.
                # Seguimos el pico reciente y amplificamos hacia ~70%.
                pico = float(np.abs(audio).max()) if audio.size else 0.0
                self._pico_ref = max(pico, self._pico_ref * 0.995, 300.0)
                ganancia = min(0.7 * 32767.0 / self._pico_ref, 30.0)
                if ganancia > 1.5:
                    audio = np.clip(audio * ganancia, -32767.0, 32767.0)

                if sr_nativa != 16000:
                    audio = resample_poly(audio, 16000, sr_nativa)
                datos = audio.astype(np.int16).tobytes()

                if not rec.AcceptWaveform(datos):
                    continue

                texto = json.loads(rec.Result()).get("text", "").strip()
                if len(texto) < 3:
                    continue

                if self.palabra_clave:
                    if not texto.startswith(self.palabra_clave):
                        continue
                    texto = texto[len(self.palabra_clave):].strip()
                    if len(texto) < 3:
                        continue

                print(f"\ntu (voz) > {texto}", flush=True)
                try:
                    respuesta = self.backend.responder(texto)
                except Exception as e:
                    print(f"robot > [fallo del backend: {type(e).__name__}: {e}]",
                          flush=True)
                    continue
                print(f"robot > {respuesta.texto}", flush=True)
                # Vaciar lo acumulado durante el turno y volver a escuchar
                while not cola_audio.empty():
                    try:
                        cola_audio.get_nowait()
                    except _queue.Empty:
                        break
                rec.Reset()


# =============================================================================
# MANOS: intenciones
# =============================================================================
# Mismo criterio que las intenciones de play_r1_ia.py: objetos tontos que
# viajan por el bus del hilo del agente al hilo del robot. Nada de MuJoCo aqui.


@dataclass(frozen=True)
class PoseManos:
    """Una pose de dedos del catalogo, para una mano o para las dos."""

    lado: str          # "izquierda" | "derecha" | "ambas"
    pose: str          # clave de amo.control.hand_controller.POSES
    intensidad: float = 1.0


@dataclass(frozen=True)
class SeguirManos:
    """Encender o apagar el seguimiento de dedos por camara."""

    activo: bool


@dataclass(frozen=True)
class SeguirBrazos:
    """Encender o apagar el seguimiento de brazos y cintura por camara."""

    activo: bool


# Pose de dedos que acompana a cada gesto de brazo. Un saludo con el puno
# cerrado queda raro; una guardia con la mano abierta, mas.
DEDOS_POR_GESTO = {
    "wave":    {"izquierda": "relajada", "derecha": "abierta"},
    "point":   {"izquierda": "relajada", "derecha": "senalar"},
    "carry":   {"izquierda": "garra",    "derecha": "garra"},
    "cross":   {"izquierda": "abierta",  "derecha": "abierta"},
    "guard":   {"izquierda": "puno",     "derecha": "puno"},
    "neutral": {"izquierda": "relajada", "derecha": "relajada"},
}


_AYUDA_MANOS = """  Poses     : {poses}
  Por voz   : "cierra la mano derecha" | "abre las manos"
              "haz una pinza" | "pulgar arriba" | "senala"
  Gestos    : cada gesto de brazo mueve tambien los dedos
  Dedos cam : {camara}
"""

_AYUDA_BRAZOS = """  Brazos cam: {brazos}
  Cintura   : {cintura}
"""


def ayuda_tren_superior(con_manos: bool, con_camara: bool, con_brazos: bool,
                        con_cintura: bool = True) -> str:
    """El recuadro de arranque, con solo lo que de verdad esta activo."""
    titulo = ("MANOS Revo2 activas (r1_manos.xml, 22 joints de dedos)"
              if con_manos else "TREN SUPERIOR por camara (r1.xml, sin dedos)")
    cuerpo = ""
    if con_manos:
        cuerpo += _AYUDA_MANOS.format(
            poses=", ".join(sorted(POSES)),
            camara=("ON (di 'copia mis manos' / 'deja de copiar')"
                    if con_camara else "OFF (arranca con --dedos-camara)"),
        )
    cuerpo += _AYUDA_BRAZOS.format(
        brazos=("ON (di 'copia mis brazos' / 'deja de copiar')"
                if con_brazos else "OFF (arranca con --brazos-camara)"),
        cintura=("sigue tu torso (+-0.35 rad)" if con_brazos and con_cintura
                 else "fija" if con_brazos else "-"),
    )
    raya = "-" * 64
    return f"\n{raya}\n  {titulo}\n{raya}\n{cuerpo}{raya}\n"


# =============================================================================
# MANOS: herramientas para el agente
# =============================================================================


class CajaConManos(ia.CajaDeHerramientas):
    """La caja de siempre + las dos herramientas de manos.

    Se hereda en vez de tocar play_r1_ia.py: quien arranque el robot sin manos
    (`--sin-manos`, o el propio play_r1_ia.py) sigue viendo el catalogo
    original, sin herramientas que no puede cumplir.
    """

    def __init__(self, bus, con_camara: bool = False, con_brazos: bool = False):
        # antes de super(): __init__ del padre llama a _construir()
        self.con_camara = con_camara
        self.con_brazos = con_brazos
        super().__init__(bus)

    # -- implementaciones ---------------------------------------------------

    def _mano(self, lado: str = "ambas", pose: str = POSE_REPOSO,
              intensidad: float = 1.0) -> str:
        try:
            lado = normalizar_lado(lado)
        except ValueError:
            return (f"ERROR: '{lado}' no es un lado valido. "
                    "Opciones: izquierda, derecha, ambas.")
        pose = str(pose).strip().lower()
        if pose not in POSES:
            return (f"ERROR: '{pose}' no es una pose de mano valida. "
                    f"Opciones: {', '.join(sorted(POSES))}.")
        intensidad = float(np.clip(intensidad, 0.0, 1.0))

        self.bus.enviar(PoseManos(lado=lado, pose=pose, intensidad=intensidad))
        cual = "ambas manos" if lado == "ambas" else f"la mano {lado}"
        extra = "" if intensidad > 0.99 else f" al {intensidad * 100:.0f}%"
        return f"Poniendo {cual} en '{pose}'{extra}."

    def _seguir_manos(self, activo: bool = True) -> str:
        if not self.con_camara:
            return ("ERROR: no hay camara de manos activa. Hay que arrancar el "
                    "robot con --dedos-camara.")
        activo = bool(activo)
        self.bus.enviar(SeguirManos(activo=activo))
        return ("Copiando tus manos con la camara: mis dedos siguen a los tuyos."
                if activo else
                "Dejo de copiar tus manos; vuelvo a mover los dedos por mi cuenta.")

    def _seguir_brazos(self, activo: bool = True) -> str:
        if not self.con_brazos:
            return ("ERROR: no hay camara de brazos activa. Hay que arrancar el "
                    "robot con --brazos-camara.")
        activo = bool(activo)
        self.bus.enviar(SeguirBrazos(activo=activo))
        return ("Copiando tu tren superior: mis brazos y mi cintura siguen a los "
                "tuyos." if activo else
                "Dejo de copiar tus brazos; los devuelvo a la politica de caminar.")

    # -- catalogo ------------------------------------------------------------

    def _construir(self) -> list:
        herramientas = super()._construir()
        herramientas.append(ia.Herramienta(
            nombre="mano",
            descripcion=(
                "Mueve los dedos de una mano (o de las dos) a una pose del "
                f"catalogo. {DESCRIPCION_POSES}. Usala cuando te pidan cerrar, "
                "abrir, apretar o hacer una senal con la mano."
            ),
            esquema={
                "type": "object",
                "properties": {
                    "lado": {
                        "type": "string",
                        "enum": ["izquierda", "derecha", "ambas"],
                        "description": "Que mano mover. Si no lo dicen, usa 'ambas'.",
                    },
                    "pose": {
                        "type": "string",
                        "enum": sorted(POSES),
                        "description": "Pose de dedos a adoptar.",
                    },
                    "intensidad": {
                        "type": "number",
                        "description": ("Cuanto aplicar la pose, de 0 a 1. 1 es la "
                                        "pose completa, 0.5 la deja a medias "
                                        "(util para 'cierra un poco la mano')."),
                    },
                },
                "required": ["pose"],
            },
            ejecutar=self._mano,
        ))
        if self.con_camara:
            herramientas.append(ia.Herramienta(
                nombre="seguir_manos",
                descripcion=(
                    "Enciende o apaga el seguimiento de manos por camara. Con el "
                    "seguimiento encendido mis dedos copian en tiempo real los "
                    "dedos del usuario. Usala si te piden que copies, imites o "
                    "sigas sus manos, o que dejes de hacerlo."
                ),
                esquema={
                    "type": "object",
                    "properties": {
                        "activo": {
                            "type": "boolean",
                            "description": "True para empezar a copiar, False para dejarlo.",
                        }
                    },
                    "required": ["activo"],
                },
                ejecutar=self._seguir_manos,
            ))
        if self.con_brazos:
            herramientas.append(ia.Herramienta(
                nombre="seguir_brazos",
                descripcion=(
                    "Enciende o apaga el seguimiento de BRAZOS por camara. Con el "
                    "encendido mis hombros, codos y mi cintura copian en tiempo "
                    "real los movimientos del usuario. Usala si te piden que "
                    "copies, imites o sigas sus brazos o su cuerpo, o que dejes "
                    "de hacerlo. Es independiente de `seguir_manos`, que solo "
                    "mueve los dedos."
                ),
                esquema={
                    "type": "object",
                    "properties": {
                        "activo": {
                            "type": "boolean",
                            "description": "True para empezar a copiar, False para dejarlo.",
                        }
                    },
                    "required": ["activo"],
                },
                ejecutar=self._seguir_brazos,
            ))
        return herramientas


# =============================================================================
# MANOS: instrucciones del sistema
# =============================================================================
# El prompt de play_r1_ia.py dice literalmente "no tienes manos que agarren, no
# ves con camara". Si no lo corregimos, el modelo se niega a usar las
# herramientas nuevas aunque las tenga delante.

_LIMITE_VIEJO = ("- Solo caminas y haces los gestos del catalogo. No tienes manos "
                 "que agarren, no puedes subir escaleras, no ves con camara.")


def instrucciones_con_manos(con_camara: bool, con_brazos: bool = False) -> str:
    nuevo = (
        "- Caminas, haces los gestos de brazo del catalogo y mueves los dedos de "
        "tus dos manos Revo2 con el catalogo de poses (herramienta `mano`). Cada "
        "gesto de brazo ya mueve los dedos que le tocan, no hace falta que los "
        "pongas tu aparte. No puedes subir escaleras ni levantar objetos pesados."
    )
    if con_camara or con_brazos:
        partes = []
        if con_camara:
            partes.append("sus dedos con `seguir_manos`")
        if con_brazos:
            partes.append("sus brazos y su cintura con `seguir_brazos`")
        nuevo += (
            "\n- Tienes una camara que ve al usuario. Puedes copiar en tiempo real "
            + " y ".join(partes) +
            ". No ves nada mas: ni caras, ni objetos, ni el entorno."
        )
    else:
        nuevo += "\n- No ves con camara."

    texto = ia.INSTRUCCIONES_SISTEMA
    if _LIMITE_VIEJO in texto:
        return texto.replace(_LIMITE_VIEJO, nuevo)
    return texto + "\n" + nuevo   # el prompt cambio: al menos no mentimos


# =============================================================================
# MANOS: backend simulado
# =============================================================================


class ManosPorPalabrasClave:
    """Envuelve el backend simulado para que tambien entienda las manos.

    Solo se usa con `--backend simulado`, que no es un LLM sino un interprete
    de palabras clave: si no le ensenamos las frases de manos, el modo por
    defecto (sin API key) no podria mover un dedo. Con un backend real esto no
    se usa: el modelo ya ve la herramienta `mano` y decide solo.
    """

    # (patron de la orden, pose)
    PATRONES = [
        (r"pu[nñ]o|cierra|aprieta|cerrad", "puno"),
        (r"abre|abierta|extiende los dedos|palma|estira los dedos", "abierta"),
        (r"pinza|pellizc", "pinza"),
        (r"\bok\b|okey|de acuerdo", "ok"),
        (r"se[nñ]ala|apunta con el dedo|[ií]ndice", "senalar"),
        (r"pulgar", "pulgar_arriba"),
        (r"\bpaz\b|victoria", "paz"),
        (r"garra|agarra|sujeta|sostiene|envuelve", "garra"),
        (r"relaja|reposo|suelta|descansa", "relajada"),
    ]

    # El sujeto de la orden: sin esto, "abre los brazos" acabaria moviendo dedos.
    _SUJETO = (r"manos?|dedos?|pu[nñ]os?|palmas?|pulgar(?:es)?|"
               r"[ií]ndice|me[nñ]ique|anular")
    # Sujeto de las ordenes de camara para el tren superior.
    SUJETO_BRAZOS = re.compile(r"\b(brazos?|hombros?|codos?|cintura|torso|"
                               r"cuerpo|movimientos?)\b")
    SUJETO = re.compile(rf"\b(?:{_SUJETO})\b")
    LADO = re.compile(rf"\b(?:{_SUJETO})\s+"
                      r"(izquierdo?a?|derecho?a?|ambas|las dos)\b")
    FRASE = re.compile(r"\b(?:la |las |mi |mis |tu |tus |ambas |los |el )?"
                       rf"(?:{_SUJETO})"
                       r"(?:\s+(?:izquierdo?a?|derecho?a?|ambas|las dos))?")
    CAMARA_ON = re.compile(r"copia|imita|sigue mis|imit[ae]")
    CAMARA_OFF = re.compile(r"deja de (copiar|imitar|seguir)|para de (copiar|imitar)|"
                            r"no (me )?copies|suelta el control")
    MOVIMIENTO = re.compile(r"adelante|avanza|camina|atras|retrocede|gira|voltea|"
                            r"rota|lateral|salud|apunt|carg|cruz|guardia|box|"
                            r"neutral|donde|estado|para\b|detente|reinicia")

    def __init__(self, backend, caja: ia.CajaDeHerramientas):
        self._backend = backend
        self._caja = caja
        self.nombre = backend.nombre
        self.modelo = backend.modelo

    def reiniciar_conversacion(self) -> None:
        self._backend.reiniciar_conversacion()

    # -- interpretacion ------------------------------------------------------

    def _lado(self, t: str) -> str:
        m = self.LADO.search(t)
        if not m:
            return "ambas"
        palabra = m.group(1)
        if palabra.startswith("izquierd"):
            return "izquierda"
        if palabra.startswith("derech"):
            return "derecha"
        return "ambas"

    def _pose(self, t: str) -> str | None:
        for patron, pose in self.PATRONES:
            if re.search(patron, t):
                return pose
        return None

    def responder(self, texto_usuario: str):
        t = texto_usuario.lower().strip()

        # -- camara: copiar / dejar de copiar --------------------------------
        # "deja de copiar" ya es inequivoco; "copia..." pide sujeto para no
        # confundirlo con cualquier otra cosa que se pueda copiar. Sin sujeto,
        # apagar apaga todo lo que estuviera copiando.
        apagar = bool(self.CAMARA_OFF.search(t))
        encender = bool(self.CAMARA_ON.search(t))
        quiere_manos = bool(self.SUJETO.search(t))
        quiere_brazos = bool(self.SUJETO_BRAZOS.search(t))

        if apagar or (encender and (quiere_manos or quiere_brazos)):
            if not (quiere_manos or quiere_brazos):
                quiere_manos = quiere_brazos = True   # "deja de copiar", a secas
            activo = not apagar
            pedidas = []
            if quiere_manos:
                pedidas.append("seguir_manos")
            if quiere_brazos:
                pedidas.append("seguir_brazos")

            disponibles = [h for h in pedidas if h in self._caja.indice]
            if not disponibles:
                que = "manos" if pedidas == ["seguir_manos"] else "brazos"
                flag = "--dedos-camara" if que == "manos" else "--brazos-camara"
                return ia.RespuestaLLM(
                    f"No tengo la camara de {que} encendida. Arrancame con "
                    f"{flag} si quieres que copie tu tren superior.", [])

            salidas = [self._caja.despachar(h, {"activo": activo}) for h in disponibles]
            return ia.RespuestaLLM(" ".join(salidas), disponibles)

        # -- poses de dedos ---------------------------------------------------
        pose = self._pose(t) if self.SUJETO.search(t) else None
        if pose is None:
            return self._backend.responder(texto_usuario)

        salida = self._caja.despachar(
            "mano", {"lado": self._lado(t), "pose": pose, "intensidad": 1.0})
        partes, usadas = [salida], ["mano"]

        # Le quitamos la frase de manos y, si queda una orden de verdad, se la
        # pasamos al backend de siempre ("camina y cierra la mano" -> las dos).
        resto = self.FRASE.sub(" ", t)
        if self.MOVIMIENTO.search(resto):
            respuesta = self._backend.responder(resto)
            if respuesta.herramientas_usadas:
                partes.append(respuesta.texto)
                usadas += respuesta.herramientas_usadas

        return ia.RespuestaLLM(" ".join(partes), usadas)


# =============================================================================
# MANOS: el entorno del robot
# =============================================================================


def construir_clase_entorno_manos(clase_con_ia, base_mod):
    """Anade las manos Revo2 a la clase de entorno de play_r1_ia.py.

    Dos cosas que hay que arreglar al cargar `r1_manos.xml`:

    1. Los INDICES. Con manos, el XML mete los 11 joints de cada mano justo
       detras de su muneca, asi que los 24 joints del cuerpo dejan de estar en
       qpos[7:31]: el brazo derecho se va a qpos[37:42]. Los actuadores en
       cambio no se mueven (los 24 del cuerpo siguen siendo ctrl[0:24]), por
       eso solo hay que resolver qpos/qvel por nombre.

    2. Los DEDOS. Son 22 DOFs que la politica de caminar no conoce ni toca;
       los lleva un PD aparte (ControladorManos) a la misma frecuencia de
       simulacion.
    """

    # Posicion del waist_yaw dentro del vector de 24 objetivos (orden MuJoCo).
    IDX_WAIST_YAW = 13

    JOINTS_CUERPO = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_roll_joint", "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    ]

    class R1ConManos(clase_con_ia):
        """El R1 con IA de play_r1_ia.py, ahora con dedos."""

        def __init__(self, bus, seguidor=None, brazos: ControladorBrazosCamara | None = None,
                     seguir_dedos: bool | None = None, **kwargs):
            # Estos tres antes de super(): el __init__ del padre acaba llamando
            # a reset(), que ya consulta las manos y los brazos.
            self.manos = None
            self.brazos = brazos
            self._qadr = None
            self._vadr = None

            self.seguidor = seguidor
            if seguir_dedos is None:
                seguir_dedos = seguidor is not None
            self.seguir_camara = bool(seguir_dedos) and seguidor is not None
            self._gesto_dedos = False
            self._pose_manual = {"izquierda": POSE_REPOSO, "derecha": POSE_REPOSO}

            kwargs.setdefault("xml_path", scene("r1_manos"))
            super().__init__(bus, **kwargs)

            self._resolver_cuerpo()
            self.manos = ControladorManos(self.model, self.data)
            if self.manos.disponible:
                print(f"[MANOS] {len(self.manos.qadr)} joints de dedos listos.",
                      flush=True)
            else:
                print(f"[MANOS] El modelo no trae manos ({self.manos.motivo}); "
                      "los dedos quedan desactivados.", flush=True)
            self.reset()   # ahora si, con los indices y las manos en su sitio

        # -- indices por nombre ----------------------------------------------

        def _resolver_cuerpo(self):
            qadr, vadr = [], []
            for nombre in JOINTS_CUERPO:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre)
                if jid < 0:
                    raise RuntimeError(f"el modelo no tiene el joint '{nombre}'")
                qadr.append(int(self.model.jnt_qposadr[jid]))
                vadr.append(int(self.model.jnt_dofadr[jid]))
            self._qadr = np.array(qadr)
            self._vadr = np.array(vadr)

        # -- lo que cambia por los indices ------------------------------------

        def reset(self):
            super().reset()
            if self._qadr is None:
                return   # todavia estamos dentro del __init__ del padre
            # El padre escribio la pose en qpos[7:31], que con manos ya no es
            # el cuerpo. La reescribimos donde toca y colocamos los dedos.
            self.data.qpos[self._qadr] = base_mod.DEFAULT_MJ
            if self.manos is not None:
                self.manos.reiniciar()
                self._pose_manual = {"izquierda": POSE_REPOSO, "derecha": POSE_REPOSO}
                self._gesto_dedos = False
            if self.brazos is not None:
                # Sin esto, tras un reset los brazos entrarian desde la ultima
                # pose copiada y darian un tiron contra la pose por defecto.
                self.brazos.reiniciar()
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

        def _read_state(self):
            qpos_mj = self.data.qpos[self._qadr].astype(np.float32)
            qvel_mj = self.data.qvel[self._vadr].astype(np.float32)
            quat = self.data.qpos[3:7].astype(np.float32)
            if self._has_gyro:
                ang_vel = self.data.sensor("imu_ang_vel").data.astype(np.float32)
            else:
                ang_vel = self.data.qvel[3:6].astype(np.float32)
            return qpos_mj, qvel_mj, quat, ang_vel

        def _apply_pd(self, target_mj):
            qpos_mj = self.data.qpos[self._qadr]
            qvel_mj = self.data.qvel[self._vadr]
            torque = (base_mod.STIFFNESS_MJ * (target_mj - qpos_mj)
                      - base_mod.DAMPING_MJ * qvel_mj)
            self.data.ctrl[:base_mod.NUM_DOFS] = np.clip(
                torque, -base_mod.TORQUE_LIMITS_MJ, base_mod.TORQUE_LIMITS_MJ)
            # Los dedos van en el mismo paso de simulacion, con su propio PD.
            if self.manos is not None:
                self.manos.aplicar_torques()

        # -- intenciones de manos ---------------------------------------------

        def _procesar_intenciones(self):
            # El padre drena el bus y descarta lo que no conoce, asi que
            # separamos primero lo nuestro y le devolvemos el resto en orden.
            pendientes = self.bus.drenar()
            mias, otras = [], []
            for intencion in pendientes:
                if isinstance(intencion, (PoseManos, SeguirManos, SeguirBrazos)):
                    mias.append(intencion)
                else:
                    otras.append(intencion)
                    if isinstance(intencion, ia.Gesto):
                        self._dedos_de_gesto(intencion.nombre)
            for intencion in otras:
                self.bus.enviar(intencion)

            super()._procesar_intenciones()

            for intencion in mias:
                self._aplicar_intencion_manos(intencion)

        def _aplicar_intencion_manos(self, intencion):
            if isinstance(intencion, SeguirBrazos):
                if self.brazos is None:
                    return
                if intencion.activo:
                    self.brazos.encender()
                else:
                    self.brazos.apagar()
                return
            if self.manos is None or not self.manos.disponible:
                return
            if isinstance(intencion, PoseManos):
                self.manos.poner_pose(intencion.lado, intencion.pose,
                                      intencion.intensidad)
                for lado in self.manos.lados(intencion.lado):
                    self._pose_manual[lado] = intencion.pose
                self._gesto_dedos = False    # una orden directa manda sobre el gesto
            elif isinstance(intencion, SeguirManos):
                self.seguir_camara = bool(intencion.activo) and self.seguidor is not None
                if not self.seguir_camara:
                    for lado, pose in self._pose_manual.items():
                        self.manos.poner_pose(lado, pose)

        def _dedos_de_gesto(self, nombre: str):
            poses = DEDOS_POR_GESTO.get(nombre)
            if not poses or self.manos is None or not self.manos.disponible:
                return
            for lado, pose in poses.items():
                self.manos.poner_pose(lado, pose)
            self._gesto_dedos = True

        # -- bucle -------------------------------------------------------------

        def _gesto_en_curso(self) -> bool:
            seq = self.arm_ctrl._active_seq
            return seq is not None and seq.active

        def _policy_step(self):
            objetivo = super()._policy_step()   # intenciones + telemetria + politica
            objetivo = self._actualizar_brazos(objetivo)
            self._actualizar_dedos()
            return objetivo

        def _actualizar_brazos(self, objetivo):
            """Mezcla los brazos de la camara con los que pide la politica.

            La IK vive en su propio hilo; aqui solo interpolamos. La mezcla
            sube y baja sola dentro del controlador, asi que mientras este a 0
            esto no toca nada y el robot se comporta como si no hubiera camara.
            Igual que con los dedos, la camara manda sobre el gesto de brazo.
            """
            if self.brazos is None:
                return objetivo
            self.brazos.avanzar()
            if self.brazos.mezcla <= 0.0:
                return objetivo
            objetivo = objetivo.copy()
            objetivo[base_mod.ARM_IDX] = self.brazos.mezclar(objetivo[base_mod.ARM_IDX])
            objetivo[IDX_WAIST_YAW] = self.brazos.mezclar_cintura(
                objetivo[IDX_WAIST_YAW])
            return objetivo.astype(np.float32)

        def _actualizar_dedos(self):
            if self.manos is None or not self.manos.disponible:
                return

            # 1) la camara manda mientras se te vea la mano
            if self.seguir_camara and self.seguidor is not None:
                for lado, curls in self.seguidor.leer().items():
                    if curls is not None:
                        self.manos.poner_curls(lado, curls, nombre="camara")

            # 2) al acabar el gesto de brazo, los dedos vuelven a lo que habia
            if self._gesto_dedos and not self._gesto_en_curso():
                for lado, pose in self._pose_manual.items():
                    self.manos.poner_pose(lado, pose)
                self._gesto_dedos = False

            self.manos.avanzar()

        def _publicar_telemetria(self):
            super()._publicar_telemetria()
            if self._contador_snapshot != 0 or self.manos is None:
                return   # el padre no ha publicado en esta vuelta
            snapshot = self.bus.leer_snapshot()
            snapshot["manos"] = self.manos.resumen()
            if self.seguidor is not None:
                snapshot["camara_manos"] = {
                    "siguiendo": self.seguir_camara,
                    "estado": self.seguidor.estado,
                }
            if self.brazos is not None:
                snapshot["camara_brazos"] = self.brazos.resumen()
            self.bus.publicar_snapshot(snapshot)

    return R1ConManos


# =============================================================================
# MAIN
# =============================================================================


def parsear_argumentos():
    p = argparse.ArgumentParser(
        description="R1 controlado por voz y consola, con respuestas habladas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", default=os.environ.get("AMO_LLM_BACKEND", "simulado"),
                   choices=sorted(ia.BACKENDS),
                   help="Proveedor de LLM (por defecto el de AMO_LLM_BACKEND o 'simulado').")
    p.add_argument("--modelo", default=os.environ.get("AMO_LLM_MODELO"),
                   help="Id del modelo LLM.")
    p.add_argument("--sin-sim", action="store_true",
                   help="No levanta MuJoCo. Solo voz + chat, para probar el audio.")
    p.add_argument("--sin-tts", action="store_true",
                   help="No habla: respuestas solo por texto.")
    p.add_argument("--sin-mic", action="store_true",
                   help="No escucha: solo consola (pero si habla las respuestas).")
    p.add_argument("--palabra-clave", default=None, metavar="PALABRA",
                   help="Solo atiende frases que empiecen por esta palabra (ej: robot).")
    p.add_argument("--mic-device", type=int, default=None, metavar="N",
                   help="Indice del dispositivo de entrada (ver `python -m sounddevice`). "
                        "Por defecto se autodetecta un headset USB si hay uno.")
    p.add_argument("--altavoz-device", default=None, metavar="ALSA",
                   help="Dispositivo ALSA de salida para la voz (ej: plughw:3,0). "
                        "Por defecto se autodetecta un headset USB si hay uno.")
    p.add_argument("--device", default="cpu", help="cpu o cuda para la politica del robot.")

    manos = p.add_argument_group("tren superior")
    manos.add_argument("--sin-manos", action="store_true",
                       help="Carga r1.xml (sin manos) en vez de r1_manos.xml. "
                            "El robot pierde los 22 joints de dedos.")
    manos.add_argument("--dedos-camara", action="store_true",
                       help="Enciende el seguimiento de dedos por camara "
                            "(MediaPipe): mis dedos copian los tuyos.")
    manos.add_argument("--brazos-camara", action="store_true",
                       help="Enciende el seguimiento de BRAZOS por camara "
                            "(MediaPipe Holistic + IK): mis hombros, codos y "
                            "cintura copian los tuyos. Sirve tambien los dedos, "
                            "asi que basta una camara para todo.")
    manos.add_argument("--sin-cintura", action="store_true",
                       help="Con --brazos-camara, copia solo los brazos: el "
                            "waist_yaw se queda quieto (mas estable al caminar).")
    manos.add_argument("--camara", default=FUENTE_POR_DEFECTO, metavar="FUENTE",
                       help="Indice de webcam (ej: 0) o URL de la camara. "
                            f"Por defecto {FUENTE_POR_DEFECTO} (DroidCam).")
    manos.add_argument("--ver-camara", action="store_true",
                       help="Abre una ventana con lo que ve la camara y los curls.")
    manos.add_argument("--sin-espejo", action="store_true",
                       help="Copia lado a lado en vez de como un espejo "
                            "(por defecto mi mano izquierda sigue a tu derecha).")
    return p.parse_args()


def main():
    args = parsear_argumentos()

    con_manos = not args.sin_manos
    con_camara = args.dedos_camara and con_manos
    con_brazos = args.brazos_camara
    if args.dedos_camara and not con_manos:
        print("[MANOS] --dedos-camara no tiene sentido con --sin-manos; "
              "ignorado.", flush=True)

    bus = ia.BusDeIntenciones()
    if con_manos or con_brazos:
        caja = CajaConManos(bus, con_camara=con_camara, con_brazos=con_brazos)
        # El prompt original dice "no tienes manos"; ahora si las tiene.
        ia.INSTRUCCIONES_SISTEMA = instrucciones_con_manos(con_camara, con_brazos)
    else:
        caja = ia.CajaDeHerramientas(bus)

    print(f"Inicializando backend '{args.backend}'...", flush=True)
    backend_real = ia.BACKENDS[args.backend](caja, modelo=args.modelo)

    # El backend simulado es un interprete de palabras clave: hay que ensenarle
    # las frases de manos a mano. Los backends de verdad ya ven la herramienta.
    if (con_manos or con_brazos) and isinstance(backend_real, ia.BackendSimulado):
        backend_real = ManosPorPalabrasClave(backend_real, caja)

    # Si hay un headset USB conectado, usarlo por defecto (micro y audifonos):
    # los flags --mic-device / --altavoz-device siempre tienen prioridad.
    mic_device, altavoz = args.mic_device, args.altavoz_device
    if mic_device is None or altavoz is None:
        mic_usb, salida_usb = _detectar_headset_usb()
        if mic_device is None and mic_usb is not None:
            mic_device = mic_usb
            print(f"[AUDIO] Microfono USB detectado (dispositivo {mic_usb}).", flush=True)
        if altavoz is None and salida_usb is not None:
            altavoz = salida_usb
            print(f"[AUDIO] Audifonos USB detectados ({salida_usb}).", flush=True)

    voz = Voz(activa=not args.sin_tts, dispositivo_alsa=altavoz)
    backend = BackendCompartido(backend_real, voz)
    parar_evt = threading.Event()

    # ---- camara del tren superior ------------------------------------------
    # Un solo stream y un solo detector: si hacen falta brazos usamos Holistic,
    # que ademas trae los dedos; si solo hacen falta dedos, Hands es mas ligero.
    seguidor = None
    if con_brazos:
        seguidor = SeguidorTrenSuperior(fuente=args.camara,
                                        espejo=not args.sin_espejo,
                                        mostrar=args.ver_camara)
        seguidor.start()
    elif con_camara:
        seguidor = SeguidorDeManos(fuente=args.camara,
                                   espejo=not args.sin_espejo,
                                   mostrar=args.ver_camara)
        seguidor.start()

    if not args.sin_mic:
        Oido(backend, voz, parar_evt,
             palabra_clave=args.palabra_clave,
             dispositivo=mic_device).start()

    # ---- modo solo-audio: probar voz y oido sin levantar MuJoCo ------------
    if args.sin_sim:
        print("\n[MODO SIN SIMULACION] Voz y chat activos; no hay robot que\n"
              "consuma las intenciones. Sirve para probar microfono y altavoz.\n")
        bus.publicar_snapshot({
            "listo": True, "posicion_m": [0.0, 0.0, 0.74],
            "orientacion_grados": 0.0, "velocidad_ms": 0.0,
            "comando_actual": [0.0, 0.0, 0.0], "gesto_activo": None,
            "altura_pelvis_m": 0.74, "caido": False, "tiempo_sim_s": 0.0,
            "nota": "simulacion no activa (--sin-sim)",
        })
        voz.decir("Hola, te escucho.")
        try:
            ia.Consola(backend, bus, parar_evt).run()   # en el hilo principal
        finally:
            if seguidor is not None:
                seguidor.detener()
        return

    # ---- modo normal: simulacion + voz + consola ---------------------------
    print("Cargando entorno base (play_r1_camina_brazos.py)...", flush=True)
    base = ia._cargar_entorno_base()
    R1ConIA = ia.construir_clase_entorno(base.R1CaminaBrazos)

    brazos = None
    if con_brazos:
        # Sin xml_path: la IK usa r1.xml aunque el robot lleve manos (misma
        # cinematica de brazo, modelo mas barato). Ver amo/control/arm_ik.py.
        brazos = ControladorBrazosCamara(seguidor,
                                         seguir_cintura=not args.sin_cintura,
                                         activo=True)
        brazos.start()
        print(f"[BRAZOS] IK de brazos lista"
              f"{'' if brazos.seguir_cintura else ' (cintura fija)'}.", flush=True)

    if con_manos or con_brazos:
        R1Final = construir_clase_entorno_manos(R1ConIA, base)
        kwargs = {} if con_manos else {"xml_path": scene("r1")}
        entorno = R1Final(bus, seguidor=seguidor, brazos=brazos,
                          seguir_dedos=con_camara, device=args.device, **kwargs)
        print(ayuda_tren_superior(con_manos, con_camara, con_brazos,
                                  con_cintura=not args.sin_cintura), flush=True)
    else:
        entorno = R1ConIA(bus, device=args.device)

    consola = ia.Consola(backend, bus, parar_evt)
    consola.start()
    voz.decir("Hola, te escucho.")

    try:
        entorno.run()
    finally:
        parar_evt.set()
        if brazos is not None:
            brazos.detener()
        if seguidor is not None:
            seguidor.detener()
        print("\nAgente detenido. Pulsa Enter si la consola sigue abierta.", flush=True)


if __name__ == "__main__":
    main()
