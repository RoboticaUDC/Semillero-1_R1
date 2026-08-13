#!/usr/bin/env python3
"""
play_r1_voz.py — R1 controlado por VOZ y por consola, con respuestas habladas.

QUE HACE
========
Toma TODO lo de `play_r1_ia.py` (robot + Claude + herramientas + consola) y le
anade una capa de voz completamente offline:

    tu hablas   ->  microfono -> Vosk (reconocimiento en espanol)
    Claude      ->  decide que herramientas llamar (igual que por texto)
    el robot    ->  ejecuta la accion en MuJoCo
    la voz      ->  Piper sintetiza la respuesta y suena por los altavoces

La consola sigue funcionando en paralelo: puedes hablarle O escribirle, y
siempre contesta por texto Y por voz. Un comando hablado ("camina hacia
adelante tres segundos") ejecuta la accion exactamente igual que uno escrito.

ARQUITECTURA
============
    [hilo oido]     microfono 16 kHz -> Vosk -> texto final
                          |                          (se silencia mientras
                          v                           el robot habla, para
    [BackendCompartido]  un lock: consola y voz       no escucharse a si
                          comparten el mismo Claude    mismo)
                          |
                          v
    [hilo consola]  input() de siempre  ->  mismo backend
                          |
                          v
    [Voz]           Piper -> wav -> aplay   (espeak-ng si Piper falta)

El hilo del robot (MuJoCo a 500 Hz) no se toca: sigue recibiendo intenciones
por el mismo BusDeIntenciones de play_r1_ia.py.

REQUISITOS (ya instalados si seguiste el setup)
===============================================
    pip install vosk sounddevice piper-tts
    conda install -c conda-forge portaudio
    modelos en <raiz>/modelos_voz/:
        vosk-model-small-es-0.42/          (reconocimiento, ~40 MB)
        es_ES-davefx-medium.onnx (+.json)  (sintesis, ~60 MB)

USO
===
    conda activate r1mujoco

    # robot completo + voz + consola:
    python scripts/r1/play_r1_voz.py

    # probar solo la voz, sin levantar MuJoCo:
    python scripts/r1/play_r1_voz.py --sin-sim

    # solo escuchar cuando la frase empiece por una palabra clave:
    python scripts/r1/play_r1_voz.py --palabra-clave robot

    # sin sintesis de voz (solo escucha) / sin microfono (solo habla):
    python scripts/r1/play_r1_voz.py --sin-tts
    python scripts/r1/play_r1_voz.py --sin-mic

NO MODIFICA NINGUN ARCHIVO EXISTENTE.
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
    return p.parse_args()


def main():
    args = parsear_argumentos()

    bus = ia.BusDeIntenciones()
    caja = ia.CajaDeHerramientas(bus)

    print(f"Inicializando backend '{args.backend}'...", flush=True)
    backend_real = ia.BACKENDS[args.backend](caja, modelo=args.modelo)

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
        ia.Consola(backend, bus, parar_evt).run()   # en el hilo principal
        return

    # ---- modo normal: simulacion + voz + consola ---------------------------
    print("Cargando entorno base (play_r1_camina_brazos.py)...", flush=True)
    base = ia._cargar_entorno_base()
    R1ConIA = ia.construir_clase_entorno(base.R1CaminaBrazos)

    entorno = R1ConIA(bus, device=args.device)

    consola = ia.Consola(backend, bus, parar_evt)
    consola.start()
    voz.decir("Hola, te escucho.")

    try:
        entorno.run()
    finally:
        parar_evt.set()
        print("\nAgente detenido. Pulsa Enter si la consola sigue abierta.", flush=True)


if __name__ == "__main__":
    main()
