"""
cuerpo_camara.py — seguimiento del TREN SUPERIOR con la camara (MediaPipe
Holistic): brazos, giro del torso y, de paso, los dedos.

De donde sale
=============
Es la parte de percepcion de `scripts/teleop/teleop_dedos.py` (clase
PoseTracker), extraida para que la puedan usar los scripts que no son de
teleoperacion. La aritmetica es la misma; lo que cambia es que aqui el espejo
y el cambio de lado se aplican DENTRO del seguidor, asi que quien lo consume
recibe todo ya en terminos del ROBOT ("izquierda" es la izquierda del robot).

Por que Holistic y no Hands
===========================
`manos_camara.SeguidorDeManos` usa `mp.solutions.hands`, que es mas ligero
pero solo ve manos. Para los brazos hacen falta los landmarks de cuerpo, y
Holistic entrega cuerpo + las dos manos en UNA pasada, con cada mano atada a
su muneca. Como un mismo stream MJPEG no se puede abrir dos veces sin pelearse,
cuando quieres brazos este seguidor sustituye al de manos y sirve los dos.

Que publica
===========
    leer()         -> {"izquierda": curls|None, "derecha": curls|None}
                      (misma firma que SeguidorDeManos: es intercambiable)
    leer_brazos()  -> LecturaBrazos | None

`LecturaBrazos.dirs` son, por lado del robot, dos vectores unitarios en el
marco del CUERPO (x adelante, y izquierda, z arriba):

    u = hombro  -> codo    (brazo)
    w = codo    -> muneca  (antebrazo)

que es exactamente lo que la IK de `amo.control.arm_ik` sabe resolver.

`LecturaBrazos.ganancia` dice cuanto fiarse de cada lado, en [0, 1]. Sale de
cuanto se sale el brazo del plano de la imagen: un brazo apuntando a la camara
se ve como un punto y su direccion es ruido puro. 0 = ignorame, 1 = fiate.

`LecturaBrazos.yaw` es el giro del torso en radianes, ya con el espejo
aplicado y listo para el waist_yaw del robot.

Hilos: el bucle de captura corre en su propio hilo. `leer()` y `leer_brazos()`
devuelven copias bajo lock, asi que se pueden llamar desde el hilo de la
simulacion sin sincronizar nada mas.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from amo.vision.manos_camara import FUENTE_POR_DEFECTO, curls_desde_landmarks

# Landmarks de pose de MediaPipe que nos interesan.
_HOMBRO_I, _HOMBRO_D = 11, 12
_CODO_I, _CODO_D = 13, 14
_MUNECA_I, _MUNECA_D = 15, 16
_CADERA_I, _CADERA_D = 23, 24

# El eje "adelante" sale del producto cruz izquierda x arriba; con el volteo
# de la imagen queda mirando hacia atras, de ahi el signo.
SIGNO_ADELANTE = -1.0

# Ventana de confianza del brazo: por debajo de LO no hacemos caso, por encima
# de HI hacemos caso del todo, y en medio se interpola suave. Mide cuanto del
# segmento vive en el plano de la imagen; lo que apunta a la camara depende
# solo de la profundidad, que es justo lo que MediaPipe estima peor.
#
# teleop_dedos.py usaba 0.75/1.00, que en la practica cierra la puerta en
# cuanto extiendes un brazo hacia la camara (con 41 grados fuera del plano ya
# ignora el brazo entero) y de pie frente al escritorio no copiaba casi nada.
# Con 0.50/0.85 solo se descarta lo que apunta a menos de 30 grados del eje de
# la camara, que es donde la direccion es de verdad ruido.
CONF_LO = 0.50
CONF_HI = 0.85

# Visibilidad minima (la que reporta MediaPipe por landmark) para creernos que
# hay alguien. Sin esto, con la sala vacia el modelo de pose se inventa un
# esqueleto de vez en cuando y el robot mueve los brazos hacia un fantasma.
VIS_MIN_TRONCO = 0.5


@dataclass(frozen=True)
class LecturaBrazos:
    """Lo que ve la camara del tren superior, ya en terminos del robot."""

    dirs: dict[str, tuple[np.ndarray, np.ndarray]]   # lado -> (brazo, antebrazo)
    ganancia: dict[str, float]                       # lado -> [0, 1]
    yaw: float                                       # giro del torso, rad


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _en_plano(v: np.ndarray) -> float:
    """Cuanto del vector unitario vive en el plano de la imagen, en [0, 1]."""
    return float(np.sqrt(max(0.0, 1.0 - float(v[2]) ** 2)))


def _rampa(x: float, lo: float, hi: float) -> float:
    """Smoothstep: 0 por debajo de lo, 1 por encima de hi, suave en medio."""
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    t = (x - lo) / (hi - lo)
    return float(t * t * (3 - 2 * t))


def _espejar(v: np.ndarray) -> np.ndarray:
    """Refleja un vector del marco del cuerpo en el plano sagital."""
    return np.array([v[0], -v[1], v[2]])


class SeguidorTrenSuperior(threading.Thread):
    """Lee la camara y publica brazos, giro de torso y dedos.

    Intercambiable con `SeguidorDeManos`: implementa `leer()`, `estado`,
    `ve_manos` y `detener()` con el mismo significado, y ademas
    `leer_brazos()`.
    """

    def __init__(self, fuente=FUENTE_POR_DEFECTO, espejo: bool = True,
                 mostrar: bool = False, confianza: float = 0.6,
                 nombre_ventana: str = "Tren superior R1"):
        super().__init__(name="seguidor-tren-superior", daemon=True)
        self.fuente = int(fuente) if str(fuente).isdigit() else fuente
        self.espejo = espejo
        self.mostrar = mostrar
        self.confianza = confianza
        self.nombre_ventana = nombre_ventana

        self._lock = threading.Lock()
        self._curls: dict[str, np.ndarray | None] = {"izquierda": None, "derecha": None}
        self._brazos: LecturaBrazos | None = None
        self._estado = "arrancando"
        self._corriendo = True

    # -- API -----------------------------------------------------------------

    def leer(self) -> dict[str, np.ndarray | None]:
        with self._lock:
            return {k: (None if v is None else v.copy()) for k, v in self._curls.items()}

    def leer_brazos(self) -> LecturaBrazos | None:
        with self._lock:
            return self._brazos

    @property
    def estado(self) -> str:
        with self._lock:
            return self._estado

    @property
    def ve_manos(self) -> bool:
        with self._lock:
            return any(v is not None for v in self._curls.values())

    @property
    def ve_cuerpo(self) -> bool:
        with self._lock:
            return self._brazos is not None

    def detener(self, espera: float = 2.0) -> None:
        """Para la captura y ESPERA a que el hilo cierre camara y MediaPipe.

        Sin el join, el interprete puede apagarse mientras MediaPipe sigue
        dentro de su grafo y el proceso muere feo en la salida.
        """
        self._corriendo = False
        if self.is_alive():
            self.join(timeout=espera)

    # -- publicacion ---------------------------------------------------------

    def _publicar(self, curls=None, brazos=..., estado=None):
        with self._lock:
            if curls is not None:
                self._curls = curls
            if brazos is not ...:
                self._brazos = brazos
            if estado is not None:
                self._estado = estado

    # -- lectura de landmarks ------------------------------------------------

    def _leer_cuerpo(self, landmarks) -> LecturaBrazos | None:
        """pose_world_landmarks -> direcciones de brazo en marco del cuerpo.

        Devuelve None si no hay un tronco creible: hombros y caderas son los
        que definen el marco del cuerpo, y si esos no se ven, todo lo demas
        (incluido el yaw) es ruido.
        """
        # MediaPipe da y hacia abajo; lo volteamos para tener y hacia arriba.
        volteo = np.array([1.0, -1.0, 1.0])

        def P(i):
            lm = landmarks[i]
            return np.array([lm.x, lm.y, lm.z]) * volteo

        def visible(i) -> float:
            return float(getattr(landmarks[i], "visibility", 1.0))

        if min(visible(_HOMBRO_I), visible(_HOMBRO_D),
               visible(_CADERA_I), visible(_CADERA_D)) < VIS_MIN_TRONCO:
            return None

        HI, HD = P(_HOMBRO_I), P(_HOMBRO_D)
        CI, CD = P(_CODO_I), P(_CODO_D)
        MI, MD = P(_MUNECA_I), P(_MUNECA_D)
        cadI, cadD = P(_CADERA_I), P(_CADERA_D)

        # Marco del cuerpo: arriba = caderas->hombros, izquierda = hombro
        # derecho->izquierdo (ortogonalizada), adelante = izquierda x arriba.
        arriba = _norm(0.5 * (HI + HD) - 0.5 * (cadI + cadD))
        izquierda = HI - HD
        izquierda = _norm(izquierda - np.dot(izquierda, arriba) * arriba)
        adelante = _norm(np.cross(izquierda, arriba)) * SIGNO_ADELANTE

        def a_cuerpo(v):
            return np.array([np.dot(v, adelante), np.dot(v, izquierda),
                             np.dot(v, arriba)])

        brazo_i, antebrazo_i = _norm(CI - HI), _norm(MI - CI)
        brazo_d, antebrazo_d = _norm(CD - HD), _norm(MD - CD)

        # Confianza por brazo: el peor de los dos segmentos manda, y ademas un
        # brazo que MediaPipe apenas ve (fuera de cuadro, tapado) no cuenta.
        g_humano_i = min(_en_plano(brazo_i), _en_plano(antebrazo_i),
                         visible(_CODO_I), visible(_MUNECA_I))
        g_humano_d = min(_en_plano(brazo_d), _en_plano(antebrazo_d),
                         visible(_CODO_D), visible(_MUNECA_D))

        hum = {
            "izquierda": (a_cuerpo(brazo_i), a_cuerpo(antebrazo_i)),
            "derecha": (a_cuerpo(brazo_d), a_cuerpo(antebrazo_d)),
        }
        gan = {"izquierda": g_humano_i, "derecha": g_humano_d}

        # Giro del torso: cuanto se ha ido de canto la linea de hombros.
        hombros = HI - HD
        yaw = float(np.arctan2(hombros[2], abs(hombros[0]) + 1e-6))

        if self.espejo:
            # Como un espejo: el brazo izquierdo del robot copia tu derecho.
            dirs = {
                "izquierda": tuple(_espejar(v) for v in hum["derecha"]),
                "derecha": tuple(_espejar(v) for v in hum["izquierda"]),
            }
            ganancia = {"izquierda": gan["derecha"], "derecha": gan["izquierda"]}
            yaw = -yaw
        else:
            dirs, ganancia = hum, gan

        return LecturaBrazos(
            dirs=dirs,
            ganancia={k: _rampa(v, CONF_LO, CONF_HI) for k, v in ganancia.items()},
            yaw=yaw,
        )

    def _leer_manos(self, res) -> dict[str, np.ndarray | None]:
        def pts(hlm):
            return np.array([[p.x, p.y, p.z] for p in hlm.landmark], dtype=np.float64)

        humano = {"left": None, "right": None}
        if res.left_hand_landmarks:
            humano["left"] = curls_desde_landmarks(pts(res.left_hand_landmarks))
        if res.right_hand_landmarks:
            humano["right"] = curls_desde_landmarks(pts(res.right_hand_landmarks))

        if self.espejo:
            return {"izquierda": humano["right"], "derecha": humano["left"]}
        return {"izquierda": humano["left"], "derecha": humano["right"]}

    # -- bucle de captura ----------------------------------------------------

    def run(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as e:
            print(f"[CUERPO] Falta mediapipe/opencv ({e}). Camara desactivada.",
                  flush=True)
            self._publicar(estado="sin dependencias")
            return

        cap = cv2.VideoCapture(self.fuente)
        if not cap.isOpened():
            print(f"[CUERPO] No abre la camara: {self.fuente}\n"
                  "         Con DroidCam por USB: adb forward tcp:4747 tcp:4747\n"
                  "         Con webcam local: --camara 0", flush=True)
            self._publicar(estado="camara no disponible")
            return

        mp_holistic = mp.solutions.holistic
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        holistic = mp_holistic.Holistic(model_complexity=1,
                                        refine_face_landmarks=False,
                                        min_detection_confidence=self.confianza,
                                        min_tracking_confidence=self.confianza)
        print(f"[CUERPO] Camara lista ({self.fuente}). Holistic: brazos + torso "
              f"+ dedos. Espejo {'ON' if self.espejo else 'OFF'}.", flush=True)
        self._publicar(estado="sin nadie a la vista")

        vacios = 0
        try:
            while self._corriendo and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    vacios += 1
                    if vacios > 100:
                        self._publicar({"izquierda": None, "derecha": None}, None,
                                       "camara sin imagen")
                        break
                    continue
                vacios = 0

                # Volteamos para verlo como un espejo: asi las etiquetas de
                # MediaPipe corresponden de verdad a tu izquierda y tu derecha.
                frame = cv2.flip(frame, 1)
                res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if res.pose_world_landmarks:
                    brazos = self._leer_cuerpo(res.pose_world_landmarks.landmark)
                else:
                    brazos = None
                curls = self._leer_manos(res)

                manos_vistas = sum(v is not None for v in curls.values())
                if brazos is None and not manos_vistas:
                    estado = "sin nadie a la vista"
                elif brazos is None:
                    estado = "veo tus manos, no tu cuerpo"
                elif manos_vistas:
                    estado = "siguiendo brazos y manos"
                else:
                    estado = "siguiendo tus brazos"
                self._publicar(curls, brazos, estado)

                if self.mostrar:
                    self._dibujar(cv2, mp_draw, mp_holistic, mp_hands, frame, res,
                                  brazos, curls)
        finally:
            cap.release()
            holistic.close()
            if self.mostrar:
                try:
                    cv2.destroyWindow(self.nombre_ventana)
                except Exception:
                    pass
            self._publicar({"izquierda": None, "derecha": None}, None, "camara cerrada")

    # -- ventana de depuracion -----------------------------------------------

    def _dibujar(self, cv2, mp_draw, mp_holistic, mp_hands, frame, res, brazos, curls):
        if res.pose_landmarks:
            mp_draw.draw_landmarks(frame, res.pose_landmarks,
                                   mp_holistic.POSE_CONNECTIONS)
        for hlm in (res.left_hand_landmarks, res.right_hand_landmarks):
            if hlm:
                mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS)

        def texto(y, s, color=(0, 255, 255), escala=0.6):
            cv2.putText(frame, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2)

        texto(28, f"espejo={'ON' if self.espejo else 'OFF'}", (0, 255, 0), 0.7)
        if brazos is None:
            texto(56, "cuerpo: no te veo", (0, 0, 255))
        else:
            texto(56, f"yaw={np.degrees(brazos.yaw):+.0f}  "
                      f"gain izq={brazos.ganancia['izquierda']:.2f} "
                      f"der={brazos.ganancia['derecha']:.2f}")

        def curls_txt(c5):
            return "--" if c5 is None else " ".join(f"{v:.1f}" for v in c5)

        texto(84, f"robot-izq [P I M A m] = {curls_txt(curls['izquierda'])}",
              (255, 200, 0), 0.55)
        texto(108, f"robot-der [P I M A m] = {curls_txt(curls['derecha'])}",
              (255, 200, 0), 0.55)
        cv2.imshow(self.nombre_ventana, frame)
        cv2.waitKey(1)
