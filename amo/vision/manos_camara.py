"""
manos_camara.py — seguimiento de dedos con la camara (MediaPipe Hands).

De donde sale
=============
Es la mitad "manos" de `scripts/teleop/teleop_dedos.py`. Alli se usa Holistic
porque tambien hacen falta los brazos y el torso; aqui solo queremos los dedos
(el resto del cuerpo lo lleva la politica de caminar), asi que usamos
`mp.solutions.hands`, que es mas ligero y ademas etiqueta cada mano como
izquierda o derecha.

El calculo del curl es EXACTAMENTE el de teleop_dedos.py: por cada dedo se
suman los angulos de las dos articulaciones (PIP y DIP) y se normaliza contra
un "puno cerrado". El pulgar tiene su propia escala porque tiene una falange
menos.

    curl 0 = dedo estirado      curl 1 = dedo cerrado

Espejo
======
Con `espejo=True` (por defecto) el robot te copia como si fuera un espejo: la
mano izquierda del robot sigue a tu mano derecha. Es lo natural cuando lo
tienes de frente. Con `espejo=False` te copia lado a lado.

Camara
======
`fuente` acepta un indice de webcam ("0") o una URL. Con DroidCam por USB:

    adb forward tcp:4747 tcp:4747
    fuente = "http://127.0.0.1:4747/video"

Hilos: el bucle de captura corre en su propio hilo. `leer()` devuelve una
copia bajo lock, asi que se puede llamar desde el hilo de la simulacion sin
sincronizar nada mas.
"""

from __future__ import annotations

import threading

import numpy as np

FUENTE_POR_DEFECTO = "http://127.0.0.1:4747/video"

# Suma de angulos (rad) que consideramos "dedo cerrado del todo".
CURL_COMPLETO = 2.6
CURL_COMPLETO_PULGAR = 2.0

# Indices de landmarks de MediaPipe:
# 0 muneca | pulgar 1-4 | indice 5-8 | medio 9-12 | anular 13-16 | menique 17-20
_DEDOS_LARGOS = {
    "indice": (5, 6, 7, 8),
    "medio": (9, 10, 11, 12),
    "anular": (13, 14, 15, 16),
    "menique": (17, 18, 19, 20),
}


def _angulo(u, v) -> float:
    du = np.linalg.norm(u)
    dv = np.linalg.norm(v)
    if du < 1e-9 or dv < 1e-9:
        return 0.0
    c = np.clip(np.dot(u, v) / (du * dv), -1.0, 1.0)
    return float(np.arccos(c))


def curls_desde_landmarks(pts) -> np.ndarray:
    """pts: array (21, 3) de una mano. Devuelve 5 curls en [0, 1]:
    [pulgar, indice, medio, anular, menique]."""
    curls = []
    a, b, c, d = pts[1], pts[2], pts[3], pts[4]
    ang = _angulo(b - a, c - b) + _angulo(c - b, d - c)
    curls.append(np.clip(ang / CURL_COMPLETO_PULGAR, 0.0, 1.0))
    for clave in ("indice", "medio", "anular", "menique"):
        i0, i1, i2, i3 = _DEDOS_LARGOS[clave]
        a, b, c, d = pts[i0], pts[i1], pts[i2], pts[i3]
        ang = _angulo(b - a, c - b) + _angulo(c - b, d - c)
        curls.append(np.clip(ang / CURL_COMPLETO, 0.0, 1.0))
    return np.array(curls, dtype=np.float64)


class SeguidorDeManos(threading.Thread):
    """Lee la camara y publica los curls de cada mano vistos en el frame.

    `leer()` devuelve {"izquierda": curls|None, "derecha": curls|None} ya en
    coordenadas del ROBOT (o sea, con el espejo aplicado). None significa "esa
    mano no se ve"; el que consume decide si mantiene la ultima pose o abre.
    """

    def __init__(self, fuente=FUENTE_POR_DEFECTO, espejo: bool = True,
                 mostrar: bool = False, confianza: float = 0.6,
                 nombre_ventana: str = "Manos R1"):
        super().__init__(name="seguidor-manos", daemon=True)
        self.fuente = int(fuente) if str(fuente).isdigit() else fuente
        self.espejo = espejo
        self.mostrar = mostrar
        self.confianza = confianza
        self.nombre_ventana = nombre_ventana

        self._lock = threading.Lock()
        self._curls: dict[str, np.ndarray | None] = {"izquierda": None, "derecha": None}
        self._estado = "arrancando"
        self._corriendo = True

    # -- API -----------------------------------------------------------------

    def leer(self) -> dict[str, np.ndarray | None]:
        with self._lock:
            return {k: (None if v is None else v.copy()) for k, v in self._curls.items()}

    @property
    def estado(self) -> str:
        with self._lock:
            return self._estado

    @property
    def ve_manos(self) -> bool:
        with self._lock:
            return any(v is not None for v in self._curls.values())

    def detener(self, espera: float = 2.0) -> None:
        """Para la captura y ESPERA a que el hilo cierre camara y MediaPipe.

        Sin el join, el interprete puede apagarse mientras MediaPipe sigue
        dentro de su grafo y el proceso muere feo en la salida.
        """
        self._corriendo = False
        if self.is_alive():
            self.join(timeout=espera)

    # -- bucle de captura ----------------------------------------------------

    def _publicar(self, curls, estado=None):
        with self._lock:
            self._curls = curls
            if estado is not None:
                self._estado = estado

    def run(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as e:
            print(f"[MANOS] Falta mediapipe/opencv ({e}). Camara desactivada.", flush=True)
            self._publicar({"izquierda": None, "derecha": None}, "sin dependencias")
            return

        cap = cv2.VideoCapture(self.fuente)
        if not cap.isOpened():
            print(f"[MANOS] No abre la camara: {self.fuente}\n"
                  "        Con DroidCam por USB: adb forward tcp:4747 tcp:4747\n"
                  "        Con webcam local: --camara 0", flush=True)
            self._publicar({"izquierda": None, "derecha": None}, "camara no disponible")
            return

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands = mp_hands.Hands(model_complexity=0, max_num_hands=2,
                               min_detection_confidence=self.confianza,
                               min_tracking_confidence=self.confianza)
        print(f"[MANOS] Camara lista ({self.fuente}). "
              f"Espejo {'ON' if self.espejo else 'OFF'}.", flush=True)
        self._publicar({"izquierda": None, "derecha": None}, "sin manos a la vista")

        vacios = 0
        try:
            while self._corriendo and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    vacios += 1
                    if vacios > 100:
                        self._publicar({"izquierda": None, "derecha": None},
                                       "camara sin imagen")
                        break
                    continue
                vacios = 0

                # Volteamos el frame para verlo como un espejo: asi la etiqueta
                # de MediaPipe ("Left"/"Right") corresponde de verdad a tu mano.
                frame = cv2.flip(frame, 1)
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                humano = {"left": None, "right": None}
                if res.multi_hand_landmarks and res.multi_handedness:
                    for lm, etiqueta in zip(res.multi_hand_landmarks, res.multi_handedness):
                        cual = etiqueta.classification[0].label.lower()  # 'left'/'right'
                        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark],
                                       dtype=np.float64)
                        humano[cual] = curls_desde_landmarks(pts)

                # Tu mano -> mano del robot
                if self.espejo:
                    curls = {"izquierda": humano["right"], "derecha": humano["left"]}
                else:
                    curls = {"izquierda": humano["left"], "derecha": humano["right"]}

                vistas = sum(v is not None for v in curls.values())
                self._publicar(curls,
                               "siguiendo tus manos" if vistas else "sin manos a la vista")

                if self.mostrar:
                    if res.multi_hand_landmarks:
                        for lm in res.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                    def txt(c5):
                        return "--" if c5 is None else " ".join(f"{v:.1f}" for v in c5)

                    cv2.putText(frame, f"espejo={'ON' if self.espejo else 'OFF'}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"robot-izq [P I M A m] = {txt(curls['izquierda'])}",
                                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
                    cv2.putText(frame, f"robot-der [P I M A m] = {txt(curls['derecha'])}",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)
                    cv2.imshow(self.nombre_ventana, frame)
                    cv2.waitKey(1)
        finally:
            cap.release()
            hands.close()
            if self.mostrar:
                try:
                    cv2.destroyWindow(self.nombre_ventana)
                except Exception:
                    pass
            self._publicar({"izquierda": None, "derecha": None}, "camara cerrada")
