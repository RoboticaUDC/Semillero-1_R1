"""
arm_ik.py — cinematica inversa de los brazos del R1 y control de brazos por
camara.

De donde sale
=============
La clase `ArmIK` de `scripts/teleop/teleop_dedos.py`, extraida para que la
puedan usar los scripts que no son de teleoperacion (play_r1_voz.py). Mismo
metodo (Gauss-Newton amortiguado sobre 4 joints por brazo) y mismas
constantes, mas la capa de control que antes vivia en `update_arms()`.

Que resuelve
============
No perseguimos una posicion de la muneca, sino DOS DIRECCIONES por brazo:

    u = hombro -> codo     (a donde apunta el brazo)
    w = codo   -> muneca   (a donde apunta el antebrazo)

Es lo que la camara puede medir con dignidad: las longitudes del humano no
son las del robot, pero los angulos si se copian. El residual pesa mas la
direccion del brazo (`W_BRAZO`) porque un error ahi arrastra todo el
antebrazo.

Reparto del trabajo entre hilos
===============================
La IK cuesta ~30 `mj_forward` por brazo y llamada; a 500 Hz no cabe en el
bucle de simulacion. Por eso `ControladorBrazosCamara` es un HILO que resuelve
a 30 Hz contra su PROPIA copia del modelo (nunca toca la `data` de la
simulacion) y publica el resultado bajo lock. El hilo de la simulacion solo
llama a `avanzar()` a 50 Hz, que es una interpolacion barata.

Uso tipico
==========
    brazos = ControladorBrazosCamara(seguidor, xml_path=scene("r1_manos"))
    brazos.start()
    brazos.encender()
    # a la frecuencia de control (50 Hz), en el hilo de la simulacion:
    brazos.avanzar()
    objetivo[14:24] = brazos.mezclar(objetivo[14:24])
"""

from __future__ import annotations

import threading
import time

import numpy as np
import mujoco

from amo.paths import scene

# =============================================================================
# JOINTS
# =============================================================================
# Los 4 joints por brazo que resuelve la IK. El wrist_roll no entra: la camara
# no da su giro de forma fiable y dejarlo suelto empeora la solucion.

JOINTS_BRAZO = {
    "izquierda": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                  "left_shoulder_yaw_joint", "left_elbow_joint"],
    "derecha": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint", "right_elbow_joint"],
}

# Cuerpos cuya posicion define los dos segmentos: hombro, codo, muneca.
CUERPOS_BRAZO = {
    "izquierda": ["left_shoulder_pitch_link", "left_elbow_link", "left_wrist_roll_link"],
    "derecha": ["right_shoulder_pitch_link", "right_elbow_link", "right_wrist_roll_link"],
}

LADOS = ("izquierda", "derecha")

# Pose de reposo de los 4 joints, por lado. Sirve de semilla y de sitio al que
# volver cuando la camara deja de ver a nadie.
Q_REPOSO = {"izquierda": np.array([0.18, 0.18, 0.0, 1.5]),
            "derecha": np.array([0.18, -0.18, 0.0, 1.5])}

# Los 10 valores de brazo en orden MuJoCo:
# [izq: pitch, roll, yaw, codo, wrist_roll | der: idem]
POSE_REPOSO_10 = np.array([0.18, 0.18, 0.0, 1.5, 0.0,
                           0.18, -0.18, 0.0, 1.5, 0.0], dtype=np.float64)
_RANURA = {"izquierda": slice(0, 4), "derecha": slice(5, 9)}

# =============================================================================
# PARAMETROS
# =============================================================================
ITERS = 6            # iteraciones de Gauss-Newton por llamada
AMORT = 3e-3         # amortiguacion (Levenberg): sube si oscila
EPS = 1e-4           # paso del jacobiano por diferencias finitas
W_BRAZO = 2.0        # peso del segmento hombro-codo en el residual

# El residual NUNCA baja a cero, y no es culpa de la IK: el brazo del R1
# cuelga abierto ~22 grados (con shoulder_roll a 0 el hombro-codo apunta a
# [0.07, 0.38, -0.92]) y el roll solo puede meterse 0.23 rad, asi que una pose
# humana con el brazo pegado al cuerpo deja un residual de ~0.5 aunque el
# seguimiento sea perfecto. Por eso NO hay umbral absoluto de aceptacion:
# comparamos contra el residual de la pose en la que ya estabamos (si la IK
# nos acerca a lo que pide la camara, vale) y contra un "suelo" que se estima
# solo con los ultimos resultados buenos.
ERR_BUENO = 0.18       # semilla del suelo mientras no hay historia
MARGEN_REINTENTO = 0.15  # cuanto puede empeorar del suelo antes de reintentar
TOLERANCIA = 1e-3      # cuanto puede empeorar respecto a la pose previa
EMA_SUELO = 0.05       # que tan rapido se adapta el suelo estimado

EMA = 0.18           # suavizado del objetivo (mas alto = mas responsivo)
PASO_MAX = 0.04      # rad como mucho por llamada a avanzar()
MEZCLA_PASO = 0.04   # ~0.5 s de transicion al encender/apagar el seguimiento

# Cintura: el giro del torso que copiamos del humano. Conservador a proposito,
# que la politica de caminar tiene que seguir equilibrando encima.
YAW_ZONA_MUERTA = 0.10
YAW_CLIP = 0.35
YAW_EMA = 0.15

# Si la camara deja de ver el cuerpo mas de esto, los brazos vuelven a reposo
# en vez de quedarse congelados con las manos arriba.
SEGUNDOS_SIN_CUERPO = 2.0


class IKBrazos:
    """Resuelve los 4 joints de un brazo a partir de dos direcciones.

    Trabaja contra su propia copia del modelo: se le puede llamar desde
    cualquier hilo sin tocar la simulacion de verdad.
    """

    def __init__(self, xml_path: str | None = None):
        # Por defecto r1.xml aunque la simulacion lleve manos: la cinematica de
        # los brazos es identica en los dos XML (verificado joint a joint) y
        # r1.xml tiene 22 qpos menos, asi que cada mj_forward de la IK sale mas
        # barato. Son ~30 por brazo y llamada, y eso se nota.
        self.model = mujoco.MjModel.from_xml_path(xml_path or scene("r1"))
        self.data = mujoco.MjData(self.model)
        self.qadr, self.limites, self.cuerpos = {}, {}, {}
        for lado in LADOS:
            adr, lim = [], []
            for nombre in JOINTS_BRAZO[lado]:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre)
                if jid < 0:
                    raise RuntimeError(f"el modelo no tiene el joint '{nombre}'")
                adr.append(int(self.model.jnt_qposadr[jid]))
                lim.append(self.model.jnt_range[jid].copy())
            self.qadr[lado] = np.array(adr)
            self.limites[lado] = np.array(lim)
            self.cuerpos[lado] = []
            for nombre in CUERPOS_BRAZO[lado]:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nombre)
                if bid < 0:
                    raise RuntimeError(f"el modelo no tiene el cuerpo '{nombre}'")
                self.cuerpos[lado].append(bid)

    # -- cinematica directa ---------------------------------------------------

    def _fk(self, lado: str, q) -> tuple[np.ndarray, np.ndarray]:
        """q (4) -> direcciones (brazo, antebrazo) en marco del cuerpo."""
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0          # cuaternion identidad de la base
        self.data.qpos[self.qadr[lado]] = q
        mujoco.mj_forward(self.model, self.data)
        hombro, codo, muneca = (self.data.xpos[b] for b in self.cuerpos[lado])
        return _norm(codo - hombro), _norm(muneca - codo)

    def _residual(self, lado, q, u_obj, w_obj) -> np.ndarray:
        u, w = self._fk(lado, q)
        return np.concatenate([W_BRAZO * (u - u_obj), w - w_obj])

    def _gauss_newton(self, lado, u_obj, w_obj, q0):
        lo, hi = self.limites[lado][:, 0], self.limites[lado][:, 1]
        q = np.clip(np.asarray(q0, dtype=np.float64).copy(), lo, hi)
        for _ in range(ITERS):
            r = self._residual(lado, q, u_obj, w_obj)
            if np.linalg.norm(r) < 1e-3:
                break
            J = np.zeros((6, 4))
            for i in range(4):
                dq = q.copy()
                dq[i] += EPS
                J[:, i] = (self._residual(lado, dq, u_obj, w_obj) - r) / EPS
            H = J.T @ J + AMORT * np.eye(4)
            siguiente = np.clip(q + np.linalg.solve(H, -J.T @ r), lo, hi)
            # Parada por paso pequeno, no solo por residual pequeno: contra un
            # objetivo inalcanzable (lo normal, ver la nota de arriba) el
            # residual se estanca en su suelo y sin esto gastariamos las 6
            # iteraciones enteras en cada frame.
            paso = float(np.max(np.abs(siguiente - q)))
            q = siguiente
            if paso < 1e-3:
                break
        return q, float(np.linalg.norm(self._residual(lado, q, u_obj, w_obj)))

    def error(self, lado: str, q, u_obj, w_obj) -> float:
        """Cuanto se aleja una pose concreta de lo que pide la camara."""
        return float(np.linalg.norm(self._residual(lado, q, u_obj, w_obj)))

    def resolver(self, lado: str, u_obj, w_obj, q_previo, umbral: float = ERR_BUENO):
        """Devuelve (q4, error). Reintenta desde otras semillas si sale mal.

        Gauss-Newton es local: desde la pose anterior converge rapido mientras
        el movimiento sea continuo, pero se atasca en minimos locales cuando el
        brazo cruza (por ejemplo, al levantarlo por encima del hombro). Las
        semillas de repuesto cubren esos cambios de rama.

        `umbral` es "esto ya esta bastante bien, no reintentes". Se pasa desde
        fuera porque el residual que se puede alcanzar depende de la pose (ver
        la nota de las constantes): con un numero fijo, o reintentabamos las 4
        semillas en cada frame (5 veces mas lento) o no reintentabamos nunca.
        """
        q, err = self._gauss_newton(lado, u_obj, w_obj, q_previo)
        if err < umbral:
            return q, err
        s = 1.0 if lado == "izquierda" else -1.0
        for semilla in (np.array([0.0, 0.05 * s, 0.0, 0.15]),
                        np.array([-1.2, 0.10 * s, 0.0, 0.60]),
                        np.array([0.0, 1.20 * s, 0.0, 0.40]),
                        Q_REPOSO[lado]):
            qi, ei = self._gauss_newton(lado, u_obj, w_obj, semilla)
            if ei < err:
                q, err = qi, ei
            if err < umbral:
                break
        return q, err


class ControladorBrazosCamara(threading.Thread):
    """Brazos (y cintura) del robot copiando los del humano que ve la camara.

    El hilo resuelve la IK a `hz`; el hilo de la simulacion llama a `avanzar()`
    y luego a `mezclar()`. La mezcla sube y baja sola: al encender entra en
    ~0.5 s y al apagar devuelve los brazos a la politica igual de suave, para
    que nunca peguen un tiron.
    """

    def __init__(self, seguidor, xml_path: str | None = None, hz: float = 30.0,
                 seguir_cintura: bool = True, activo: bool = False):
        super().__init__(name="brazos-camara", daemon=True)
        self.seguidor = seguidor
        self.seguir_cintura = seguir_cintura
        self.periodo = 1.0 / float(hz)

        self.ik = IKBrazos(xml_path)
        self.activo = bool(activo)

        self._lock = threading.Lock()
        self._q_ik = {lado: Q_REPOSO[lado].copy() for lado in LADOS}
        self._deseado = POSE_REPOSO_10.copy()   # publicado por el hilo de IK
        self._yaw_deseado = 0.0
        self._error = {lado: 0.0 for lado in LADOS}
        self._suelo: dict[str, float | None] = {lado: None for lado in LADOS}
        self._visto = 0.0                        # ultima vez que se vio cuerpo
        self._corriendo = True

        # Solo los toca el hilo de la simulacion: sin lock.
        self.objetivo = POSE_REPOSO_10.copy()
        self.yaw = 0.0
        self.mezcla = 0.0

    # -- ordenes -------------------------------------------------------------

    def encender(self) -> None:
        self.activo = True

    def apagar(self) -> None:
        self.activo = False

    def detener(self, espera: float = 1.0) -> None:
        """Para el hilo y ESPERA a que salga.

        El join no es cortesia: si el interprete se apaga mientras este hilo
        esta dentro de `mj_forward` sobre su propio modelo, el proceso muere
        con un segfault en la salida.
        """
        self._corriendo = False
        if self.is_alive():
            self.join(timeout=espera)

    # -- hilo de IK ----------------------------------------------------------

    def run(self):
        fallos = 0
        while self._corriendo:
            t0 = time.monotonic()
            if self.activo:
                try:
                    self._resolver_una_vez()
                except Exception as e:
                    # Si este hilo muere, los brazos se quedan congelados y
                    # nadie se entera: mejor avisar una vez y seguir. La mezcla
                    # baja sola a los 2 s sin lecturas, asi que el robot vuelve
                    # a la politica en vez de quedarse en una pose rara.
                    fallos += 1
                    if fallos == 1:
                        print(f"[BRAZOS] Fallo en la IK ({type(e).__name__}: {e}). "
                              "Sigo intentando.", flush=True)
            resto = self.periodo - (time.monotonic() - t0)
            time.sleep(max(resto, 0.001))

    def _resolver_una_vez(self):
        lectura = self.seguidor.leer_brazos() if self.seguidor is not None else None
        ahora = time.monotonic()

        if lectura is None:
            # Sin cuerpo a la vista mantenemos la ultima pose un rato (los
            # huecos de deteccion duran decimas de segundo); si la ausencia se
            # alarga, volvemos a reposo.
            with self._lock:
                perdido = ahora - self._visto
                if perdido > SEGUNDOS_SIN_CUERPO:
                    self._q_ik = {lado: Q_REPOSO[lado].copy() for lado in LADOS}
                    self._deseado = POSE_REPOSO_10.copy()
                    self._yaw_deseado = 0.0
            return

        q_previo = {lado: self._q_ik[lado].copy() for lado in LADOS}
        nuevo, error = {}, {}
        for lado in LADOS:
            u, w = lectura.dirs[lado]
            suelo = self._suelo[lado]
            umbral = ERR_BUENO if suelo is None else suelo + MARGEN_REINTENTO
            err_previo = self.ik.error(lado, q_previo[lado], u, w)
            q, err = self.ik.resolver(lado, u, w, q_previo[lado], umbral=umbral)

            g = float(lectura.ganancia[lado])
            # Dos puertas. La ganancia: un brazo apuntando a la camara no tiene
            # direccion fiable, asi que se queda donde estaba. Y la mejora: si
            # la IK no nos acerca al menos tanto como la pose en la que ya
            # estamos, es que ha divergido; mejor no moverse.
            if err <= err_previo + TOLERANCIA:
                nuevo[lado] = g * q + (1.0 - g) * q_previo[lado]
                self._suelo[lado] = (err if suelo is None
                                     else (1 - EMA_SUELO) * suelo + EMA_SUELO * err)
            else:
                nuevo[lado] = q_previo[lado]
            error[lado] = err

        deseado = POSE_REPOSO_10.copy()
        for lado in LADOS:
            deseado[_RANURA[lado]] = nuevo[lado]

        yaw = lectura.yaw if self.seguir_cintura else 0.0
        if abs(yaw) < YAW_ZONA_MUERTA:
            yaw = 0.0
        else:
            yaw -= np.sign(yaw) * YAW_ZONA_MUERTA
        yaw = float(np.clip(yaw, -YAW_CLIP, YAW_CLIP))

        with self._lock:
            self._q_ik = nuevo
            self._deseado = deseado
            self._yaw_deseado = (1 - YAW_EMA) * self._yaw_deseado + YAW_EMA * yaw
            self._error = error
            self._visto = ahora

    # -- hilo de la simulacion -----------------------------------------------

    def avanzar(self) -> None:
        """Acerca el objetivo al ultimo resultado de la IK. Llamar a 50 Hz."""
        with self._lock:
            deseado = self._deseado.copy()
            yaw_deseado = self._yaw_deseado
            hay_cuerpo = (time.monotonic() - self._visto) < SEGUNDOS_SIN_CUERPO

        if self.activo and hay_cuerpo:
            self.mezcla = min(1.0, self.mezcla + MEZCLA_PASO)
        else:
            self.mezcla = max(0.0, self.mezcla - MEZCLA_PASO)

        suave = (1 - EMA) * self.objetivo + EMA * deseado
        delta = np.clip(suave - self.objetivo, -PASO_MAX, PASO_MAX)
        self.objetivo = self.objetivo + delta
        self.yaw = (1 - EMA) * self.yaw + EMA * yaw_deseado

    def mezclar(self, brazos_politica) -> np.ndarray:
        """Fusiona los 10 valores de la politica con los de la camara."""
        if self.mezcla <= 0.0:
            return np.asarray(brazos_politica, dtype=np.float64)
        m = self.mezcla
        return (1.0 - m) * np.asarray(brazos_politica, dtype=np.float64) + m * self.objetivo

    def mezclar_cintura(self, yaw_politica: float) -> float:
        if self.mezcla <= 0.0 or not self.seguir_cintura:
            return float(yaw_politica)
        m = self.mezcla
        return float((1.0 - m) * yaw_politica + m * self.yaw)

    def reiniciar(self) -> None:
        """Vuelve a reposo sin transitorio (para el reset del robot)."""
        with self._lock:
            self._q_ik = {lado: Q_REPOSO[lado].copy() for lado in LADOS}
            self._deseado = POSE_REPOSO_10.copy()
            self._yaw_deseado = 0.0
        self.objetivo = POSE_REPOSO_10.copy()
        self.yaw = 0.0
        self.mezcla = 0.0

    # -- lectura -------------------------------------------------------------

    def resumen(self) -> dict:
        with self._lock:
            error = dict(self._error)
            hay_cuerpo = (time.monotonic() - self._visto) < SEGUNDOS_SIN_CUERPO
        grados = np.degrees(self.objetivo)
        return {
            "siguiendo": bool(self.activo),
            "cuerpo_a_la_vista": bool(hay_cuerpo),
            "mezcla": round(float(self.mezcla), 2),
            "error_ik": {lado: round(float(e), 3) for lado, e in error.items()},
            "yaw_cintura": round(float(self.yaw), 3),
            # hombro (pitch, roll, yaw) y codo, en grados: es lo que hay que
            # mirar cuando el robot "no copia bien" y quieres ver que le llega.
            "brazos_grados": {
                lado: [round(float(g), 1) for g in grados[_RANURA[lado]]]
                for lado in LADOS
            },
        }


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v
