"""
hand_controller.py — control de los dedos de las manos Revo2 del R1.

De donde sale
=============
La logica de dedos vivia dentro de `scripts/teleop/teleop_dedos.py` (curl por
dedo -> rango real del joint, suavizado y PD). Aqui esta extraida a un modulo
para que la puedan usar tambien los scripts que NO son de teleoperacion
(play_r1_voz.py, por ejemplo), sin arrastrar la camara ni la IK de brazos.

Modelo de mano
==============
Cada mano Revo2 son 11 joints:

    [pulgar_meta, pulgar_prox, pulgar_dist,
     indice_prox, indice_dist, medio_prox, medio_dist,
     anular_prox, anular_dist, menique_prox, menique_dist]

Pero por dedo solo mandamos UN numero, el "curl" (cierre) en [0, 1]:

    curl 0 -> dedo estirado -> joint en su minimo (jnt_range[0])
    curl 1 -> dedo cerrado  -> joint en su maximo (jnt_range[1])

Los limites se leen del XML, nunca se escriben a mano: si cambias el modelo,
esto sigue funcionando. Los 5 curls (pulgar, indice, medio, anular, menique)
se expanden a los 11 joints; el pulgar alimenta tambien su metacarpo, que es
el que lo cierra a traves de la palma.

Todo se resuelve POR NOMBRE (jnt_qposadr / jnt_dofadr / actuador). Con las
manos puestas, el XML mete los 11 joints de cada mano justo detras de su
muneca, asi que los indices fijos del modelo sin manos ya no valen.

Uso tipico
==========
    manos = ControladorManos(model, data)
    manos.poner_pose("derecha", "puno")
    # a la frecuencia de control (50 Hz):
    manos.avanzar()
    # en cada paso de simulacion (500 Hz):
    manos.aplicar_torques()

Hilos: esta clase NO lleva locks. Llamala siempre desde el hilo de la
simulacion; si la orden viene de otro hilo (consola, voz, camara), pasala
antes por una cola.
"""

from __future__ import annotations

import numpy as np
import mujoco

# =============================================================================
# NOMBRES
# =============================================================================

LADOS = ("izquierda", "derecha")

DEDOS = ("pulgar", "indice", "medio", "anular", "menique")

# 11 joints por mano, en el mismo orden en ambos lados.
JOINTS_MANO = {
    "izquierda": [
        "left_thumb_metacarpal_joint", "left_thumb_proximal_joint", "left_thumb_distal_joint",
        "left_index_proximal_joint", "left_index_distal_joint",
        "left_middle_proximal_joint", "left_middle_distal_joint",
        "left_ring_proximal_joint", "left_ring_distal_joint",
        "left_pinky_proximal_joint", "left_pinky_distal_joint",
    ],
    "derecha": [
        "right_thumb_metacarpal_joint", "right_thumb_proximal_joint", "right_thumb_distal_joint",
        "right_index_proximal_joint", "right_index_distal_joint",
        "right_middle_proximal_joint", "right_middle_distal_joint",
        "right_ring_proximal_joint", "right_ring_distal_joint",
        "right_pinky_proximal_joint", "right_pinky_distal_joint",
    ],
}

JOINTS_TODOS = JOINTS_MANO["izquierda"] + JOINTS_MANO["derecha"]

# =============================================================================
# POSES
# =============================================================================
# Catalogo cerrado de poses, en curls [pulgar, indice, medio, anular, menique].
# Igual criterio que los gestos de brazo: el que manda elige de una lista
# verificada, no inventa angulos.

POSES: dict[str, np.ndarray] = {
    "abierta":       np.array([0.00, 0.00, 0.00, 0.00, 0.00]),
    "relajada":      np.array([0.20, 0.18, 0.18, 0.18, 0.18]),
    "puno":          np.array([1.00, 1.00, 1.00, 1.00, 1.00]),
    "garra":         np.array([0.45, 0.55, 0.55, 0.55, 0.55]),
    "pinza":         np.array([0.90, 0.85, 0.90, 0.95, 1.00]),
    "ok":            np.array([0.90, 0.85, 0.05, 0.05, 0.05]),
    "senalar":       np.array([0.85, 0.00, 1.00, 1.00, 1.00]),
    "pulgar_arriba": np.array([0.00, 1.00, 1.00, 1.00, 1.00]),
    "paz":           np.array([1.00, 0.00, 0.00, 1.00, 1.00]),
}

DESCRIPCION_POSES = (
    "abierta = mano completamente extendida; "
    "relajada = pose de reposo, dedos apenas curvados; "
    "puno = mano cerrada del todo; "
    "garra = dedos a medio cerrar, como para envolver algo; "
    "pinza = pulgar e indice juntos, resto cerrado; "
    "ok = circulo con pulgar e indice, los otros tres extendidos; "
    "senalar = solo el indice extendido; "
    "pulgar_arriba = solo el pulgar extendido; "
    "paz = indice y medio extendidos en V"
)

POSE_REPOSO = "relajada"


def expandir_curls(c5) -> np.ndarray:
    """5 curls (pulgar..menique) -> 11 factores, uno por joint de la mano."""
    c = np.asarray(c5, dtype=np.float64)
    return np.array([c[0], c[0], c[0],
                     c[1], c[1],
                     c[2], c[2],
                     c[3], c[3],
                     c[4], c[4]], dtype=np.float64)


def normalizar_lado(lado: str) -> str:
    """Acepta 'izq', 'left', 'derecha', 'ambas'... y devuelve el nombre canonico."""
    t = str(lado).strip().lower()
    if t in ("izquierda", "izq", "izquierdo", "left", "l"):
        return "izquierda"
    if t in ("derecha", "der", "derecho", "right", "r"):
        return "derecha"
    if t in ("ambas", "ambos", "las dos", "both", "todas"):
        return "ambas"
    raise ValueError(f"lado desconocido: '{lado}'")


# =============================================================================
# CONTROLADOR
# =============================================================================


class ControladorManos:
    """Los 22 joints de dedos: objetivo suavizado + PD contra el modelo.

    Si el XML cargado no tiene manos (r1.xml en vez de r1_manos.xml), la clase
    se construye igual pero `disponible` queda en False y todos los metodos
    son no-ops. Asi el script que la usa no necesita ramas por todas partes.
    """

    KP = 4.0            # PD de dedos; los actuadores son motores de torque
    KD = 0.15
    EMA = 0.35          # suavizado del objetivo (mas alto = mas responsivo)
    PASO_MAX = 0.20     # rad como mucho por llamada a avanzar()

    def __init__(self, model, data, pose_inicial: str = POSE_REPOSO):
        self.model = model
        self.data = data
        self.disponible = False
        self.motivo = ""

        try:
            self._resolver()
        except Exception as e:               # XML sin manos, o con otros nombres
            self.motivo = str(e)
            self.pose_nombre = {lado: "sin manos" for lado in LADOS}
            return

        self.rango = {"izquierda": slice(0, 11), "derecha": slice(11, 22)}
        self.curls = {lado: np.zeros(5) for lado in LADOS}
        self.pose_nombre = {lado: pose_inicial for lado in LADOS}
        self.objetivo = self.lo.copy()       # lo que se manda al PD
        self._deseado = self.lo.copy()       # a donde queremos llegar
        self.disponible = True
        self.poner_pose("ambas", pose_inicial)
        self.objetivo = self._deseado.copy()  # arranque sin transitorio

    # -- resolucion por nombre ----------------------------------------------

    def _resolver(self):
        joint_a_act = {}
        for aid in range(self.model.nu):
            jid = self.model.actuator_trnid[aid, 0]
            jn = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jn is not None:
                joint_a_act[jn] = aid

        qadr, vadr, act, lo, hi, clo, chi = [], [], [], [], [], [], []
        for nombre in JOINTS_TODOS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nombre)
            if jid < 0:
                raise RuntimeError(f"el modelo no tiene el joint '{nombre}'")
            if nombre not in joint_a_act:
                raise RuntimeError(f"el joint '{nombre}' no tiene actuador")
            aid = joint_a_act[nombre]
            qadr.append(int(self.model.jnt_qposadr[jid]))
            vadr.append(int(self.model.jnt_dofadr[jid]))
            act.append(int(aid))
            r = self.model.jnt_range[jid]
            lo.append(float(r[0]))
            hi.append(float(r[1]))
            cr = self.model.actuator_ctrlrange[aid]
            if self.model.actuator_ctrllimited[aid]:
                clo.append(float(cr[0]))
                chi.append(float(cr[1]))
            else:
                clo.append(-2.0)
                chi.append(2.0)

        self.qadr = np.array(qadr)
        self.vadr = np.array(vadr)
        self.act = np.array(act)
        self.lo = np.array(lo)
        self.hi = np.array(hi)
        self.clo = np.array(clo)
        self.chi = np.array(chi)

    # -- ordenes -------------------------------------------------------------

    def lados(self, lado: str) -> tuple[str, ...]:
        """'ambas' -> ('izquierda', 'derecha'); un lado -> ese lado."""
        lado = normalizar_lado(lado)
        return LADOS if lado == "ambas" else (lado,)

    def poner_curls(self, lado: str, c5, nombre: str = "manual") -> None:
        """Fija el cierre de cada dedo, en [0, 1]."""
        if not self.disponible:
            return
        c5 = np.clip(np.asarray(c5, dtype=np.float64), 0.0, 1.0)
        for l in self.lados(lado):
            self.curls[l] = c5.copy()
            self.pose_nombre[l] = nombre
            self._deseado[self.rango[l]] = self._curls_a_qpos(l, c5)

    def poner_pose(self, lado: str, nombre: str, intensidad: float = 1.0) -> None:
        """Aplica una pose del catalogo. `intensidad` la escala (0 = abierta)."""
        if not self.disponible:
            return
        if nombre not in POSES:
            raise ValueError(f"pose desconocida: '{nombre}'")
        c5 = POSES[nombre] * float(np.clip(intensidad, 0.0, 1.0))
        self.poner_curls(lado, c5, nombre=nombre)

    def poner_reposo(self) -> None:
        self.poner_pose("ambas", POSE_REPOSO)

    def _curls_a_qpos(self, lado: str, c5) -> np.ndarray:
        r = self.rango[lado]
        lo, hi = self.lo[r], self.hi[r]
        return lo + (hi - lo) * expandir_curls(c5)

    # -- bucle ---------------------------------------------------------------

    def avanzar(self) -> None:
        """Acerca el objetivo al deseado. Llamar a la frecuencia de control."""
        if not self.disponible:
            return
        suave = (1 - self.EMA) * self.objetivo + self.EMA * self._deseado
        delta = np.clip(suave - self.objetivo, -self.PASO_MAX, self.PASO_MAX)
        self.objetivo = np.clip(self.objetivo + delta, self.lo, self.hi)

    def aplicar_torques(self) -> None:
        """PD de dedos. Llamar en cada paso de simulacion (modo dinamico)."""
        if not self.disponible:
            return
        q = self.data.qpos[self.qadr]
        qd = self.data.qvel[self.vadr]
        tau = self.KP * (self.objetivo - q) - self.KD * qd
        self.data.ctrl[self.act] = np.clip(tau, self.clo, self.chi)

    def escribir_qpos(self) -> None:
        """Mete el objetivo directamente en qpos (modo cinematico)."""
        if not self.disponible:
            return
        self.data.qpos[self.qadr] = self.objetivo

    def reiniciar(self, pose: str = POSE_REPOSO) -> None:
        """Manos a la pose de reposo, sin transitorio, y qpos escrito."""
        if not self.disponible:
            return
        self.poner_pose("ambas", pose)
        self.objetivo = self._deseado.copy()
        self.data.qpos[self.qadr] = self.objetivo
        self.data.qvel[self.vadr] = 0.0

    # -- lectura -------------------------------------------------------------

    def cierre(self, lado: str) -> float:
        """Cuanto esta cerrada la mano ahora mismo, en [0, 1], medido de qpos."""
        if not self.disponible:
            return 0.0
        lado = normalizar_lado(lado)
        r = self.rango[lado]
        q = self.data.qpos[self.qadr[r]]
        frac = (q - self.lo[r]) / np.maximum(self.hi[r] - self.lo[r], 1e-6)
        return float(np.clip(frac.mean(), 0.0, 1.0))

    def resumen(self) -> dict:
        """Estado de las manos para la telemetria."""
        if not self.disponible:
            return {"disponible": False, "motivo": self.motivo}
        return {
            "disponible": True,
            "izquierda": {"pose": self.pose_nombre["izquierda"],
                          "cierre": round(self.cierre("izquierda"), 2)},
            "derecha": {"pose": self.pose_nombre["derecha"],
                        "cierre": round(self.cierre("derecha"), 2)},
        }
