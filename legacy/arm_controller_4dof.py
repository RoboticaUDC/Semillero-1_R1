"""
arm_controller.py
=================
Módulo de control de brazos para el robot G1 en MuJoCo.
Integra secuencias automáticas animadas y control manual mejorado.

CÓMO INTEGRAR EN banda.py:
  1. Copiar este archivo a la carpeta AMO/
  2. En banda.py agregar al inicio:
       from arm_controller import ArmController, ArmSequence
  3. En HumanoidEnv.__init__(), después de inicializar el viewer:
       self.arm_ctrl = ArmController(self.model, self.data)
  4. En el loop de run(), reemplazar:
       self.update_robot_arm()
     por:
       self.arm_ctrl.update(self.viewer_state.arm_target)
       self.arm_ctrl.apply(self.data)
  5. En create_key_callback(), agregar las teclas F1-F6:
       Ver sección TECLAS más abajo.

TECLAS:
  F1  = Saludar (agitar mano derecha)
  F2  = Apuntar hacia adelante (brazo derecho)
  F3  = Pose de carga (ambos brazos extendidos)
  F4  = Brazos en cruz
  F5  = Pose de guardia / defensa
  F6  = Volver a pose neutral
  F7  = Pausar / reanudar secuencia activa

ÍNDICES DE JOINTS DEL BRAZO G1 (dentro de qpos[7:30]):
  15 = left_shoulder_pitch
  16 = left_shoulder_roll
  17 = left_shoulder_yaw
  18 = left_elbow
  19 = right_shoulder_pitch
  20 = right_shoulder_roll
  21 = right_shoulder_yaw
  22 = right_elbow
  (total 8 joints, índices 15-22 dentro del array de 23 DOFs)
"""

import numpy as np
import math
import time
from enum import Enum, auto


# =============================================================================
# POSES PREDEFINIDAS
# Formato: [lsp, lsr, lsy, le, rsp, rsr, rsy, re]
#   l = left, r = right
#   sp = shoulder pitch, sr = shoulder roll, sy = shoulder yaw, e = elbow
# =============================================================================

class Pose:
    """Poses estáticas de referencia para los 8 joints de brazos."""

    NEUTRAL = np.array([
        0.20,  0.20,  0.00,  1.28,   # brazo izquierdo
        0.20, -0.20,  0.00,  1.28,   # brazo derecho
    ], dtype=np.float32)

    # Brazo derecho levantado, listo para saludar
    WAVE_READY = np.array([
        0.20,  0.20,  0.00,  1.28,   # izq: neutral
        0.35, -1.56, -1.26, -0.40,   # der: levantado
    ], dtype=np.float32)

    # Brazo derecho agitando (codo flexionado)
    WAVE_UP = np.array([
        0.20,  0.20,  0.00,  1.28,   # izq: neutral
        0.35, -1.56, -1.26,  1.07,   # der: codo arriba
    ], dtype=np.float32)

    # Apuntar con brazo derecho hacia adelante
    POINT_RIGHT = np.array([
        0.20,  0.20,  0.00,  1.28,   # izq: neutral
       -0.10, -0.20,  0.00,  0.05,   # der: extendido al frente
    ], dtype=np.float32)

    # Apuntar con ambos brazos hacia adelante (pose de carga)
    CARRY = np.array([
       -0.10,  0.20,  0.00,  0.30,   # izq: extendido frente
       -0.10, -0.20,  0.00,  0.30,   # der: extendido frente
    ], dtype=np.float32)

    # Brazos en cruz (horizontal)
    CROSS = np.array([
        0.00,  1.57,  0.00,  0.10,   # izq: horizontal
        0.00, -1.57,  0.00,  0.10,   # der: horizontal
    ], dtype=np.float32)

    # Pose de guardia (boxeo ligero)
    GUARD = np.array([
        0.60,  0.50,  0.00,  1.20,   # izq: guardia alta
        0.60, -0.50,  0.00,  1.20,   # der: guardia alta
    ], dtype=np.float32)

    # Brazos hacia abajo, relajados
    RELAXED = np.array([
        0.60,  0.10,  0.00,  1.50,   # izq: caídos
        0.60, -0.10,  0.00,  1.50,   # der: caídos
    ], dtype=np.float32)


# =============================================================================
# SECUENCIAS DE MOVIMIENTO
# =============================================================================

class SequenceState(Enum):
    IDLE     = auto()
    RUNNING  = auto()
    PAUSED   = auto()
    DONE     = auto()


class ArmSequence:
    """
    Define una secuencia de poses con tiempos de transición.
    Cada keyframe es (pose_array, duration_seconds).
    loop=True repite la secuencia infinitamente.
    """
    def __init__(self, name: str, keyframes: list, loop: bool = False):
        self.name       = name
        self.keyframes  = keyframes   # [(pose_np, duration_s), ...]
        self.loop       = loop
        self.state      = SequenceState.IDLE
        self._frame_idx = 0
        self._t_start   = 0.0
        self._pose_start = None

    def reset(self):
        self._frame_idx  = 0
        self._t_start    = time.time()
        self._pose_start = None
        self.state       = SequenceState.RUNNING

    def start(self, current_pose: np.ndarray):
        self._pose_start = current_pose.copy()
        self.reset()

    def pause(self):
        if self.state == SequenceState.RUNNING:
            self.state = SequenceState.PAUSED
        elif self.state == SequenceState.PAUSED:
            self.state = SequenceState.RUNNING
            self._t_start = time.time()

    def stop(self):
        self.state = SequenceState.IDLE

    @property
    def active(self) -> bool:
        return self.state == SequenceState.RUNNING

    def update(self, current_pose: np.ndarray) -> np.ndarray:
        """
        Devuelve la pose interpolada para el instante actual.
        Llámala en cada frame cuando la secuencia está activa.
        """
        if self.state != SequenceState.RUNNING:
            return current_pose

        if self._frame_idx >= len(self.keyframes):
            if self.loop:
                self._frame_idx  = 0
                self._t_start    = time.time()
                self._pose_start = current_pose.copy()
            else:
                self.state = SequenceState.DONE
                return self.keyframes[-1][0]

        target_pose, duration = self.keyframes[self._frame_idx]
        elapsed = time.time() - self._t_start

        if self._pose_start is None:
            self._pose_start = current_pose.copy()

        # Progreso con easing suave (ease in-out cúbico)
        t = min(elapsed / max(duration, 0.001), 1.0)
        t_smooth = self._ease_in_out(t)
        interpolated = self._pose_start + (target_pose - self._pose_start) * t_smooth

        # Avanzar al siguiente keyframe
        if elapsed >= duration:
            self._pose_start = target_pose.copy()
            self._frame_idx += 1
            self._t_start    = time.time()

        return interpolated

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Interpolación ease-in-out cúbica."""
        if t < 0.5:
            return 4 * t * t * t
        return 1.0 - pow(-2 * t + 2, 3) / 2.0


# =============================================================================
# SECUENCIAS PREDEFINIDAS
# =============================================================================

def make_wave_sequence() -> ArmSequence:
    """Saludo: levanta el brazo y agita 5 veces."""
    keyframes = [
        (Pose.WAVE_READY, 1.2),   # levantar brazo
        (Pose.WAVE_UP,    0.25),  # agitar arriba
        (Pose.WAVE_READY, 0.25),  # agitar abajo
        (Pose.WAVE_UP,    0.25),
        (Pose.WAVE_READY, 0.25),
        (Pose.WAVE_UP,    0.25),
        (Pose.WAVE_READY, 0.25),
        (Pose.WAVE_UP,    0.25),
        (Pose.WAVE_READY, 0.25),
        (Pose.NEUTRAL,    1.5),   # bajar brazo
    ]
    return ArmSequence("Saludar", keyframes, loop=False)


def make_point_sequence() -> ArmSequence:
    """Apunta hacia adelante con el brazo derecho."""
    keyframes = [
        (Pose.POINT_RIGHT, 1.0),   # extender brazo
        (Pose.POINT_RIGHT, 2.0),   # mantener 2s
        (Pose.NEUTRAL,     1.0),   # volver
    ]
    return ArmSequence("Apuntar", keyframes, loop=False)


def make_carry_sequence() -> ArmSequence:
    """Extiende ambos brazos para cargar un objeto."""
    keyframes = [
        (Pose.CARRY,   1.5),   # extender
        (Pose.CARRY,   3.0),   # mantener
        (Pose.NEUTRAL, 1.5),   # volver
    ]
    return ArmSequence("Cargar", keyframes, loop=False)


def make_cross_sequence() -> ArmSequence:
    """Brazos en cruz."""
    keyframes = [
        (Pose.CROSS,   1.5),
        (Pose.CROSS,   2.0),
        (Pose.NEUTRAL, 1.5),
    ]
    return ArmSequence("Cruz", keyframes, loop=False)


def make_guard_sequence() -> ArmSequence:
    """Pose de guardia / defensa."""
    keyframes = [
        (Pose.GUARD,   1.0),
        (Pose.GUARD,   4.0),   # mantener 4s
        (Pose.NEUTRAL, 1.0),
    ]
    return ArmSequence("Guardia", keyframes, loop=False)


def make_neutral_sequence() -> ArmSequence:
    """Vuelta rápida a pose neutral."""
    keyframes = [
        (Pose.NEUTRAL, 1.2),
    ]
    return ArmSequence("Neutral", keyframes, loop=False)


# =============================================================================
# CONTROLADOR PRINCIPAL
# =============================================================================

class ArmController:
    """
    Controlador de brazos que gestiona secuencias automáticas
    y control manual, aplicando torques PD a los joints del brazo.

    Uso básico:
        ctrl = ArmController(model, data)
        ctrl.play("wave")          # iniciar secuencia
        # en cada frame:
        ctrl.update(manual_target) # actualizar
        ctrl.apply(data)           # aplicar torques
    """

    # Índices de los 8 joints de brazo dentro del array de 23 DOFs
    ARM_INDICES = np.arange(15, 23)   # joints 15..22

    # Ganancias PD para los brazos
    KP = np.array([80, 60, 60, 80, 80, 60, 60, 80], dtype=np.float32)
    KD = np.array([ 3,  2,  2,  2,  3,  2,  2,  2], dtype=np.float32)

    # Límites de torque por joint (Nm)
    TORQUE_LIMIT = np.array([25, 25, 25, 25, 25, 25, 25, 25], dtype=np.float32)

    # Límites articulares (rad) — aproximados para G1
    JOINT_LOWER = np.array([-3.14, -0.10, -3.14, -1.57, -3.14, -3.14, -3.14, -1.57], dtype=np.float32)
    JOINT_UPPER = np.array([ 3.14,  3.49,  3.14,  4.45,  3.14,  0.10,  3.14,  4.45], dtype=np.float32)

    def __init__(self, model, data):
        self.model = model
        self.data  = data

        # Pose actual y objetivo (comandada)
        self.current_pose = Pose.NEUTRAL.copy()
        self.target_pose  = Pose.NEUTRAL.copy()

        # Secuencias disponibles
        self.sequences: dict[str, ArmSequence] = {
            "wave":    make_wave_sequence(),
            "point":   make_point_sequence(),
            "carry":   make_carry_sequence(),
            "cross":   make_cross_sequence(),
            "guard":   make_guard_sequence(),
            "neutral": make_neutral_sequence(),
        }
        self._active_seq: ArmSequence | None = None

        # Control manual: paso por joint
        self.manual_step = 0.05
        self.manual_fine = 0.01   # paso fino con Shift

        # Estado
        self.manual_override = False   # True = ignorar secuencia activa
        self._last_status_t  = 0.0

    # ── API pública ────────────────────────────────────────────────────────

    def play(self, name: str):
        """Inicia una secuencia por nombre. Detiene la anterior."""
        if name not in self.sequences:
            print(f"[ARM] Secuencia '{name}' no existe. Opciones: {list(self.sequences.keys())}")
            return
        if self._active_seq is not None:
            self._active_seq.stop()
        self._active_seq     = self.sequences[name]
        self.manual_override = False
        self._active_seq.start(self.current_pose)
        print(f"[ARM] ▶ Secuencia: {self._active_seq.name}")

    def pause_toggle(self):
        """Pausa/reanuda la secuencia activa."""
        if self._active_seq is not None:
            self._active_seq.pause()
            status = "⏸ PAUSADA" if self._active_seq.state == SequenceState.PAUSED else "▶ REANUDADA"
            print(f"[ARM] {status}: {self._active_seq.name}")

    def stop(self):
        """Detiene la secuencia activa."""
        if self._active_seq is not None:
            self._active_seq.stop()
            self._active_seq = None
        self.manual_override = True

    def set_manual(self, joint_idx: int, delta: float):
        """
        Mueve un joint específico manualmente.
        joint_idx: 0-3 = brazo izquierdo, 4-7 = brazo derecho
        delta: desplazamiento en radianes
        """
        self.manual_override = True
        if self._active_seq is not None:
            self._active_seq.stop()
            self._active_seq = None

        new_val = self.target_pose[joint_idx] + delta
        new_val = np.clip(new_val, self.JOINT_LOWER[joint_idx], self.JOINT_UPPER[joint_idx])
        self.target_pose[joint_idx] = new_val
        names = ["L-Pitch", "L-Roll", "L-Yaw", "L-Elbow",
                 "R-Pitch", "R-Roll", "R-Yaw", "R-Elbow"]
        print(f"[ARM] {names[joint_idx]}: {new_val:+.2f} rad")

    def update(self, manual_target: np.ndarray | None = None):
        """
        Actualiza la pose objetivo. Llámala en cada frame.
        manual_target: array de 5 valores del viewer_state.arm_target (opcional).
        """
        if self._active_seq is not None and self._active_seq.active:
            # Secuencia automática tiene prioridad
            self.target_pose = self._active_seq.update(self.current_pose)
            # Verificar si terminó
            if self._active_seq.state == SequenceState.DONE:
                print(f"[ARM] ✓ Secuencia '{self._active_seq.name}' completada")
                self._active_seq = None
        elif manual_target is not None and not self.manual_override:
            # Control manual externo (viewer_state.arm_target)
            # Solo usa los primeros 5 valores (el viewer tiene 5 joints del brazo extra)
            # Para el brazo derecho:
            self.target_pose[4] = manual_target[0]   # shoulder pitch
            self.target_pose[5] = manual_target[1]   # shoulder roll
            self.target_pose[6] = manual_target[2]   # shoulder yaw
            self.target_pose[7] = manual_target[3]   # elbow

    def apply(self, data):
        """
        Calcula y aplica los torques PD a los joints del brazo.
        Llámala después de update() en cada paso de simulación.
        """
        # Posición y velocidad actuales de los 8 joints de brazo
        q  = data.qpos[22:30].astype(np.float32)   # qpos[22..29]
        qd = data.qvel[21:29].astype(np.float32)   # qvel[21..28]

        # Actualizar pose actual medida
        self.current_pose = q.copy()

        # Torque PD
        error  = self.target_pose - q
        torque = self.KP * error - self.KD * qd
        torque = np.clip(torque, -self.TORQUE_LIMIT, self.TORQUE_LIMIT)

        # Aplicar a los actuadores (joints 15..22 del robot = ctrl[15..22])
        data.ctrl[14:24] = torque

    def get_status(self) -> str:
        """Devuelve un string con el estado actual."""
        if self._active_seq is not None and self._active_seq.active:
            return f"SEQ:{self._active_seq.name}"
        if self.manual_override:
            return "MANUAL"
        return "IDLE"

    def print_pose(self):
        """Imprime la pose actual de los brazos."""
        names = ["L-Pitch", "L-Roll", "L-Yaw", "L-Elbow",
                 "R-Pitch", "R-Roll", "R-Yaw", "R-Elbow"]
        print("\n[ARM] Pose actual:")
        for i, (n, v) in enumerate(zip(names, self.current_pose)):
            bar = "█" * int(abs(v) * 5)
            print(f"  {n:10s} [{i}]: {v:+.3f} rad  {bar}")
        print(f"  Estado: {self.get_status()}")


# =============================================================================
# INTEGRACIÓN CON EL KEY CALLBACK DE banda.py
# =============================================================================
# Agrega esto en create_key_callback(), pasando arm_ctrl como parámetro:
#
# KEY_F1  = 290
# KEY_F2  = 291
# KEY_F3  = 292
# KEY_F4  = 293
# KEY_F5  = 294
# KEY_F6  = 295
# KEY_F7  = 296
#
# def create_key_callback(state: ViewerState, arm_ctrl: ArmController):
#     def key_callback(keycode):
#         ...
#         # Secuencias de brazos
#         elif keycode == 290: arm_ctrl.play("wave")
#         elif keycode == 291: arm_ctrl.play("point")
#         elif keycode == 292: arm_ctrl.play("carry")
#         elif keycode == 293: arm_ctrl.play("cross")
#         elif keycode == 294: arm_ctrl.play("guard")
#         elif keycode == 295: arm_ctrl.play("neutral")
#         elif keycode == 296: arm_ctrl.pause_toggle()
#         # Control manual fino de brazos (Shift + teclas)
#         elif keycode == ord('P'): arm_ctrl.print_pose()
#         ...

# =============================================================================
# DEMO STANDALONE (sin MuJoCo) — para probar la lógica de interpolación
# =============================================================================
if __name__ == "__main__":
    print("=== Demo ArmController (sin MuJoCo) ===\n")

    seq = make_wave_sequence()
    pose = Pose.NEUTRAL.copy()
    seq.start(pose)

    print("Simulando secuencia 'Saludar':")
    t0 = time.time()
    frames = 0
    while seq.active:
        pose = seq.update(pose)
        frames += 1
        if frames % 50 == 0:
            elapsed = time.time() - t0
            print(f"  t={elapsed:.1f}s  R-shoulder_pitch={pose[4]:+.3f}  R-elbow={pose[7]:+.3f}")
        time.sleep(0.01)

    print(f"\nSecuencia completada en {time.time()-t0:.1f}s ({frames} frames)")
    print(f"Pose final: {np.round(pose, 3)}")
    print("\nTodo OK — listo para integrar en banda.py")