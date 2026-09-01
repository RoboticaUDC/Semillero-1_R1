"""Codigo compartido entre todos los scripts de AMO / R1.

Submodulos:
    amo.paths        rutas del repo (assets, policies) resueltas sin depender del CWD
    amo.math_utils   conversiones de quaternion usadas por casi todos los scripts
    amo.control      controladores reutilizables (brazos, gestos, manos)
    amo.vision       percepcion por camara (seguimiento de dedos con MediaPipe)
"""

__all__ = ["paths", "math_utils", "control", "vision"]
