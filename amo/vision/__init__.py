"""Percepcion por camara (MediaPipe) reutilizable entre scripts."""

from amo.vision.manos_camara import SeguidorDeManos, curls_desde_landmarks
from amo.vision.cuerpo_camara import LecturaBrazos, SeguidorTrenSuperior

__all__ = ["SeguidorDeManos", "curls_desde_landmarks",
           "SeguidorTrenSuperior", "LecturaBrazos"]
