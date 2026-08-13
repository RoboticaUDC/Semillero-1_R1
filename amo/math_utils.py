"""Conversiones de quaternion. Estaban copiadas en casi todos los scripts.

Convencion de MuJoCo: los quaternions son (w, x, y, z), que es lo que hay en
qpos[3:7] del free joint y lo que devuelve el sensor framequat.
"""

import numpy as np


def quat_to_euler(q):
    """Quaternion (w, x, y, z) -> [roll, pitch, yaw] en radianes.

    Devuelve un np.ndarray de 3, asi que tambien se puede desempaquetar:
        roll, pitch, yaw = quat_to_euler(q)
    """
    qw, qx, qy, qz = q

    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)

    # cerca de los polos el arcsin se sale de [-1, 1] por error numerico
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.copysign(np.pi / 2.0, sinp) if abs(sinp) >= 1.0 else np.arcsin(sinp)

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)

    return np.array([roll, pitch, yaw])


def quat_rotate_inverse(q_wxyz, v):
    """Pasa el vector v del frame mundo al frame del cuerpo.

    Con v = [0, 0, -1] da el projected_gravity que esperan las politicas
    entrenadas en Isaac Lab.
    """
    q_wxyz = np.asarray(q_wxyz, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    w = q_wxyz[0]
    q_vec = q_wxyz[1:]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * (2.0 * w)
    c = q_vec * (2.0 * np.dot(q_vec, v))
    return a - b + c


# Alias historico: play_amo.py del paper original lo llama asi.
quatToEuler = quat_to_euler
