#!/usr/bin/env python3
"""
pose_sender.py — Mitad "percepcion" del teleop hibrido.

Captura camara + MediaPipe Pose, calcula las direcciones de brazo en el
frame del torso (fwd/left/up), el yaw del torso y la confianza por
profundidad de cada brazo -- exactamente el mismo calculo que hacia
PoseTracker dentro de teleop_r1_v5.py -- y las manda por UDP a
teleop_r1 (C++) en localhost.

Se queda en Python porque MediaPipe en C++ puro requiere compilar con
Bazel (no vale la pena para esto). La otra mitad (IK + control del
robot) corre en C++ (teleop_r1.cpp).

FORMATO DEL PAQUETE UDP (72 bytes, little-endian, sin padding porque el
double de 8 bytes va primero y el resto son floats/uint32 de 4 bytes):

  double    timestamp         segundos (time.time())
  uint32    valid             1 = se detecto pose este frame, 0 = no
  float[15] payload = [
      yaw,                    yaw crudo del torso (rad), SIN mirror
      conf_L, conf_R,         confianza [0,1] por profundidad
      uL(3), wL(3),           brazo IZQUIERDO humano: hombro->codo, codo->muneca
      uR(3), wR(3)            brazo DERECHO humano: hombro->codo, codo->muneca
  ]                           todo en frame del torso (fwd, left, up)

El mirror (L<->R) NO se aplica aca -- eso lo hace teleop_r1.cpp en vivo
con la tecla M, igual que hacia tracker.mirror en el script original.
"""

import socket
import struct
import time

import cv2
import mediapipe as mp
import numpy as np

CAM_URL = 1   # FaceCam 1000X local
UDP_IP = "127.0.0.1"
UDP_PORT = 5555
SHOW_CAMERA = True
FORWARD_SIGN = -1.0

PACKET_FMT = "<dI15f"
PACKET_SIZE = struct.calcsize(PACKET_FMT)
assert PACKET_SIZE == 72, f"tamano de paquete inesperado: {PACKET_SIZE}"


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[POSE] Abriendo camara: {CAM_URL}")
    cap = cv2.VideoCapture(CAM_URL)
    if not cap.isOpened():
        print(f"[POSE] No abre la camara: {CAM_URL}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(model_complexity=0,
                         min_detection_confidence=0.6,
                         min_tracking_confidence=0.6)

    print(f"[POSE] Camara lista. Enviando UDP a {UDP_IP}:{UDP_PORT} "
          f"({PACKET_SIZE} bytes/paquete)")
    print("[POSE] ESC en la ventana de video para salir.\n")

    frame_count = 0
    t_fps = time.time()

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_world_landmarks:
                lm = res.pose_world_landmarks.landmark
                flip = np.array([1.0, -1.0, 1.0])
                P = lambda i: np.array([lm[i].x, lm[i].y, lm[i].z]) * flip
                LS, RS, LE, RE, LW, RW = P(11), P(12), P(13), P(14), P(15), P(16)
                LH, RH = P(23), P(24)

                up = _norm(0.5 * (LS + RS) - 0.5 * (LH + RH))
                left = LS - RS
                left = _norm(left - np.dot(left, up) * up)
                fwd = _norm(np.cross(left, up)) * FORWARD_SIGN

                segLu = _norm(LE - LS)
                segLf = _norm(LW - LE)
                segRu = _norm(RE - RS)
                segRf = _norm(RW - RE)

                inplane = lambda v: float(np.sqrt(max(0.0, 1.0 - v[2] * v[2])))
                cL = min(inplane(segLu), inplane(segLf))
                cR = min(inplane(segRu), inplane(segRf))

                sh = LS - RS
                yaw_raw = float(np.arctan2(sh[2], abs(sh[0]) + 1e-6))

                def body(v):
                    return np.array([np.dot(v, fwd), np.dot(v, left), np.dot(v, up)])

                uL, wL, uR, wR = body(segLu), body(segLf), body(segRu), body(segRf)

                payload = [yaw_raw, cL, cR, *uL, *wL, *uR, *wR]
                packet = struct.pack(PACKET_FMT, time.time(), 1, *payload)
                sock.sendto(packet, (UDP_IP, UDP_PORT))

                if SHOW_CAMERA:
                    mp_draw.draw_landmarks(frame, res.pose_landmarks,
                                            mp_pose.POSE_CONNECTIONS)
                    cv2.putText(frame,
                                f"YAW={np.degrees(yaw_raw):+.0f}  L={cL:.2f}  R={cR:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2)
            else:
                packet = struct.pack(PACKET_FMT, time.time(), 0, *([0.0] * 15))
                sock.sendto(packet, (UDP_IP, UDP_PORT))

            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30.0 / (now - t_fps)
                t_fps = now
                print(f"[POSE] FPS: {fps:.1f}")

            if SHOW_CAMERA:
                cv2.imshow("pose_sender", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
    finally:
        cap.release()
        if SHOW_CAMERA:
            cv2.destroyAllWindows()
        sock.close()
        print("\n[POSE] Fin")


if __name__ == "__main__":
    main()
