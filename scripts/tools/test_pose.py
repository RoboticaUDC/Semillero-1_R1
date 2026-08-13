import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture('http://127.0.0.1:4747/video')   # cambia a 1 o 2 si tu webcam no es la 0
if not cap.isOpened():
    print("ERROR: no se pudo abrir la webcam. Prueba VideoCapture(1) o (2).")
    raise SystemExit

print("Ventana abierta. Ponte frente a la camara. ESC para salir.")
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # espejo, mas intuitivo
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if res.pose_landmarks:
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # imprimir un angulo de ejemplo: codo derecho
        lm = res.pose_landmarks.landmark
        print(f"\rDetectando cuerpo OK  (visibilidad hombro der: {lm[12].visibility:.2f})   ", end="")
    cv2.imshow("MediaPipe Pose - ESC para salir", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
