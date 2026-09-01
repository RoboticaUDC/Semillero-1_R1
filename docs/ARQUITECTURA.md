# Arquitectura

Cómo está organizado el proyecto, qué hace cada componente y qué datos circulan
entre ellos.

---

## 1. La idea de fondo

Todo el proyecto es la misma estructura repetida, con capas encima:

```
    escena MuJoCo (.xml)  ──►  bucle de simulación 500 Hz  ──►  visor
                                        ▲       │
                          objetivos PD  │       │  estado (qpos, qvel, sensores)
                                        │       ▼
                                  ┌─────────────────────┐
                                  │  capa de control    │
                                  └─────────────────────┘
                                   política (.pt)  ·  gestos  ·  IK de cámara
                                   agente LLM      ·  voz     ·  teclado
```

El bucle de simulación es **el único que toca `data.qpos` / `data.ctrl`**. Todo lo
que tarde más de un paso de simulación (una llamada a un LLM son 0.5–3 s, la IK
son ~30 `mj_forward`, MediaPipe son ~30 ms por frame) corre en su propio hilo y se
comunica por cola o por snapshot bajo lock. Es la restricción que explica casi
todas las decisiones de diseño que siguen.

---

## 2. Los dos robots

### Unitree G1 — `assets/g1.xml`

23 articulaciones. Es el robot del paper AMO y su código llegó ya escrito.

```
qpos[0:7]    base libre: x, y, z + quaternion (w, x, y, z)
qpos[7:30]   23 joints:  6 pierna izq · 6 pierna der · 3 cintura (yaw, roll, pitch)
                         · 4+4 brazos (shoulder pitch/roll/yaw, elbow) ... según variante
```

Políticas: `amo_jit.pt` (la política AMO, 17.8 MB) más `adapter_jit.pt` y
`adapter_norm_stats.pt` (el adaptador y su normalización).

### R1 — `assets/r1.xml` y `assets/r1_manos.xml`

24 articulaciones, entrenado desde cero en Isaac Lab. Orden en `qpos` tras la base:

```
 0- 5  pierna izquierda   hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
 6-11  pierna derecha     igual
12-13  cintura            waist_roll, waist_yaw
14-18  brazo izquierdo    shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
19-23  brazo derecho      igual
```

`r1_manos.xml` es el mismo robot con las manos **Revo2 de BrainCo**: 11
articulaciones por mano, 22 en total.

> **Esta es la trampa número uno del proyecto.** MuJoCo inserta los 11 joints de
> la mano izquierda *entre* el brazo izquierdo y el derecho. Con manos puestas,
> `qpos[26..30]` ya no es el brazo derecho: son dedos. Por eso todo el código
> nuevo resuelve las direcciones **por nombre** (`jnt_qposadr`, `jnt_dofadr`, el
> actuador de cada joint) una sola vez al arrancar, y nunca por índice fijo. Las
> versiones de `legacy/` que leían por índice se rompen con el XML de manos.

Políticas: `r1_policy.pt` (primera versión) y `r1_policy_v2.pt` (la que se usa).

---

## 3. El paquete `amo/` — lo que comparten todos los scripts

Ningún script importa de otro script. Lo común vive aquí.

### `amo/paths.py`

Resuelve `assets/` y `policies/` a partir de la ubicación del propio archivo, no
del directorio de trabajo. Por eso los scripts corren desde donde sea.

```python
from amo.paths import scene, policy
model = mujoco.MjModel.from_xml_path(scene("r1"))       # o "r1.xml", o ruta absoluta
net   = torch.jit.load(policy("r1_v2"))                  # o "r1_policy_v2.pt"
```

Alias disponibles: escenas `g1`, `r1`, `r1_manos`; políticas `amo`, `adapter`,
`adapter_stats`, `r1`, `r1_v2`. Un alias inexistente da un `FileNotFoundError` que
lista los válidos.

### `amo/math_utils.py`

`quat_to_euler(q)` y `quat_rotate_inverse(q, v)`. Convención MuJoCo: los
quaternions son `(w, x, y, z)`, que es lo que hay en `qpos[3:7]` y lo que devuelve
el sensor `framequat`. `quat_rotate_inverse(q, [0,0,-1])` da el `projected_gravity`
que esperan las políticas entrenadas en Isaac Lab.

### `amo/control/`

| Clase | Qué hace |
|---|---|
| `ArmController`, `ArmSequence` | gestos de brazo animados por keyframes con interpolación *ease-in-out*, transición suave al entrar y salir del gesto |
| `IKBrazos` | cinemática inversa de los brazos: Gauss-Newton amortiguado sobre 4 joints por brazo |
| `ControladorBrazosCamara` | hilo que resuelve la IK a 30 Hz contra su propia copia del modelo y publica el resultado |
| `ControladorManos` | los 22 joints de dedos: curls → ángulos → PD |

**Gestos disponibles** (`wave`, `point`, `carry`, `cross`, `guard`, `neutral`),
mapeados a `F1`–`F6` en los scripts y expuestos como herramienta al LLM.

**Poses de mano** (`abierta`, `relajada`, `puno`, `garra`, `pinza`, `ok`,
`senalar`, `pulgar_arriba`, `paz`). Cada gesto de brazo arrastra su pose de dedos
coordinada: `wave` abre la mano, `point` extiende el índice, `guard` cierra los
puños.

### `amo/vision/`

| Clase | Detector | Qué entrega |
|---|---|---|
| `SeguidorDeManos` | `mp.solutions.hands` | curls por dedo, ligero, etiqueta izquierda/derecha |
| `SeguidorTrenSuperior` | `mp.solutions.holistic` | brazos + torso + las dos manos en una pasada |

Se usa uno **o** el otro: un mismo stream MJPEG no se puede abrir dos veces sin
pelearse. Cuando quieres brazos, Holistic sustituye al de manos y sirve los dos.

Ambos aplican el espejo *dentro* del seguidor, así que quien los consume recibe
todo ya en términos del robot: `"izquierda"` es la izquierda del robot.

---

## 4. Cómo camina el R1 — el flujo de la política

Es el núcleo del proyecto. `scripts/r1/play_r1_isaac.py` reconstruye dentro de
MuJoCo exactamente la observación que la red veía en Isaac Lab.

```
        MuJoCo                        reordenar             red
  qpos, qvel, quaternion  ──►  orden MuJoCo → Isaac  ──►  obs(405)  ──►  acción(24)
                                                                             │
   ctrl (torque PD)  ◄──  orden Isaac → MuJoCo  ◄──  default + 0.25·acción  ◄┘
```

**Observación — 405 valores**, con historial de 5 pasos aplanado `[t-4 … t]`:

| Bloque | Dimensión | Escala |
|---|---|---|
| `base_ang_vel` | 3 × 5 = 15 | ×0.2 |
| `projected_gravity` | 3 × 5 = 15 | ×1.0 |
| `velocity_commands` (vx, vy, wz) | 3 × 5 = 15 | ×1.0 |
| `joint_pos_rel` | 24 × 5 = 120 | ×1.0 |
| `joint_vel_rel` | 24 × 5 = 120 | ×0.05 |
| `last_action` | 24 × 5 = 120 | ×1.0 |

**Arquitectura de la red:** `405 → 512 → 256 → 128 → 24`, exportada a TorchScript.

**El "array de oro":** Isaac Lab y MuJoCo numeran las articulaciones en orden
distinto. `mujoco_from_isaac` es el vector de permutación que traduce entre los
dos. Lo imprime `scripts/tools/check_r1.py`, y hay que regenerarlo **cada vez que
cambie el USD o el XML**. Si está mal, el robot se mueve pero de forma incoherente,
que es el fallo más difícil de diagnosticar del proyecto.

**Comandos:** `vx`, `vy`, `wz` adimensionales, recortados a `[-1, 1]` porque es el
rango en el que se entrenó. Por encima de 0.6 el robot se vuelve inestable.

---

## 5. Las capas de control, de menos a más

Cada script añade una capa sobre el anterior sin modificarlo: los de arriba
importan a los de abajo con `importlib` y heredan de su clase de entorno.

```
banda_estabilidad_r1.py   solo PD, sin política ni sensores. Se queda de pie.
        │
play_r1_isaac.py          + política nativa. Camina con las flechas.
        │
play_r1_camina_brazos.py  + gestos de brazo (F1–F7) mezclados con la política.
        │
play_r1_ia.py             + agente LLM por consola: lenguaje natural → herramientas.
        │
play_r1_voz.py            + manos Revo2, voz en español, brazos y dedos por cámara.
```

### El reparto de articulaciones

Cuando conviven la política y las capas de arriba, se reparten los DOF:

| Articulaciones | Quién manda | Cómo |
|---|---|---|
| piernas + cintura (0–13) | siempre la política | directo |
| brazos (14–23) | política por defecto; el gesto o la IK toman el control al activarse | mezcla progresiva de ~0.5 s al entrar y al salir |
| dedos (24–45, solo con manos) | nunca la política: PD propio | pose fija, gesto o cámara |
| `waist_yaw` | política; la cámara lo mezcla con `--brazos-camara` | recortado a ±0.35 rad |

La observación que ve la política **congela los brazos** (mete la pose por defecto
en la obs en vez de la real), así que la política ni se entera de que otro le está
moviendo los brazos. Es lo que permite gesticular sin desestabilizar la marcha.

Los dedos son DOF aparte con su propio PD, así que tampoco interfieren con el
equilibrio.

---

## 6. El agente LLM

### Hilos

```
    [hilo consola]  stdin ──► LLM ──► ejecuta herramientas
    [hilo oído]     micrófono 16 kHz ──► Vosk ──► texto ──► LLM
                                  │
                                  ▼
                       BusDeIntenciones (queue.Queue)
                                  │
                                  ▼
    [hilo robot]    bucle MuJoCo 500 Hz: drena la cola, nunca bloquea
                                  │
                                  ▼ snapshot a 10 Hz (bajo lock)
                       lo leen los hilos de LLM y de voz
```

El hilo del robot jamás toca la red. El hilo del LLM jamás toca `data.qpos`: lee el
snapshot. Consola y voz comparten un mismo backend detrás de un lock
(`BackendCompartido`), así que puedes hablarle o escribirle indistintamente.

Mientras el robot habla, el micrófono se silencia, para no escucharse a sí mismo.

### Las intenciones — lo que viaja por el bus

`Mover(vx, vy, wz, hasta)` · `Parar()` · `Gesto(nombre)` · `Reiniciar()`
· y en la versión con manos, `PoseManos`, `SeguirManos`, `SeguirBrazos`.

`Mover` lleva un instante de vencimiento absoluto (`time.monotonic() + duración`),
no una duración: el bucle de control pone el comando a cero cuando vence. Sin eso,
un LLM que se cuelga deja al robot caminando indefinidamente.

### Las herramientas — el catálogo cerrado

| Herramienta | Argumentos | Qué hace |
|---|---|---|
| `mover` | `vx`, `vy`, `wz` ∈ [-1,1], `duracion_s` ≤ 30 (requerido) | camina y se detiene solo |
| `parar` | — | comando a cero |
| `gesto` | `nombre` ∈ {wave, point, carry, cross, guard, neutral} | gesto de brazo |
| `consultar_estado` | — | devuelve el snapshot en JSON |
| `reiniciar` | — | robot al origen, de pie |
| `mano` * | lado + pose del catálogo | cierra/abre los dedos |
| `seguir_manos` * | on/off | los dedos copian los tuyos por cámara |
| `seguir_brazos` * | on/off | los brazos copian los tuyos por cámara |

\* solo en `play_r1_voz.py`.

Los argumentos se recortan al rango válido antes de llegar al robot; una
herramienta que reviente devuelve un `"ERROR: …"` como texto y no tumba el agente.

**Por qué un catálogo cerrado:** el LLM elige de una lista de acciones ya
verificadas y nunca genera ángulos de articulación. Un LLM produciendo diez floats
de radianes da poses imposibles, autocolisiones y tirones que desestabilizan el
balance.

### Los backends

Todo lo específico de un proveedor está aislado en una subclase de `BackendLLM`:

| Backend | Necesita | Para qué |
|---|---|---|
| `simulado` | nada | reglas por palabras clave, sin red ni key. Es el que va por defecto |
| `openai` | `pip install openai` + `OPENAI_API_KEY` | GPT |
| `anthropic` | `pip install anthropic` + `ANTHROPIC_API_KEY` | Claude |
| `plantilla` | escribir 3 métodos | esqueleto para enchufar cualquier otra API |

---

## 7. Teleoperación por cámara

```
   cámara ──► MediaPipe ──► direcciones + curls ──► IK (hilo, 30 Hz) ──► joints
```

La IK **no persigue la posición de la muñeca**, persigue **dos direcciones por
brazo**:

```
u = hombro → codo      (a dónde apunta el brazo)
w = codo   → muñeca    (a dónde apunta el antebrazo)
```

Es lo que la cámara puede medir con dignidad: las longitudes del humano no son las
del robot, pero los ángulos sí se copian. El residual pesa más la dirección del
brazo (`W_BRAZO = 2.0`) porque un error ahí arrastra todo el antebrazo.

Se resuelven 4 joints por brazo (shoulder pitch/roll/yaw + elbow). El `wrist_roll`
queda fuera: la cámara no da su giro de forma fiable y meterlo empeora la solución.

**La ganancia de confianza** es la pieza que hace esto usable. Un brazo que apunta
a la cámara se ve como un punto y su dirección es ruido puro. `LecturaBrazos.ganancia`
mide cuánto se sale el brazo del plano de la imagen y da un peso en `[0, 1]`:
0 = ignórame, 1 = fíate.

**El curl de los dedos:** por cada dedo se suman los ángulos de sus dos
articulaciones y se normaliza contra un puño cerrado (`CURL_COMPLETO = 2.6` rad;
el pulgar tiene su propia escala, 2.0, porque tiene una falange menos).

```
curl 0 → dedo estirado → joint en jnt_range[0]
curl 1 → dedo cerrado  → joint en jnt_range[1]
```

Los límites se leen del XML, nunca se escriben a mano: si cambias el modelo, esto
sigue funcionando. Los 5 curls se expanden a los 11 joints de la mano; el pulgar
alimenta también su metacarpo, que es el que lo cierra a través de la palma.

---

## 8. Los datos del proyecto

### Snapshot de estado (JSON) — lo que el robot sabe de sí mismo

Lo publica el bucle a 10 Hz y lo devuelve la herramienta `consultar_estado`:

```json
{
  "listo": true,
  "posicion_m": [0.0, 0.0, 0.0],
  "orientacion_grados": 0.0,
  "velocidad_ms": 0.0,
  "comando_actual": [0.0, 0.0, 0.0],
  "gesto_activo": null,
  "altura_pelvis_m": 0.74,
  "caido": false
}
```

`caido` se decide por la altura de la pelvis: por debajo de **0.45 m** se considera
caída (el R1 de pie está en ~0.74 m).

### Paquete UDP del teleop híbrido — 72 bytes, little-endian

`cpp/mujoco/pose_sender.py` (percepción, Python) → `teleop_r1` (control, C++), a
`127.0.0.1:5555`:

```
double     timestamp        segundos (time.time())
uint32     valid            1 = se detectó pose este frame
float[15]  yaw,             giro del torso en rad, SIN espejo
           conf_L, conf_R,  confianza [0,1] por profundidad
           uL(3), wL(3),    brazo izquierdo humano: hombro→codo, codo→muñeca
           uR(3), wR(3)     brazo derecho
```

Todo en frame del torso (adelante, izquierda, arriba). El espejo se aplica en el
lado C++ con la tecla `M`, no aquí. La percepción se quedó en Python porque
MediaPipe en C++ obliga a compilar con Bazel.

### Artefactos de modelo

| Archivo | Qué es | Tamaño |
|---|---|---|
| `policies/amo_jit.pt` | política AMO del paper (G1) | 17.8 MB |
| `policies/adapter_jit.pt` | adaptador de AMO | 1.7 MB |
| `policies/adapter_norm_stats.pt` | media y desviación del adaptador | 2 KB |
| `policies/r1_policy.pt` | R1 nativa, primera versión | 1.5 MB |
| `policies/r1_policy_v2.pt` | R1 nativa, la que se usa | 1.5 MB |
| `assets/r1_description/R1.urdf`, `r1.usd` | descripción del R1 para Isaac Lab | — |
| `assets/meshes/*.STL` | mallas de colisión y visuales | — |

Del checkpoint de entrenamiento al `.pt` que carga MuJoCo:

```
logs/rsl_rl/<run>/model_N.pt  ──►  inspeccionar_checkpoint.py  (ver arquitectura)
                              ──►  export_policy_v2.py         ──►  policies/*.pt
```

---

## 9. El C++

Ports directos, misma lógica y mismas constantes que sus equivalentes en Python.

| Binario | Equivale a | Depende de |
|---|---|---|
| `cpp/mujoco/play_r1_isaac` | `scripts/r1/play_r1_isaac.py` | MuJoCo C + GLFW + LibTorch |
| `cpp/mujoco/teleop_r1` | `scripts/teleop/teleop_manos.py` (mitad de control) | MuJoCo C + GLFW + Eigen |
| `cpp/deploy/r1_deploy` | — (no hay equivalente en Python) | LibTorch + unitree_sdk2 + DDS |

`r1_deploy` es el único que habla con hardware. Publica por DDS con el IDL
`unitree_hg` (el R1 tiene 24 actuadores, por encima de los 20 de `NUM_MOTOR_IDL_GO`,
así que `unitree_mujoco` lo sirve por `G1Bridge`).

Se configura cambiando dos constantes en `r1_deploy.cpp`:

| Modo | `DOMAIN_ID` | `NET_IFACE` |
|---|---|---|
| validación contra `unitree_mujoco` en loopback | `1` | `"lo"` |
| robot real | `0` (o el que uses) | tu interfaz de red |

**Pendiente de verificar contra el robot físico** (anotado en la cabecera del
archivo): el orden de motores `SDK_INDEX[]`, la convención de la IMU y el `mode()`
del `motor_cmd`. En simulación no importan; en el robot real sí.

---

## 10. Convenciones del código

- **Todo en español**, incluidos nombres de clases, métodos y variables del código
  nuevo. El código heredado del paper mantiene sus nombres en inglés.
- **Resolver por nombre, nunca por índice** cuando se trate de articulaciones. Los
  índices fijos se rompen al cambiar de XML.
- **El prólogo de tres líneas** al inicio de cada script hace importable `amo/` sin
  instalar nada:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  ```
- **Cada script lleva su docstring de cabecera** con qué hace, la arquitectura, el
  uso y las teclas. Es la documentación de referencia de cada uno; esto es el mapa.
- **Nada se borra**: lo superado se mueve a `legacy/` con las rutas actualizadas y
  una nota de por qué se reemplazó.
