# Cómo ejecutar cada cosa

Todos los scripts resuelven sus rutas a partir de su propia ubicación, así que
**puedes lanzarlos desde cualquier directorio**. Los ejemplos usan la raíz del
repo por comodidad.

## Entornos

Hay dos, con versiones incompatibles entre sí. Antes de cada comando, activa el que toque.

| Entorno | Para qué | Cómo se crea |
|---|---|---|
| `amo` | scripts de `scripts/g1/` (código original del paper) | `conda create -n amo python=3.8 && pip install -r requirements.txt` |
| `r1mujoco` | todo lo del R1: `scripts/r1/`, `scripts/teleop/`, `scripts/tools/` | `conda create -n r1mujoco python=3.10 && pip install -r requirements-r1.txt` |

Isaac Lab (`scripts/tools/check_r1*.py`) usa su propio entorno, `isaaclab_clean`.

---

## 1. G1 — el robot del paper AMO

### `scripts/g1/play_amo.py` — política AMO completa

```bash
conda activate amo && python scripts/g1/play_amo.py
```

Teclas: `↑`/`↓` vx · `←`/`→` giro · `Q`/`E` vy lateral · `Z`/`X` altura del torso
(rango [-0.5, 0.8]) · `J`/`U` torso yaw · `K`/`I` torso pitch · `L`/`O` torso roll.

### `scripts/g1/play_amo_stable.py` — solo PD, sin IA

Mantiene al G1 de pie con control de posición puro. Útil para verificar que el
modelo y la pose base están bien antes de meter la política.

```bash
conda activate amo && python scripts/g1/play_amo_stable.py
```

### `scripts/g1/banda_v2_1.py` — G1 + gestos de brazo

```bash
conda activate amo && python scripts/g1/banda_v2_1.py
```

Teclas: `F1` saludar · `F6` neutral · `ESC` salir.

---

## 2. R1 — política nativa entrenada en Isaac Lab

### `scripts/r1/play_r1_isaac.py` — caminar

Corre la política nativa del R1 (`policies/r1_policy_v2.pt`) dentro de MuJoCo,
reconstruyendo la observación de 405 dimensiones que veía en Isaac.

```bash
conda activate r1mujoco && python scripts/r1/play_r1_isaac.py
```

Teclas: `↑`/`↓` vx · `←`/`→` giro (wz) · `Q`/`E` vy lateral · `ESPACIO` parar ·
`R` reset · `ESC` salir.

### `scripts/r1/play_r1_camina_brazos.py` — caminar + gestos

Piernas y cintura las lleva la política; al activar un gesto, el `ArmController`
toma los brazos con una transición suave y los devuelve al terminar.

```bash
conda activate r1mujoco && python scripts/r1/play_r1_camina_brazos.py
```

Teclas: las de arriba, más `F1` saludar · `F2` apuntar · `F3` cargar · `F4` cruz ·
`F5` guardia · `F6` neutral · `F7` pausar/reanudar el gesto.

### `scripts/r1/banda_r1.py` — escenario completo (LIDAR, cámara, bandas)

El más pesado: R1 con LIDAR 2D, mapeo, cámara OpenCV, dos bandas transportadoras
y brazo con gripper de ventosas.

```bash
conda activate r1mujoco && python scripts/r1/banda_r1.py
```

Teclas: movimiento igual que `play_amo.py`, más `V` LIDAR · `M` mapa · `B` volcar
lecturas del LIDAR · `C` limpiar mapa · `P` cámara · `F1`–`F8` gestos de brazo.

### `scripts/r1/banda_estabilidad_r1.py` — solo balance

Versión mínima, sin política ni sensores: comprueba que el R1 se queda de pie.
Es lo primero que hay que correr cuando tocas `assets/r1.xml`.

```bash
conda activate r1mujoco && python scripts/r1/banda_estabilidad_r1.py
```

---

## 3. Teleoperación con cámara

Ambos usan MediaPipe. Ponte de frente a la cámara, a ~2 m, que se te vea de la
cintura para arriba.

**Cámara del celular con DroidCam por USB:**

```bash
adb forward tcp:4747 tcp:4747
```

Y por WiFi, edita `CAM_URL` en el script y pon `http://IP_DEL_CELULAR:4747/video`.

### `scripts/teleop/teleop_manos.py` — cuerpo y brazos

Resuelve todas las direcciones de joints **por nombre**, así que funciona igual
con `r1.xml` y con `r1_manos.xml`. Los dedos quedan sueltos (torque 0).

```bash
conda activate r1mujoco && python scripts/teleop/teleop_manos.py
```

Teclas: `T` teleop on/off · `M` espejo · `K` cinemático/dinámico · `B` soporte ·
`R` reset · `ESC` salir.

### `scripts/teleop/teleop_dedos.py` — lo anterior + dedos

Usa MediaPipe **Holistic** para sacar cuerpo y las dos manos en una sola pasada,
y mapea el "curl" de cada dedo al rango real de cada joint Revo2.

```bash
conda activate r1mujoco && python scripts/teleop/teleop_dedos.py
```

Teclas: las de arriba, más `F` dedos on/off.

### `scripts/tools/test_pose.py` — probar la cámara

Antes de pelearte con la teleoperación, comprueba que MediaPipe te ve:

```bash
conda activate r1mujoco && python scripts/tools/test_pose.py
```

---

## 4. Herramientas

### `scripts/tools/calibrar_brazos.py` — descubrir el sentido de cada joint

Sin física: el robot flota y mueves un joint a la vez para ver hacia dónde dobla.

```bash
conda activate r1mujoco && python scripts/tools/calibrar_brazos.py
```

Teclas: `N`/`P` siguiente/anterior joint · `↑`/`↓` ±0.1 rad · `0` joint a cero ·
`I` pose idle · `Z` todos los brazos a cero · `ESC` salir.

### `scripts/tools/check_r1.py` — orden de joints Isaac vs MuJoCo

Imprime el mapeo `mujoco_from_isaac`, que es el "array de oro" que usan los
scripts de `scripts/r1/`. Córrelo cada vez que cambies el USD o el XML.

```bash
conda activate isaaclab_clean && python scripts/tools/check_r1.py
```

### `scripts/tools/check_r1_bodies.py` — listar los bodies del R1

```bash
conda activate isaaclab_clean && python scripts/tools/check_r1_bodies.py
```

### `scripts/tools/inspeccionar_checkpoint.py` — ver las capas de un checkpoint

```bash
python scripts/tools/inspeccionar_checkpoint.py --ckpt third_party/IsaacLab/logs/rsl_rl/2026-06-23_17-40-45/model_5400.pt
```

### `scripts/tools/export_policy_v2.py` — checkpoint → TorchScript

Genera el `.pt` que cargan los scripts de `scripts/r1/`.

```bash
python scripts/tools/export_policy_v2.py --out policies/r1_policy_v3.pt
```

Sin argumentos usa el último run conocido y sobrescribe `policies/r1_policy_v2.pt`.

---

## 5. C++

### `cpp/mujoco/` — versión C++ de la simulación

```bash
cmake -S cpp/mujoco -B cpp/mujoco/build -DCMAKE_PREFIX_PATH=/ruta/a/libtorch
cmake --build cpp/mujoco/build -j
```

Se generan dos binarios (`play_r1_isaac` solo si encuentra LibTorch). Ojo: hay
que lanzarlos **desde la raíz del repo**, porque sus rutas por defecto
(`assets/r1.xml`, `policies/r1_policy_v2.pt`) son relativas al directorio actual:

```bash
./cpp/mujoco/build/play_r1_isaac
./cpp/mujoco/build/teleop_r1
```

O pasando las rutas a mano: `./cpp/mujoco/build/teleop_r1 assets/r1_manos.xml`.

El teleop en C++ es solo la mitad de control: la percepción sigue en Python
porque MediaPipe en C++ obliga a compilar con Bazel. Hay que correr las dos
partes a la vez, en terminales separadas:

```bash
python cpp/mujoco/pose_sender.py       # manda pose por UDP a 127.0.0.1:5555
```

```bash
./cpp/mujoco/build/teleop_r1           # recibe y controla
```

Teclas de `teleop_r1`: `T` teleop · `M` espejo · `K` cinemático/dinámico ·
`B` soporte · `R` reset · `ESC` salir.

### `cpp/deploy/` — despliegue en el robot real (unitree_sdk2 + DDS)

```bash
cmake -S cpp/deploy -B cpp/deploy/build -DCMAKE_PREFIX_PATH=/ruta/a/libtorch
cmake --build cpp/deploy/build -j
```

```bash
./cpp/deploy/build/r1_deploy policies/r1_policy_v2.pt
```

Antes de conectarlo al robot real hay que cambiar `DOMAIN_ID` y `NET_IFACE` en
`cpp/deploy/r1_deploy.cpp`: para validar contra `unitree_mujoco` en loopback van
en `1` y `"lo"`; para el robot real, `0` (o el que uses) y tu interfaz de red.
Quedan pendientes de verificar en el robot real el orden de motores `SDK_INDEX[]`,
la convención de la IMU y el `mode()` del `motor_cmd` — está anotado en la
cabecera del archivo.

### `scripts/deploy/saludo_g1_real.py` — saludo del G1 por el SDK

Pregunta por consola cuántas veces saludar. Sin argumentos va contra `unitree_mujoco`
en loopback; con argumento, contra el robot real por esa interfaz de red.

```bash
python scripts/deploy/saludo_g1_real.py          # simulación (interfaz lo, domain 1)
```

```bash
python scripts/deploy/saludo_g1_real.py eth0     # robot real
```

---

## 6. Versiones anteriores

`legacy/` conserva las versiones superadas, con las rutas ya actualizadas. Ver
[legacy/README.md](../legacy/README.md) para saber qué reemplazó a qué.
