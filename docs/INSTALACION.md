# Instalación en una máquina nueva

De cero a "el R1 camina en pantalla". Probado en Ubuntu con Linux 7.0 y CUDA
opcional.

---

## 0. Lo que necesitas antes de empezar

| Requisito | Por qué | Obligatorio |
|---|---|---|
| Miniconda / Anaconda | los dos entornos del proyecto | sí |
| Un display con GL (`:0` o `:1`) | MuJoCo abre ventanas y renderiza | sí |
| `git` | clonar los `third_party/` | sí |
| GPU NVIDIA + CUDA | solo `scripts/g1/play_amo.py` y entrenar en Isaac Lab | no |
| Micrófono y altavoces | `play_r1_voz.py` | no |
| Webcam o celular con DroidCam | teleoperación y `--dedos-camara` | no |
| LibTorch + MuJoCo C + unitree_sdk2 | compilar `cpp/` | no |

```bash
git clone <url-del-repo> AMO && cd AMO
```

---

## 1. Los dos entornos conda

Hay **dos entornos con versiones incompatibles entre sí**. No es un capricho: el
código del paper AMO está escrito contra numpy 1.23 y mujoco 3.2, y el código del
R1 contra numpy 2.2 y mujoco 3.10. Fijar los dos en el mismo entorno rompe uno.

### `robotica` — el G1 (código del paper)

```bash
conda create -n robotica python=3.11 -y && conda activate robotica && pip install -r requirements.txt
```

Aporta: `numpy 1.23.5`, `mujoco 3.2.3`, `mujoco-python-viewer 0.1.4`, `glfw 2.9.0`,
`torch`. Los scripts de `scripts/g1/` son los únicos que usan `mujoco_viewer`
(el paquete viejo, distinto de `mujoco.viewer`), y solo existe aquí.

> El entorno instalado hoy tiene además `imageio` y `pillow` (los usa
> `test/G1/evidencia_g1/render_g1.py`) y una versión de torch más nueva que la
> fijada en `requirements.txt`. Si quieres reproducir exactamente el entorno del
> paper, respeta el pin; si quieres reproducir *esta* máquina, añade
> `imageio pillow` después del `pip install`.

### `r1mujoco` — todo lo demás

```bash
conda create -n r1mujoco python=3.10 -y && conda activate r1mujoco && pip install -r requirements-r1.txt
```

Aporta: `mujoco 3.10`, `numpy 2.2`, `torch`, `scipy`, `mediapipe`,
`opencv-contrib-python`, `matplotlib`, `openai`. Cubre `scripts/r1/`,
`scripts/teleop/`, `scripts/tools/` y `test/robustez_r1/`.

### `isaaclab_clean` — entrenar y comprobar el orden de joints

Solo hace falta para `scripts/tools/check_r1.py`, `check_r1_bodies.py` y para
reentrenar la política. Se instala siguiendo la
[documentación de Isaac Lab](https://isaac-sim.github.io/IsaacLab/), no desde este
repo. Requiere GPU NVIDIA.

### Comprobación

```bash
conda run -n r1mujoco python -c "import mujoco, torch, numpy; print(mujoco.__version__, torch.__version__, numpy.__version__)"
```

---

## 2. Extras opcionales, según lo que quieras usar

### Voz (`play_r1_voz.py`)

Reconocimiento y síntesis, ambos **offline**: no sale audio de la máquina.

```bash
conda activate r1mujoco && conda install -c conda-forge portaudio -y && pip install vosk sounddevice piper-tts
```

Los modelos van en `modelos_voz/` en la raíz del repo (está en `.gitignore`, se
descargan aparte):

```
modelos_voz/
├── vosk-model-small-es-0.42/          reconocimiento en español, ~40 MB
├── es_ES-davefx-medium.onnx           síntesis en español, ~60 MB
└── es_ES-davefx-medium.onnx.json      configuración de la voz
```

Vosk: <https://alphacephei.com/vosk/models> · Piper: <https://github.com/rhasspy/piper>

### Agente LLM (`play_r1_ia.py`, `play_r1_voz.py`)

```bash
pip install openai      # backend --backend openai
pip install anthropic   # backend --backend anthropic
```

Sin ninguno de los dos, el backend por defecto (`simulado`) funciona igual: usa
reglas por palabras clave, sin red ni API key. Sirve para demostrar el sistema
completo antes de tener credenciales.

La key va en un `.env` **en la raíz del repo**:

```
OPENAI_API_KEY=sk-proj-...
AMO_LLM_BACKEND=openai
```

`.env` está en `.gitignore`; la key nunca viaja en el código ni en git. Las
variables ya exportadas en el entorno tienen prioridad sobre el `.env`.

### Cámara (teleoperación y `--dedos-camara` / `--brazos-camara`)

`mediapipe` y `opencv` ya vienen en `requirements-r1.txt`. Para usar el celular
como cámara con DroidCam por USB:

```bash
adb forward tcp:4747 tcp:4747
```

y la fuente pasa a ser `http://127.0.0.1:4747/video` (es el valor por defecto).
Por WiFi, `http://IP_DEL_CELULAR:4747/video`. Con webcam local basta con pasar el
índice: `--camara 0`.

Comprueba que MediaPipe te ve antes de pelearte con nada más:

```bash
conda activate r1mujoco && python scripts/tools/test_pose.py
```

---

## 3. Repositorios externos (`third_party/`)

No están versionados aquí: cada uno es su propio repo git. Solo hacen falta para
entrenar (Isaac Lab), validar el despliegue contra un simulador DDS
(`unitree_mujoco`) o consultar la descripción de las manos (`brainco`).

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/isaac-sim/IsaacLab.git
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
git clone https://github.com/unitreerobotics/unitree_mujoco.git
git clone https://github.com/BrainCoTech/brainco-description.git
```

Para correr los scripts de `scripts/r1/` y `scripts/teleop/` **no hace falta
ninguno**: los assets y las políticas ya están en `assets/` y `policies/`.

---

## 4. Compilar el C++ (opcional)

Los ports a C++ hacen exactamente lo mismo que sus equivalentes en Python; solo
los necesitas para el robot real o para medir latencias.

### Dependencias

| Dependencia | Dónde | Notas |
|---|---|---|
| MuJoCo (C) | `~/.mujoco/mujoco-3.3.6` | ruta por defecto en el CMake; se cambia con `-DMUJOCO_DIR=` |
| GLFW3 | `apt install libglfw3-dev` | visor |
| Eigen 3 | `apt install libeigen3-dev` | solo `teleop_r1` |
| LibTorch (ABI cxx11) | descarga de pytorch.org | se pasa con `-DCMAKE_PREFIX_PATH=` |
| unitree_sdk2 | instalado en `/usr/local` | solo `cpp/deploy/` |

### Simulación

```bash
cmake -S cpp/mujoco -B cpp/mujoco/build -DCMAKE_PREFIX_PATH=/ruta/a/libtorch && cmake --build cpp/mujoco/build -j
```

Salen dos binarios. `teleop_r1` no depende de Torch y se compila siempre;
`play_r1_isaac` solo si CMake encuentra LibTorch (si no, avisa y lo omite).

### Despliegue en hardware

```bash
cmake -S cpp/deploy -B cpp/deploy/build -DCMAKE_PREFIX_PATH=/ruta/a/libtorch && cmake --build cpp/deploy/build -j
```

> El `CMakeLists.txt` de `deploy` pone `/usr/include` **antes** que las rutas de
> Torch a propósito: Torch arrastra su propia copia de `fmt/core.h`, más nueva que
> la del sistema, y choca con el `libspdlog-dev` de apt que usa `dds_wrapper`. Sin
> ese `BEFORE`, la compilación falla.

---

## 5. Verificación final

```bash
conda run -n r1mujoco python scripts/r1/banda_estabilidad_r1.py
```

Si el R1 se queda de pie sin caerse, el modelo, las rutas y el contexto GL están
bien. Luego:

```bash
conda run -n r1mujoco python scripts/r1/play_r1_isaac.py
```

Si camina con las flechas, la política y el orden de joints también.

---

## Problemas conocidos

**`ModuleNotFoundError: mujoco_viewer`** — estás corriendo un script de
`scripts/g1/` en `r1mujoco`. Los del G1 van en `robotica`.

**`Torch not compiled with CUDA enabled`** — `scripts/g1/play_amo.py` exige GPU
(imprime `torch.cuda.get_device_name(0)` al arrancar, sin guardas). En una máquina
sin GPU no arranca. El resto del proyecto corre en CPU sin problema.

**EGL falla al renderizar offscreen** — en esta máquina `MUJOCO_GL=egl` da error de
EGL. La solución que funciona es usar `glfw` sobre un display real:
`DISPLAY=:1 MUJOCO_GL=glfw ...`.

**`requirements.txt` tiene una línea pegada por error al final** — un comando de
shell suelto (`DISPLAY=:1 MUJOCO_GL=glfw conda run ...`) que rompe
`pip install -r requirements.txt`. Bórrala antes de instalar.

**El `.env` no se lee** — el parser inline de `play_r1_ia.py` usa
`Path.read_text()` sin `encoding`, así que un `.env` con BOM, en Latin-1 o con
bytes binarios revienta con `UnicodeDecodeError`. Está documentado como hallazgo
abierto en [PRUEBAS.md](PRUEBAS.md).
