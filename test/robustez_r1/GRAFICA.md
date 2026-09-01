# Evidencia gráfica — R1 en el visor MuJoCo

Este documento complementa la evidencia de código ([README.md](README.md)) con
evidencia **visual** de los scripts de R1 corriendo: qué se ve cuando funcionan,
qué se ve cuando **no** funcionan bien, y cómo reproducirlo tú mismo en pantalla.

Hay dos herramientas:

| Script | Qué hace | Abre ventana |
|---|---|---|
| `captura_grafica.py` | Corre los scripts en **headless** y guarda GIFs + métricas | No |
| `ver_en_vivo.py` | Abre el **visor real** en tu pantalla, escenario por escenario | Sí |

---

## 1. Captura automática (para el informe)

```bash
DISPLAY=:1 MUJOCO_GL=glfw conda run -n r1mujoco \
    python test/robustez_r1/captura_grafica.py
```

Sustituye el visor por un renderer offscreen (no abre ventana), corre el código
**real** de cada script y guarda en `evidencia/grafica/`:

- un **GIF** por corrida (`animacion.gif`) + primer/último frame,
- la **altura de la pelvis** paso a paso (`altura_pelvis.json`) — si baja de
  0.45 m se marca como caída,
- el **traceback** de cada crash (`*_TRACEBACK.txt`),
- el resumen en [`evidencia/grafica/informe_grafica.md`](evidencia/grafica/informe_grafica.md).

### Resultado de la última corrida

**Runtime — quieto (valida) vs. caminando a vx=0.9 (estres):**

| Corrida | Modo | z inicial | z mínima | Veredicto |
|---|---|---|---|---|
| `estabilidad_valida` | quieto | 0.74 | 0.74 | se mantuvo de pie |
| `camina_valida` | quieto | 0.74 | 0.739 | se mantuvo de pie |
| `camina_estres` | caminando | 0.74 | 0.697 | **camina estable** |
| `isaac_valida` | quieto | 0.74 | 0.739 | se mantuvo de pie |
| `isaac_estres` | caminando | 0.74 | 0.698 | **camina estable** |
| `banda_r1_valida` | quieto | 0.74 | 0.712 | se mantuvo de pie |
| `banda_r1_estres` | caminando | 0.74 | **0.073** | **SE CAE** 🔴 |

> **Hallazgo visual:** los scripts con la política nativa `r1_v2`
> (`play_r1_camina_brazos`, `play_r1_isaac`) caminan estables. En cambio
> `banda_r1`, que aplica la política **AMO entrenada para el G1** sobre el R1,
> al recibir el comando de caminar **se desploma** (la pelvis cae de 0.74 m a
> 0.07 m). Ver `evidencia/grafica/banda_r1_estres/animacion.gif`.

**Fixtures rotos — crash al cargar (esperado):**

| Corrida | Defecto | Excepción |
|---|---|---|
| `estabilidad_xml_roto` | malla STL inexistente | `ValueError: Error opening file … NO_EXISTE_…` |
| `camina_politica_rota` | `.pt` truncado | `RuntimeError: PytorchStreamReader … failed finding central directory` |
| `isaac_xml_roto` | XML sin cerrar | `ValueError: XML parse error 15` |
| `banda_r1_stats_rotos` | stats sin `input_std` | `KeyError: 'input_std'` |

---

## 2. Verlo en vivo en tu pantalla

`ver_en_vivo.py` abre el visor MuJoCo real. Ajusta `DISPLAY` al de tu sesión
gráfica (normalmente `:0` o `:1`).

**Baseline (funciona) — usa las flechas para caminar:**

```bash
DISPLAY=:1 conda run -n r1mujoco python test/robustez_r1/ver_en_vivo.py camina
```

**El fallo estrella: banda_r1 se cae solo al caminar** (inyecta el comando y lo
ves desplomarse):

```bash
DISPLAY=:1 conda run -n r1mujoco python test/robustez_r1/ver_en_vivo.py banda_r1_cae
```

**Crashes con fixtures rotos** (revientan en la terminal, no llega a abrir
ventana):

```bash
DISPLAY=:1 conda run -n r1mujoco python test/robustez_r1/ver_en_vivo.py camina_roto
DISPLAY=:1 conda run -n r1mujoco python test/robustez_r1/ver_en_vivo.py estabilidad_roto
DISPLAY=:1 conda run -n r1mujoco python test/robustez_r1/ver_en_vivo.py banda_r1_roto
```

Escenarios completos: `estabilidad`, `camina`, `isaac`, `banda_r1`,
`banda_r1_cae`, `estabilidad_roto`, `camina_roto`, `banda_r1_roto`. Ejecuta el
script sin argumentos para ver la lista. Cierra la ventana (ESC) para terminar.

---

## 3. Lo mismo para el G1

```bash
DISPLAY=:1 MUJOCO_GL=glfw conda run -n r1mujoco \
    python test/robustez_r1/captura_grafica_g1.py
```

Los tres scripts del G1 arrancan distinto y se manejan por separado. Salida en
[`evidencia/grafica_g1/`](evidencia/grafica_g1/informe_g1.md).

| Corrida | Script | Resultado |
|---|---|---|
| `g1_stable_valida` | play_amo_stable (quieto) | se mantiene de pie (z≈0.76) |
| `g1_stable_camina` | play_amo_stable (vx=0.5) | **camina estable** |
| `g1_stable_xml_roto` | play_amo_stable + XML roto | crash `XML parse error 15` |
| `g1_banda_valida` | banda_v2_1 (equilibrio) | **CRASH real** 🔴 `ValueError: shapes (8,) (10,)` |
| `g1_banda_xml_roto` | banda_v2_1 + XML roto | crash `XML parse error 15` |
| `g1_play_amo_cuda` | play_amo (política AMO) | **no arranca sin GPU** `Torch not compiled with CUDA enabled` |

> **Hallazgos del G1:**
> - `play_amo_stable` (marcha programada, sin red) es el único que camina bien
>   sin GPU.
> - **`banda_v2_1` está roto hoy**: en cada corrida revienta al mezclar la pose
>   de brazos — `pd_target[15:]` son 8 valores pero `arm_ctrl.target_pose` tiene
>   10 ([banda_v2_1.py:268](../../scripts/g1/banda_v2_1.py)). No es un fixture
>   inducido: es un bug del propio script. Traceback en
>   `evidencia/grafica_g1/g1_banda_valida_TRACEBACK.txt`.
> - `play_amo` (la política AMO de verdad) **exige CUDA**; en esta máquina sin
>   GPU no arranca.

### Verlo en vivo (G1)

```bash
# marcha programada (funciona; W/S/A/D para moverlo) — necesita mujoco_viewer
DISPLAY=:1 conda run -n robotica  python test/robustez_r1/ver_en_vivo.py g1_camina

# banda_v2_1: veras el crash del bug de brazos en la terminal
DISPLAY=:1 conda run -n r1mujoco python test/robustez_r1/ver_en_vivo.py g1_banda

# play_amo: solo arranca en una maquina con GPU CUDA
DISPLAY=:1 conda run -n r1mujoco python test/robustez_r1/ver_en_vivo.py g1_play_amo
```

> `g1_camina` y `g1_play_amo` importan `mujoco_viewer`, que **no** está en el
> env `r1mujoco` (sí en `robotica`). `g1_banda` usa `mujoco.viewer` y corre en
> `r1mujoco`.

---

## Notas

- Se necesita un display con GL. Aquí se usó `:1`; el render offscreen y el visor
  real funcionan sobre ese contexto.
- `MUJOCO_GL=egl` no funcionó en esta máquina (error de EGL); por eso se usa
  `glfw` sobre el display `:1`.
- Nada de esto toca `assets/` ni `policies/` del repo: los scripts se apuntan a
  los fixtures rotos de `fixtures/` en tiempo de carga.
