# Pruebas

Hay **dos baterías**, con propósitos distintos. Ninguna es `pytest`: son arneses
que corren el código real del proyecto y guardan evidencia.

| Batería | Qué mide | Salida |
|---|---|---|
| [`test/robustez_r1/`](../test/robustez_r1/README.md) | qué pasa cuando la entrada es mala (R1 y G1) | tablas, GIFs, tracebacks |
| [`test/G1/evidencia_g1/`](../test/G1/evidencia_g1/INFORME.md) | cómo se comporta el G1 y por qué se cae | vídeos, gráficas, telemetría |

> Los fixtures de `test/robustez_r1/fixtures/` están **rotos a propósito**. No los
> uses en una simulación real. Las claves de API que aparecen son ficticias
> (`sk-proj-FAKE-…`).

---

## 1. Robustez — qué pasa con entradas malas

Se alimenta el código real con documentos deliberadamente defectuosos y se
registra, caso por caso, cómo responde el sistema. **El objetivo no es que todo
pase**, sino documentar el comportamiento ante entradas malas.

```bash
conda run -n r1mujoco python test/robustez_r1/arnes_robustez.py
```

El arnés importa las mismas funciones que usan los scripts en producción — no
simula los resultados.

| Bloque | Código real ejercitado | Fixtures |
|---|---|---|
| 1. Credenciales `.env` | parser inline de `play_r1_ia.py` | 11 `.env` mal formados |
| 2. Estado del robot JSON | `BackendSimulado._narrar_estado` | 13 snapshots |
| 3. Argumentos de herramienta | `CajaDeHerramientas.despachar` | 18 llamadas del LLM |
| 4. Escenas MuJoCo | `mujoco.MjModel.from_xml_path` | 9 por robot |
| 5. Políticas TorchScript | `torch.jit.load` | 5 por robot |
| 6. Normalización del adaptador | `torch.load` + validación de dims/NaN | 6 `stats_*.pt` |

Cada caso se clasifica en **aceptado** / **rechazado_controlado** /
**excepcion_no_controlada**. Los `00_valido` son controles: entradas correctas que
deben pasar, para demostrar que un fallo viene del defecto inyectado y no del
propio arnés.

**Resultado de la última corrida: 76 casos — 42 rechazos controlados, 26
aceptados, 8 excepciones no controladas.** Esas 8 son los hallazgos a corregir:

1. **El parser de `.env` revienta con bytes no UTF-8** → `UnicodeDecodeError`.
   `Path.read_text()` sin `encoding` ni `errors` falla si el `.env` trae BOM
   latino, bytes binarios o está en Latin-1. Un `.env` mal codificado impide
   arrancar `play_r1_ia.py`.
   *Arreglo:* leerlo con `encoding="utf-8", errors="replace"`.

2. **`_narrar_estado` confía en el esquema del JSON** → `KeyError` si falta
   `posicion_m`, `ValueError` si trae 2 o 4 elementos, `TypeError` si un campo
   llega `null` o como string, `TypeError` si la raíz es una lista.
   *Arreglo:* validar presencia, tipo y longitud antes de formatear, degradando al
   texto crudo (que es justo lo que ya hace con un JSON no parseable).

El detalle completo está en
[`evidencia/informe.md`](../test/robustez_r1/evidencia/informe.md) y los datos
crudos en `evidencia/resultados.json`.

---

## 2. Evidencia gráfica — verlo caerse

Complementa lo anterior con evidencia **visual**: qué se ve cuando funciona y qué
se ve cuando no. Detalle en
[`test/robustez_r1/GRAFICA.md`](../test/robustez_r1/GRAFICA.md).

### Captura automática (headless, genera GIFs y métricas)

```bash
DISPLAY=:1 MUJOCO_GL=glfw conda run -n r1mujoco python test/robustez_r1/captura_grafica.py
```

```bash
DISPLAY=:1 MUJOCO_GL=glfw conda run -n r1mujoco python test/robustez_r1/captura_grafica_g1.py
```

Sustituye el visor por un renderer offscreen — no abre ventana — y corre el código
real de cada script. Guarda un GIF por corrida, la altura de la pelvis paso a paso
(por debajo de **0.45 m** se marca como caída) y el traceback de cada crash.

### Verlo en vivo en pantalla

```bash
DISPLAY=:1 conda run -n r1mujoco python test/robustez_r1/ver_en_vivo.py banda_r1_cae
```

Escenarios: `estabilidad`, `camina`, `isaac`, `banda_r1`, `banda_r1_cae`,
`estabilidad_roto`, `camina_roto`, `banda_r1_roto`, `g1_camina`, `g1_banda`,
`g1_play_amo`. Sin argumentos lista todos. `ESC` para cerrar.

> `g1_camina` y `g1_play_amo` importan `mujoco_viewer`, que solo está en el entorno
> `robotica`. Cámbialo con `conda run -n robotica` para esos dos.

### Hallazgos

| Corrida | Modo | z inicial | z mínima | Veredicto |
|---|---|---|---|---|
| `estabilidad_valida` | quieto | 0.74 | 0.74 | de pie |
| `camina_estres` | vx=0.9 | 0.74 | 0.697 | **camina estable** |
| `isaac_estres` | vx=0.9 | 0.74 | 0.698 | **camina estable** |
| `banda_r1_estres` | vx=0.9 | 0.74 | **0.073** | **SE CAE** 🔴 |

- Los scripts con la **política nativa `r1_v2`** (`play_r1_isaac`,
  `play_r1_camina_brazos`) caminan estables.
- **`banda_r1` se desploma al caminar**: aplica la política AMO *entrenada para el
  G1* sobre el R1. La pelvis cae de 0.74 m a 0.07 m.
- **`banda_v2_1` del G1 está roto hoy**: revienta al mezclar la pose de brazos —
  `pd_target[15:]` son 8 valores pero `arm_ctrl.target_pose` tiene 10
  ([banda_v2_1.py:268](../scripts/g1/banda_v2_1.py)). No es un fixture inducido, es
  un bug del propio script.
- **`play_amo` exige CUDA** y no arranca en una máquina sin GPU.

---

## 3. Evidencia del G1 — por qué se cae

14 escenarios renderizados del G1: los que funcionan y los degradados a propósito
para documentar cada modo de fallo por separado. Informe completo en
[`test/G1/evidencia_g1/INFORME.md`](../test/G1/evidencia_g1/INFORME.md).

```bash
conda run -n robotica python test/G1/evidencia_g1/render_g1.py          # todos
conda run -n robotica python test/G1/evidencia_g1/render_g1.py 01 06 09 # por prefijo
conda run -n robotica python test/G1/evidencia_g1/render_g1.py --lista
```

```bash
conda run -n robotica python test/G1/evidencia_g1/resumen_g1.py     # gráficas y montajes
conda run -n robotica python test/G1/evidencia_g1/build_pagina.py   # pagina.html autocontenida
```

> `resumen_g1.py` necesita `matplotlib`, que **no** está instalado hoy en
> `robotica`. Añádelo con `conda run -n robotica pip install matplotlib` antes de
> ejecutarlo. `render_g1.py` y `build_pagina.py` sí corren tal cual.

Genera `media/*.mp4` y `*.gif` (960×540 @ 50 fps con HUD), `frames/*.png`,
`graficas/*.png` y `data/*.npz` con la telemetría cruda (t, z, roll, pitch, x, y,
‖τ‖, ‖qvel‖).

### Los dos hallazgos de fondo

1. **El controlador PD no tiene realimentación de equilibrio.**
   `play_amo_stable.py` persigue `BASE_POSE` articulación por articulación; nada lee
   el roll/pitch ni la velocidad de la base, así que el robot se inclina
   progresivamente y cae siempre alrededor de los 2.8 s. Subir el límite de par de
   60 a 200 Nm no cambia nada (probado): no es saturación, es ausencia de lazo de
   balance en tobillo y cadera.

2. **El vector de observación de `play_amo.py` no coincide con el que espera la
   red.** La traza de `amo_jit.pt` segmenta su entrada como `obs_prop = obs[0:93]`,
   `obs_demo = obs[198:215]`, `obs_priv_explicit = obs[215:218]` e historial de
   `[-1, 10, 93]`. `play_amo.py` construye en cambio
   `qpos(30) + qvel(29) + adapter(15) + last_action(15) + phase(4)` con historial de
   11 y 20 comandos al final. Los comandos nunca llegan al tramo `obs_demo` que la
   red realmente lee. Por eso la política entrega pares incoherentes y el robot
   colapsa en 0.34 s.

---

## Cómo añadir un caso

**A la batería de robustez:** deja el fixture en la carpeta que le toque
(`fixtures/env/`, `estado_json/`, `tool_args/`, `escenas/{r1,g1}/`,
`politicas/{r1,g1}/`) con el defecto en el nombre del archivo, y vuelve a correr
`arnes_robustez.py`. Los de `tool_args` llevan además un campo `defecto` que lo
describe en texto.

**A la evidencia del G1:** añade el escenario a la lista de `render_g1.py`
siguiendo el patrón numerado, y vuelve a correr `render_g1.py` + `resumen_g1.py`.
