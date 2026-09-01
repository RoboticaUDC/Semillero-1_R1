# Evidencia de simulación — Unitree G1 (proyecto AMO)

Generado el 2026-08-18 con `render_g1.py` + `resumen_g1.py`.
Todo el material corresponde **solo al G1** (`g1.xml`, 23 DoF, `amo_jit.pt`, `adapter_jit.pt`,
`play_amo.py`, `play_amo_stable.py`, `banda*.py`, `ArmController.py`). No se incluye nada del R1.

## Qué hay en la carpeta

| Carpeta | Contenido |
|---|---|
| `media/*.mp4` | 14 escenarios en 960×540 @ 50 fps con HUD (tiempo, altura de pelvis, roll/pitch, comando, par) |
| `media/*.gif` | Los mismos escenarios en GIF 560 px @ 20 fps |
| `media/gif_ligero/*.gif` | Versiones compactas 440 px @ 12 fps (3–9 MB) para pegar en documentos |
| `media/comparativa_2x2.*` | Cuatro modos de fallo en paralelo (policy AMO, ruido, latencia, pose inicial) |
| `frames/*.png` | Fotograma de inicio, medio y final de cada escenario |
| `graficas/` | Altura de pelvis, tiempo de supervivencia, inclinación de base, mosaico de fallos |
| `data/*.npz` | Telemetría cruda (t, z, roll, pitch, x, y, ‖τ‖, ‖qvel‖) |

## Resultados

| # | Escenario | Archivo de origen | Resultado |
|---|---|---|---|
| 01 | De pie con PD puro hacia `BASE_POSE` | `play_amo_stable.py` | Cae a 2.80 s |
| 02 | Marcha con patrón cíclico (vx = 0.35 m/s) | `play_amo_stable.py` | Cae a 2.82 s |
| 03 | Control de altura de torso (Z/X) | `play_amo_stable.py` | Cae a 2.84 s |
| 04 | Secuencias de brazos sobre postura PD | `ArmController.py` + `banda_v2.py` | Cae a 2.76 s |
| 05 | Policy AMO real (policy + adapter) | `play_amo.py` | Cae a 0.34 s |
| 06 | FALLO inducido: adapter desconectado | `play_amo.py` degradado | Cae a 0.38 s |
| 07 | FALLO inducido: ruido σ = 22 Nm en los pares | `play_amo_stable.py` degradado | Cae a 0.94 s |
| 08 | FALLO inducido: ganancias PD al 20 % | `play_amo_stable.py` degradado | Cae a 1.08 s |
| 09 | FALLO inducido: 45 ms de latencia de estado | `play_amo_stable.py` degradado | Cae a 0.98 s |
| 10 | FALLO inducido: pose inicial sin flexión (`qpos = 0`) | `banda.py` | Cae a 1.78 s |
| 11 | FALLO inducido: arranque con 16° de pitch | `play_amo_stable.py` degradado | Cae a 0.94 s |
| 12 | FALLO inducido: soltado desde 1.15 m | `play_amo_stable.py` degradado | Cae a 1.26 s |
| 13 | FALLO inducido: escala de acción ×3 | `play_amo.py` degradado | Cae a 0.26 s |
| 14 | Poses de brazos con pelvis anclada | `ArmController.py` | Estable 16 s (banco de pruebas) |

Los escenarios 01–05 son el comportamiento **tal cual está el código hoy**, sin degradar.
Los 06–13 son degradaciones deliberadas para documentar cada modo de fallo por separado.
El 14 fija la pelvis para poder validar las 7 poses de brazos sin que la caída las tape.

## Hallazgos del proceso

1. **El controlador PD no tiene realimentación de equilibrio.** `play_amo_stable.py` solo persigue
   `BASE_POSE` articulación por articulación; nada lee el roll/pitch ni la velocidad de la base, así
   que el robot se inclina progresivamente y cae siempre alrededor de los 2.8 s. Subir el límite de
   par de 60 a 200 Nm no cambia nada (probado): el problema no es saturación, es ausencia de lazo de
   balance en tobillo/cadera.

2. **El vector de observación de `play_amo.py` no coincide con el que espera la red.** La traza de
   `amo_jit.pt` segmenta internamente su entrada así:
   - repite las últimas 372 columnas al frente (4 × 93) e inserta 105 ceros en la columna 465,
   - `obs_prop = obs[0:93]`, `obs_demo = obs[198:215]` (17 valores), `obs_priv_explicit = obs[215:218]`,
   - historial = últimas 930 columnas, remodeladas a `[-1, 10, 93]`.

   `play_amo.py` construye en cambio `qpos(30) + qvel(29) + adapter(15) + last_action(15) + phase(4)`
   con un historial de 11 y 20 comandos al final. Los comandos nunca llegan al tramo `obs_demo` que
   la red realmente lee, y el bloque propioceptivo tampoco está en el orden esperado. Por eso la
   policy entrega pares incoherentes y el robot colapsa en 0.34 s.

3. **`amo_jit.pt` está trazado con `cuda:0` fijo dentro del grafo**, así que no se puede ejecutar en
   CPU. Además el entorno `r1deploy` (torch 2.3.1+cu121) no tiene kernels para la RTX 5060 Ti
   (`no kernel image is available`); la evidencia se generó con torch 2.7.0+cu128 en `env_isaaclab`,
   donde sí se instaló MuJoCo.

4. **`ArmController.py` está indexado para el R1, no para el G1.** Usa 10 joints de brazo en
   `qpos[21:31]` y escribe en `ctrl[14:24]`, mientras que el G1 tiene 8 joints de brazo en
   `qpos[22:30]` / `ctrl[15:23]`; en el G1 esa escritura pisa el `waist_pitch`. El arnés de evidencia
   reimplementa las poses con la indexación correcta del G1.

5. **Modos de fallo ordenados por severidad** (tiempo hasta la caída): escala de acción ×3 (0.26 s) <
   policy AMO tal cual (0.34 s) < adapter desconectado (0.38 s) < arranque inclinado (0.94 s) ≈ ruido
   de pares (0.94 s) < latencia 45 ms (0.98 s) < ganancias bajas (1.08 s) < caída libre (1.26 s) <
   pose inicial `qpos=0` (1.78 s) < PD nominal (2.80 s).

## Cómo reproducir

```bash
/home/udc/miniconda3/envs/env_isaaclab/bin/python evidencia_g1/render_g1.py          # todos
/home/udc/miniconda3/envs/env_isaaclab/bin/python evidencia_g1/render_g1.py 05 09    # algunos
/home/udc/miniconda3/envs/env_isaaclab/bin/python evidencia_g1/resumen_g1.py         # gráficas
```

Los escenarios se definen en la lista `ESCENARIOS` de `render_g1.py`; los fallos se inyectan con el
diccionario `fallos` (`ruido_par`, `kp_escala`, `latencia_ms`, `pose_inicial`, `adapter_off`,
`escala_accion`).
