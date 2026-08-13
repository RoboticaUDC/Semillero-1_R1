# legacy/

Versiones anteriores que quedaron superadas por otras. **No se borraron**: siguen
funcionando (se les actualizaron las rutas al nuevo layout) y sirven de
referencia cuando algo se rompe en la version nueva.

| Archivo | Lo reemplaza | Por que se movio aqui |
|---|---|---|
| `teleop_r1.py` | `scripts/teleop/teleop_manos.py` | primera version: calculaba los angulos con formulas y suponia la orientacion de los frames del hombro. El codo doblaba al reves. |
| `teleop_r1_ik.py` | idem | introdujo la IK numerica, pero se quedaba atrapada en minimos locales. |
| `teleop_r1_ik2.py` | idem | IK con reintentos desde varias semillas. Sigue leyendo el estado por indice fijo, asi que se rompe con el XML de manos. |
| `banda.py` | `scripts/g1/banda_v2_1.py` | G1 con LIDAR + bandas + gripper. La estabilidad inicial no funciona. |
| `banda_v2.py` | idem | igual que `banda.py` pero con `ArmController`. Superado por `banda_v2_1.py`, que es mucho mas corto. |
| `play_R1_isaac.py` | `scripts/r1/play_r1_isaac.py` | era un duplicado byte a byte, solo cambiaba la mayuscula del nombre. |
| `arm_controller_4dof.py` | `amo/control/arm_controller.py` | version vieja con 4 joints por brazo (sin `wrist_roll`). |
| `r1.xml.bak` | `assets/r1.xml` | respaldo del modelo antes de agregar los brazos completos. |

Para ejecutarlos, igual que los de `scripts/`:

```bash
python legacy/banda.py
```
