# Evidencia grafica — scripts G1 en runtime

- **Generado:** 2026-08-18 19:02:51 -0500
- **Corridas:** 6

Cada corrida ejecuta el script REAL del G1 en headless.

## Runtime

| Corrida | Modo | Frames | z ini | z min | z fin | Veredicto/Estado | GIF |
|---|---|---|---|---|---|---|---|
| `g1_stable_valida` | valida | 40 | 0.793 | 0.753 | 0.759 | camina/se mantiene de pie | evidencia/grafica_g1/g1_stable_valida/animacion.gif |
| `g1_stable_camina` | estres | 40 | 0.793 | 0.753 | 0.759 | camina/se mantiene de pie | evidencia/grafica_g1/g1_stable_camina/animacion.gif |

## Crashes / requisitos

| Corrida | Estado | Error |
|---|---|---|
| `g1_stable_xml_roto` | crasheo_esperado | ValueError: XML parse error 15: Error=XML_ERROR_PARSING ErrorID=15 (0xf) Line number=3  |
| `g1_banda_valida` | crasheo | ValueError: operands could not be broadcast together with shapes (8,) (10,)  |
| `g1_banda_xml_roto` | crasheo_esperado | ValueError: XML parse error 15: Error=XML_ERROR_PARSING ErrorID=15 (0xf) Line number=3  |
| `g1_play_amo_cuda` | no_ejecutable | AssertionError: Torch not compiled with CUDA enabled |
