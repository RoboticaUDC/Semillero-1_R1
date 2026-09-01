# Evidencia grafica — scripts R1 en runtime

- **Generado:** 2026-08-18 18:47:06 -0500
- **Corridas:** 11

Cada corrida ejecuta el script REAL en headless (visor sustituido por un renderer offscreen). Las validas guardan un GIF y la altura de la pelvis; las de fixture roto capturan el crash.

## Corridas de runtime (valida = quieto, estres = caminando)

| Corrida | Modo | Frames | z ini | z min | z fin | Veredicto | GIF |
|---|---|---|---|---|---|---|---|
| `estabilidad_valida` | valida | 60 | 0.74 | 0.74 | 0.743 | se mantuvo de pie | evidencia/grafica/estabilidad_valida/animacion.gif |
| `camina_valida` | valida | 60 | 0.74 | 0.739 | 0.74 | se mantuvo de pie | evidencia/grafica/camina_valida/animacion.gif |
| `camina_estres` | estres | 180 | 0.74 | 0.697 | 0.725 | se mantuvo de pie | evidencia/grafica/camina_estres/animacion.gif |
| `isaac_valida` | valida | 60 | 0.74 | 0.739 | 0.74 | se mantuvo de pie | evidencia/grafica/isaac_valida/animacion.gif |
| `isaac_estres` | estres | 180 | 0.74 | 0.698 | 0.729 | se mantuvo de pie | evidencia/grafica/isaac_estres/animacion.gif |
| `banda_r1_valida` | valida | 60 | 0.74 | 0.712 | 0.712 | se mantuvo de pie | evidencia/grafica/banda_r1_valida/animacion.gif |
| `banda_r1_estres` | estres | 180 | 0.74 | 0.073 | 0.086 | SE CAYO / inestable | evidencia/grafica/banda_r1_estres/animacion.gif |

## Corridas con fixture roto (crash esperado)

| Corrida | Defecto | Resultado | Error |
|---|---|---|---|
| `estabilidad_xml_roto` | escena con malla STL que no existe | crasheo_esperado | ValueError: Error: Error opening file '../../../../../assets/r1_description/meshes/NO_EXISTE_pelv... |
| `camina_politica_rota` | politica TorchScript truncada a la mitad | crasheo_esperado | RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory. T... |
| `isaac_xml_roto` | XML de escena sin etiqueta de cierre | crasheo_esperado | ValueError: XML parse error 15:
Error=XML_ERROR_PARSING ErrorID=15 (0xf) Line number=3
 |
| `banda_r1_stats_rotos` | adapter_norm_stats sin la clave input_std | crasheo_esperado | KeyError: 'input_std' |
