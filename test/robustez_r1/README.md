# Pruebas de robustez R1 / G1 — evidencia de manejo de fallos

Este directorio contiene una batería de **pruebas de robustez** para los flujos
de R1 y G1: se alimenta el código real del proyecto con documentos
**deliberadamente defectuosos** (credenciales mal escritas, JSON malformado,
argumentos inválidos del LLM, escenas MuJoCo corruptas y políticas `.pt` rotas)
y se registra, caso por caso, cómo responde el sistema.

El objetivo no es que todo pase, sino **documentar el comportamiento ante
entradas malas**: qué se valida, qué se rechaza de forma controlada y qué
todavía revienta con una excepción cruda.

> 📺 **Evidencia gráfica (visor MuJoCo), R1 y G1:** para ver los scripts
> corriendo — `banda_r1` cayéndose al caminar, el bug real de `banda_v2_1` del
> G1, `play_amo` exigiendo GPU y los crashes con fixtures rotos — mira
> [GRAFICA.md](GRAFICA.md). Ahí están los GIFs y los comandos para reproducirlo
> en vivo en tu pantalla.

> ⚠️ Los archivos de `fixtures/` están rotos a propósito. No los uses en una
> simulación real. Las claves de API que aparecen son ficticias (`sk-proj-FAKE-…`).

---

## Cómo se generó la evidencia

```bash
conda run -n r1mujoco python test/robustez_r1/arnes_robustez.py
```

El arnés (`arnes_robustez.py`) **importa las mismas funciones que usan los
scripts en producción** — no simula los resultados:

| Bloque | Código real ejercitado | Fixtures |
|---|---|---|
| 1. Credenciales `.env` | parser inline de `scripts/r1/play_r1_ia.py` | `fixtures/env/` |
| 2. Estado del robot JSON | `BackendSimulado._narrar_estado` (play_r1_ia.py) | `fixtures/estado_json/` |
| 3. Argumentos de herramienta | `CajaDeHerramientas.despachar` (play_r1_ia.py) | `fixtures/tool_args/` |
| 4. Escenas MuJoCo | `mujoco.MjModel.from_xml_path` (usado por R1 y G1) | `fixtures/escenas/{r1,g1}/` |
| 5. Políticas TorchScript | `torch.jit.load` (usado por R1 y G1) | `fixtures/politicas/{r1,g1}/` |
| 6. Normalización adaptador | `torch.load` + validación de dims/NaN | `fixtures/politicas/stats_*.pt` |

Cada caso se clasifica en:

- **aceptado** — el código tomó la entrada como válida (a veces es correcto,
  a veces revela una validación que falta).
- **rechazado_controlado** — se detectó el problema y se devolvió un error claro
  o una degradación limpia, sin tumbar el proceso.
- **excepcion_no_controlada** — la entrada mala provocó una excepción cruda que
  el flujo actual no atrapa. **Estos son los hallazgos a corregir.**

---

## Resultados (última ejecución)

Ver la tabla completa en [`evidencia/informe.md`](evidencia/informe.md) y los
datos crudos en [`evidencia/resultados.json`](evidencia/resultados.json). El log
de consola queda en [`evidencia/ejecucion.log`](evidencia/ejecucion.log).

**76 casos** ejecutados. Resumen: 42 rechazos controlados, 26 aceptados,
**8 excepciones no controladas** (hallazgos).

### Hallazgos abiertos (excepciones no controladas)

1. **Parser `.env` con bytes no UTF-8** (`env/08_binario.env`,
   `env/11_latin1.env`) → `UnicodeDecodeError`. `Path.read_text()` sin `encoding`
   ni `errors` revienta si el `.env` trae un BOM latino, bytes binarios o está en
   Latin-1. Un `.env` mal codificado impide arrancar `play_r1_ia.py`.

2. **`_narrar_estado` confía en el esquema del JSON** (`estado_json/03,04,05,07,10,11`):
   - falta la clave `posicion_m` → `KeyError`
   - `posicion_m` con 2 o 4 elementos → `ValueError` al desempaquetar
   - `velocidad_ms`/campos como `null` o string → `TypeError`/`ValueError` al formatear
   - la raíz es una lista en vez de un objeto → `TypeError`

   El JSON lo produce el propio `consultar_estado`, así que hoy no ocurre en
   condiciones normales; pero si el snapshot cambia de forma o llega parcial, la
   narración del estado tumba el turno del agente.

Sugerencia de corrección para ambos: leer el `.env` con
`encoding="utf-8", errors="replace"` (o `ignore`), y en `_narrar_estado` validar
presencia/tipo/longitud de las claves antes de formatear, degradando al texto
crudo (que es justo lo que ya hace ante un JSON no parseable).

---

## Estructura

```
test/robustez_r1/
├── README.md                 # este archivo
├── arnes_robustez.py         # ejecuta el código real contra los fixtures
├── fixtures/
│   ├── env/                  # 11 .env mal formados
│   ├── estado_json/          # 13 snapshots de estado JSON
│   ├── tool_args/            # 18 llamadas de herramienta del LLM
│   ├── escenas/{r1,g1}/      # 9 escenas MuJoCo cada uno (incluye control 00_valido)
│   └── politicas/{r1,g1}/    # 5 .pt corruptos cada uno + stats_*.pt
└── evidencia/
    ├── informe.md            # tabla legible caso por caso
    ├── resultados.json       # datos crudos
    └── ejecucion.log         # salida de consola de la corrida
```

Cada fixture lleva en su nombre el defecto que inyecta, y los de `tool_args`
incluyen un campo `defecto` que lo describe. Los `00_valido` / `stats_00_valido`
son **controles**: entradas correctas que deben pasar, para demostrar que un
fallo posterior viene del defecto inyectado y no del propio arnés.
