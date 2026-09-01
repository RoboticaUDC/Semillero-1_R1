# AMO — Control de humanoides G1 y R1

Simulación, teleoperación, control por lenguaje natural y despliegue en hardware
para dos robots humanoides:

| Robot | Qué es | DOF | Política | Estado |
|---|---|---|---|---|
| **Unitree G1** | El robot del paper [AMO (RSS 2025)](https://amo-humanoid.github.io/) | 23 | `amo_jit.pt` + `adapter_jit.pt` | código original + adaptaciones propias |
| **R1** | Robot propio, entrenado desde cero en Isaac Lab | 24 (+22 de dedos con manos Revo2) | `r1_policy_v2.pt` | línea principal de trabajo |

Sobre esa base hay cuatro capas construidas encima: gestos de brazo, teleoperación
por cámara (MediaPipe + IK), un agente LLM que traduce lenguaje natural a acciones
del robot, y voz offline (Vosk + Piper) en español.

---

## Documentación

| Documento | Para qué |
|---|---|
| [docs/INSTALACION.md](docs/INSTALACION.md) | Montar el proyecto en una máquina nueva: conda, C++, modelos de voz, Isaac Lab |
| [docs/EJECUTAR.md](docs/EJECUTAR.md) | El comando exacto de cada script, con sus teclas y argumentos |
| [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md) | Cómo está organizado, qué hace cada componente y qué datos circulan |
| [docs/PRUEBAS.md](docs/PRUEBAS.md) | Las dos baterías de pruebas, cómo correrlas y qué encontraron |
| [legacy/README.md](legacy/README.md) | Versiones superadas y qué reemplazó a qué |

---

## Arranque rápido

```bash
conda activate r1mujoco && python scripts/r1/play_r1_isaac.py
```

R1 caminando con su política nativa. Flechas para moverlo, `ESPACIO` para parar,
`ESC` para salir. Si eso funciona, el resto también.

El plato fuerte — R1 con manos, voz en español y agente LLM:

```bash
conda activate r1mujoco && python scripts/r1/play_r1_voz.py --backend openai
```

---

## Estructura del repositorio

```
AMO/
├── amo/                  paquete Python compartido (rutas, mates, control, visión)
│   ├── paths.py          resuelve assets/ y policies/ sin depender del CWD
│   ├── math_utils.py     conversiones de quaternion (convención MuJoCo w,x,y,z)
│   ├── control/          ArmController (gestos), IKBrazos, ControladorManos
│   └── vision/           MediaPipe: seguimiento de manos y de tren superior
│
├── scripts/              ejecutables — uno por experimento
│   ├── g1/               los del paper AMO (play_amo, play_amo_stable, banda_v2_1)
│   ├── r1/               los del R1 (play_r1_isaac, camina_brazos, banda_r1, ia, voz)
│   ├── teleop/           teleoperación por cámara (teleop_manos, teleop_dedos)
│   ├── tools/            utilidades (calibrar, exportar política, comprobar orden de joints)
│   └── deploy/           robot real por SDK de Unitree
│
├── cpp/                  ports a C++
│   ├── mujoco/           play_r1_isaac.cpp, teleop_r1.cpp + pose_sender.py
│   └── deploy/           r1_deploy.cpp (unitree_sdk2 + DDS, hardware real)
│
├── assets/               modelos MuJoCo (g1.xml, r1.xml, r1_manos.xml) + mallas + URDF/USD
├── policies/             redes entrenadas exportadas a TorchScript (.pt)
├── modelos_voz/          Vosk (STT) y Piper (TTS)  — no versionado, se descarga
├── test/                 dos baterías de pruebas con su evidencia
├── third_party/          clones externos (IsaacLab, unitree_*, brainco) — no versionado
├── legacy/               versiones anteriores, funcionales, guardadas como referencia
├── docs/                 esta documentación
└── bin/                  binarios compilados — no versionado
```

**Regla de oro del layout:** nada en `scripts/` importa de otro `scripts/` por ruta
relativa. Lo compartido vive en `amo/`, y cada script se hace importable con el mismo
prólogo de tres líneas (`sys.path.insert` a la raíz). Por eso todos se pueden lanzar
desde cualquier directorio.

---

## Lo que hay que tener en cuenta

**Dos entornos conda, incompatibles entre sí.** `robotica` (Python 3.11) fija
numpy 1.23 / mujoco 3.2 porque es lo que exige el código del paper; `r1mujoco`
(Python 3.10) usa numpy 2.2 / mujoco 3.10. Mezclarlos rompe uno de los dos.
El G1 va en `robotica`, el R1 en `r1mujoco`. Isaac Lab tiene el suyo,
`isaaclab_clean`.

**La API key vive en `.env` en la raíz**, nunca en el código. `.env` está en
`.gitignore`. Las variables ya exportadas en el entorno mandan sobre el `.env`.

**El teclado siempre gana.** En los scripts con LLM, `ESPACIO` cancela cualquier
cosa que haya mandado el agente. Es el freno de emergencia y está por diseño.

**El bucle de MuJoCo nunca bloquea.** Corre a 500 Hz con ~2 ms por paso. Todo lo
que tarde (LLM, cámara, IK, síntesis de voz) vive en otro hilo y se comunica por
colas o snapshots bajo lock. Un `requests.post` dentro del bucle tira al robot.

**El catálogo de acciones del LLM es cerrado.** El agente elige de una lista de
movimientos y gestos ya verificados; nunca genera ángulos de articulación. Un LLM
produciendo radianes da poses imposibles y autocolisiones.

**Antes de tocar hardware real**, lee el aviso del paper original: los fallos de
transferencia sim-a-real dañan el robot. `cpp/deploy/r1_deploy.cpp` tiene tres
puntos marcados como pendientes de verificar contra el robot físico.

---

## Créditos

La política AMO del G1, `scripts/g1/play_amo.py` y los assets del G1 vienen del
trabajo de Jialong Li, Xuxin Cheng, Tianshu Huang, Shiqi Yang, Ri-Zhao Qiu y
Xiaolong Wang (UC San Diego), publicado en RSS 2025 bajo Apache 2.0 —
[web](https://amo-humanoid.github.io/) · [arXiv](https://arxiv.org/abs/2505.03738).

```bibtex
@article{li2025amo,
  title={AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control},
  author={Li, Jialong and Cheng, Xuxin and Huang, Tianshu and Yang, Shiqi and Qiu, Rizhao and Wang, Xiaolong},
  journal={Robotics: Science and Systems 2025},
  year={2025}
}
```

Todo lo del R1 (modelo, política, teleoperación, agente LLM, voz, ports a C++ y
pruebas) es trabajo propio de este repositorio.

## Aviso

Desplegar estos modelos en hardware físico es peligroso. Se suministran para
investigación; no nos hacemos responsables de daños derivados de su despliegue.

## Licencia

Apache 2.0 — ver [LICENSE](LICENSE).
