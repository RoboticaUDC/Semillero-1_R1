"""Rutas del repositorio, resueltas a partir de la ubicacion de este archivo.

Antes cada script usaba rutas relativas ("r1.xml", "amo_jit.pt"), asi que solo
funcionaban si los ejecutabas parado en la raiz del repo. Con esto los scripts
corren desde cualquier directorio.

Uso tipico:

    from amo.paths import scene, policy

    model = mujoco.MjModel.from_xml_path(scene("r1.xml"))
    net   = torch.jit.load(policy("r1_policy_v2.pt"))
"""

from pathlib import Path

# amo/paths.py -> amo/ -> raiz del repo
ROOT = Path(__file__).resolve().parents[1]

ASSETS = ROOT / "assets"        # XML de MuJoCo + mallas + descripciones URDF/USD
POLICIES = ROOT / "policies"    # redes exportadas a TorchScript (.pt)
THIRD_PARTY = ROOT / "third_party"  # clones externos (IsaacLab, unitree_*, brainco)

# XML disponibles (nombre corto -> archivo)
SCENES = {
    "g1": "g1.xml",              # Unitree G1, el del paper AMO
    "r1": "r1.xml",              # R1, 24 DOF, sin manos
    "r1_manos": "r1_manos.xml",  # R1 + manos Revo2 (22 DOF extra de dedos)
}

# Politicas disponibles (nombre corto -> archivo)
POLICY_FILES = {
    "amo": "amo_jit.pt",                     # politica AMO del paper (G1)
    "adapter": "adapter_jit.pt",             # adaptador de AMO (G1)
    "adapter_stats": "adapter_norm_stats.pt",  # normalizacion del adaptador
    "r1": "r1_policy.pt",                    # R1 nativa, primera version
    "r1_v2": "r1_policy_v2.pt",              # R1 nativa, la buena
}


def _resolve(base: Path, name: str, alias: dict, que: str) -> str:
    """Acepta un alias corto, un nombre de archivo o una ruta absoluta."""
    p = Path(name)
    if p.is_absolute():
        candidato = p
    else:
        candidato = base / alias.get(name, name)

    if not candidato.exists():
        opciones = ", ".join(sorted(alias))
        raise FileNotFoundError(
            f"No existe {que} '{name}' (buscado en {candidato}).\n"
            f"Alias validos: {opciones}"
        )
    return str(candidato)


def scene(name: str) -> str:
    """Ruta absoluta a un XML de MuJoCo. Acepta 'r1' o 'r1.xml'."""
    return _resolve(ASSETS, name, SCENES, "la escena")


def policy(name: str) -> str:
    """Ruta absoluta a una politica .pt. Acepta 'r1_v2' o 'r1_policy_v2.pt'."""
    return _resolve(POLICIES, name, POLICY_FILES, "la politica")
