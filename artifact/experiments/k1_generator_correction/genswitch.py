"""K1 · seleção do gerador para a ponte E10, sem modificar o código E10.

`use("v7")` restaura o gerador original (o mesmo módulo carregado por
experiments.e10_eaml.scenarios); `use("v7_1")` instala a versão corrigida
`pipeline_completo/synthetic_driver_risk_v7_1.py`. A troca altera apenas o
módulo apontado por `scenarios._gen` e o dicionário `scenarios.RECIPES_ALL`
(reconstruído do mesmo módulo; as receitas são idênticas nas duas versões).
Listas de categorias, mix, personas, ruído e sementes não mudam.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from experiments.e10_eaml import scenarios

_BACKEND = Path(__file__).resolve().parents[2]
PATHS = {"v7": _BACKEND / "pipeline_completo" / "synthetic_driver_risk_v7.py",
         "v7_1": _BACKEND / "pipeline_completo" / "synthetic_driver_risk_v7_1.py"}
EXPECTED_SHA16 = {"v7": "efede62fe3aceed8"}
_ORIG = scenarios._gen
_MODS = {"v7": _ORIG}


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load(version: str):
    if version not in _MODS:
        spec = importlib.util.spec_from_file_location(f"sdr_{version}", PATHS[version])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"sdr_{version}"] = mod
        spec.loader.exec_module(mod)
        _MODS[version] = mod
    return _MODS[version]


def use(version: str):
    assert sha16(PATHS["v7"]) == EXPECTED_SHA16["v7"], "gerador original alterado!"
    mod = _load(version)
    scenarios._gen = mod
    scenarios.RECIPES_ALL.clear()
    scenarios.RECIPES_ALL.update({**mod.RECIPES_TRAIN, **mod.RECIPES_BORDERLINE})
    return mod


def current() -> str:
    return next(k for k, m in _MODS.items() if m is scenarios._gen)
