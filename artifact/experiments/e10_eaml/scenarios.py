"""E10 · scenarios — ponte gerador v7 + personas + fluxos exógenos pareados.

Implementa a Seção 3–4 do protocolo (docs/PROTOCOLO_E10_EAML_confirmatorio.md):

* receitas do gerador `pipeline_completo/synthetic_driver_risk_v7.py`
  (N01–N03, A01–A03, L01–L03, C01–C04 abruptos, C05–C10 progressivos);
* 12 personas (`sim/driver_personas.py`) com risk_mod; observabilidade,
  responsividade e suscetibilidade à fadiga definidas AQUI (tabela fixa);
* números aleatórios comuns: TODOS os sorteios (receita por slot, ruído de
  observação, resposta do motorista) são pré-amostrados por seed e
  compartilhados pelos braços; `exogenous_hash` permite a asserção de
  igualdade exigida pelo protocolo (§8).

Nenhum braço vê rótulo de episódio nem informação futura: os braços consomem
apenas `dx` (variação do vetor de entrada, causal) e `p_obs[t]` (o que o
modelo produziria SE computasse no frame t).
"""
from __future__ import annotations

import hashlib
import importlib.util
import random as _random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

_BACKEND = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ imports
def _load_module(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_gen = _load_module("sdr_v7", "pipeline_completo/synthetic_driver_risk_v7.py")
_personas = _load_module("sd_personas", "sim/driver_personas.py")

RECIPES_ALL: Dict[str, dict] = {**_gen.RECIPES_TRAIN, **_gen.RECIPES_BORDERLINE}

CAT_NORMAL = [k for k, v in RECIPES_ALL.items() if v["categoria"] == "Normal"]
CAT_ATENCAO = [k for k, v in RECIPES_ALL.items() if v["categoria"] == "Atencao"]
CAT_ALERTA = [k for k, v in RECIPES_ALL.items() if v["categoria"] == "Alerta"]
CAT_CRIT_ABRUPTO = [k for k, v in RECIPES_ALL.items()
                    if v["categoria"] == "Critico" and not v.get("progressive")]
CAT_CRIT_PROG = [k for k, v in RECIPES_ALL.items()
                 if v["categoria"] == "Critico" and v.get("progressive")]

# Mix por viagem (12 slots — §3): 6 N, 2 A, 2 L, 1 C-abrupto, 1 C-progressivo
TRIP_SLOTS = (["N"] * 6) + (["A"] * 2) + (["L"] * 2) + ["CA", "CP"]
SLOT_POOLS = {"N": CAT_NORMAL, "A": CAT_ATENCAO, "L": CAT_ALERTA,
              "CA": CAT_CRIT_ABRUPTO, "CP": CAT_CRIT_PROG}

# Mix A-OSC (§12, análise adicional): injeta a patologia que motiva a
# mediação — risco oscilante com cruzamentos transitórios do limiar.
TRIP_SLOTS_OSC = (["N"] * 4) + (["A"] * 2) + ["L", "LO", "LO", "CA", "CP", "CO"]

# Receitas oscilantes VIRTUAIS (não tocam o gerador v7; curvas determinísticas
# sintetizadas aqui, com fase sorteada no fluxo exógeno de receitas):
#   L04-osc     — Alerta: y oscila 0,50±0,12 (período 12 frames) — cruza
#                 θ_on≈0,55 de forma transitória; alarme frame-level metralha.
#   C11-flutter — Crítico abrupto: flutter pré-onset 0,45±0,12 (período 8) e,
#                 do onset (t=48) em diante, rampa até ~0,90 persistente.
OSC_ONSET_C11 = 48

WINDOW_T = 96                 # frames por episódio (10 s)
DT_S = 10.0 / WINDOW_T        # segundos por frame
SENSOR_LEVEL = 2
OBS_NOISE_GAIN = 0.10         # §4: ruído de observação

# ---- Perfis de observação/resposta por persona (tabela fixa, §3–4) --------
# (observabilidade, responsividade, suscetibilidade à fadiga)
PERSONA_PROFILES: Dict[str, tuple] = {
    "P01": (0.90, 0.85, 0.15), "P02": (0.85, 0.60, 0.30),
    "P03": (0.75, 0.65, 0.55), "P04": (0.85, 0.70, 0.35),
    "P05": (0.80, 0.75, 0.30), "P06": (0.85, 0.70, 0.30),
    "P07": (0.85, 0.60, 0.35), "P08": (0.70, 0.55, 0.60),
    "P09": (0.85, 0.60, 0.40), "P10": (0.65, 0.45, 0.50),
    "P11": (0.70, 0.75, 0.35), "P12": (0.75, 0.70, 0.50),
}
ALL_PERSONAS = list(PERSONA_PROFILES)


@dataclass
class Episode:
    persona: str
    trip: int
    slot: int
    recipe: str
    categoria: str            # Normal | Atencao | Alerta | Critico
    progressive: bool
    onset: int                # event_start; -1 se não crítico/sem evento
    p_obs: np.ndarray         # [T] — saída do modelo SE computado no frame t
    dx: np.ndarray            # [T] — variação causal do vetor de entrada
    response_draw: float      # uniforme pré-sorteado p/ resposta do motorista


@dataclass
class SeedData:
    seed: int
    personas: List[str]
    n_trips: int
    episodes: List[Episode] = field(default_factory=list)
    exogenous_hash: str = ""

    def trips(self):
        """Itera (persona, trip, [episódios do trip em ordem de slot])."""
        by = {}
        for ep in self.episodes:
            by.setdefault((ep.persona, ep.trip), []).append(ep)
        for key in sorted(by):
            yield key[0], key[1], sorted(by[key], key=lambda e: e.slot)


def _dx_series(X: np.ndarray) -> np.ndarray:
    """Variação causal e livre de escala do vetor de entrada (proxy Δx_t)."""
    d = np.abs(np.diff(X, axis=0)) / (1.0 + np.abs(X[:-1]))
    dx = np.concatenate([[1.0], d.mean(axis=1)])   # frame 0 força computação
    return dx.astype(np.float64)


def _osc_recipe(kind: str, phase: int):
    """Sintetiza (y_clean, recipe, categoria, progressive, onset) oscilante."""
    t = np.arange(WINDOW_T, dtype=np.float64)
    if kind == "LO":
        y = 0.50 + 0.12 * np.sin(2 * np.pi * (t + phase) / 12.0)
        return np.clip(y, 0, 1), "L04-osc", "Alerta", False, -1
    # CO — flutter pré-onset + crítico persistente
    y = 0.45 + 0.12 * np.sin(2 * np.pi * (t + phase) / 8.0)
    ramp = np.minimum(1.0, (t - OSC_ONSET_C11) / 12.0)
    post = 0.45 + ramp * (0.90 - 0.45)
    y = np.where(t >= OSC_ONSET_C11, np.maximum(y, post), y)
    return np.clip(y, 0, 1), "C11-flutter", "Critico", False, OSC_ONSET_C11


def _osc_dx(y: np.ndarray) -> np.ndarray:
    """Δx derivado da curva: espelha a variação do sinal na entrada.
    Oscilações (|Δy|≈0,06/frame) disparam o MHEG; platôs (0,002) reutilizam."""
    d = 0.002 + 2.0 * np.abs(np.diff(y))
    return np.concatenate([[1.0], d]).astype(np.float64)


def generate_seed(seed: int, personas: List[str] = None,
                  n_trips: int = 20, osc_mix: bool = False) -> SeedData:
    """Gera todos os episódios de uma seed com exogenia pré-amostrada.

    ``osc_mix=True`` usa o mix A-OSC (protocolo §12): dois slots L04-osc e um
    C11-flutter por viagem, sintetizados por receita virtual determinística.
    """
    personas = personas or ALL_PERSONAS
    root = np.random.SeedSequence(seed)
    ss_recipe, ss_gen, ss_obs, ss_resp = root.spawn(4)
    rng_recipe = np.random.default_rng(ss_recipe)
    rng_obs = np.random.default_rng(ss_obs)
    rng_resp = np.random.default_rng(ss_resp)
    gen_seeds = np.random.default_rng(ss_gen).integers(0, 2**31 - 1,
                                                       size=10_000_000)

    slots_template = TRIP_SLOTS_OSC if osc_mix else TRIP_SLOTS
    data = SeedData(seed=seed, personas=personas, n_trips=n_trips)
    digest = hashlib.sha256()
    idx = 0
    for persona in personas:
        risk_mod = _personas.PERSONAS[persona].risk_mod
        observability = PERSONA_PROFILES[persona][0]
        for trip in range(n_trips):
            order = list(slots_template)
            rng_recipe.shuffle(order)
            for slot, cat in enumerate(order):
                if cat in ("LO", "CO"):
                    phase = int(rng_recipe.integers(0, 12))
                    y_clean, recipe, categoria, progressive, onset = \
                        _osc_recipe(cat, phase)
                    dx = _osc_dx(y_clean)
                else:
                    recipe = SLOT_POOLS[cat][
                        int(rng_recipe.integers(0, len(SLOT_POOLS[cat])))]
                    # o gerador v7 usa np.random/random globais — semeia por episódio
                    child = int(gen_seeds[idx]); idx += 1
                    np.random.seed(child); _random.seed(child)
                    X, _, y_clean, _, meta = _gen.generate_episode(
                        recipe, RECIPES_ALL, window=WINDOW_T,
                        SENSOR_LEVEL=SENSOR_LEVEL)
                    categoria = meta["categoria"]
                    progressive = bool(RECIPES_ALL[recipe].get("progressive"))
                    onset = int(meta.get("event_start", -1))
                    dx = _dx_series(X)

                p_true = np.clip(np.asarray(y_clean, dtype=np.float64)
                                 * risk_mod, 0, 1)
                eps = rng_obs.standard_normal(WINDOW_T)
                p_obs = np.clip(
                    p_true + eps * OBS_NOISE_GAIN * (1.2 - observability), 0, 1)
                episode = Episode(
                    persona=persona, trip=trip, slot=slot, recipe=recipe,
                    categoria=categoria, progressive=progressive,
                    onset=onset, p_obs=p_obs, dx=dx,
                    response_draw=float(rng_resp.uniform()),
                )
                data.episodes.append(episode)
                digest.update(p_obs.tobytes()); digest.update(episode.dx.tobytes())
                digest.update(np.float64(episode.response_draw).tobytes())

    data.exogenous_hash = digest.hexdigest()
    return data


def response_probability(persona: str, alerts_so_far_in_trip: int,
                         n_episodes: int = len(TRIP_SLOTS)) -> float:
    """q = clip(responsividade − 0,4·fadiga_acumulada, 0,05, 1) — §4."""
    _, responsiveness, fatigue = PERSONA_PROFILES[persona]
    drag = fatigue * min(1.0, alerts_so_far_in_trip / n_episodes)
    return float(np.clip(responsiveness - 0.4 * drag, 0.05, 1.0))
