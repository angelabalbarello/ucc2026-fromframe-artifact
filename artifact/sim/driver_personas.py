# driver_personas.py
# SDRCognitive — Simulador de ROTEIROS e CENÁRIOS de teste por TIPO de motorista
# ============================================================================
#
# OBJETIVO
#   Gerar, de forma autocontida (sem importar o gerador SHRP2 v7), um conjunto
#   de "roteiros de teste" (test scripts) e "cenários de teste" (test scenarios)
#   para DIVERSOS TIPOS / PERSONAS de motorista. É o ângulo de PERSONALIZAÇÃO da
#   tese (perfil por condutor Lambda(d)): em vez de receitas isoladas de
#   comportamento, modelamos um condutor com propensão própria numa viagem.
#
# ESPAÇO DE VARIÁVEIS — v9 (37 dims)  [AMPLIADO]
#   O laboratório evoluiu o espaço x_t: v7 (31) -> v8 (34, +som de risco rodoviário)
#   -> v9 (37, +dinâmica veicular). Este gerador opera no espaço CANÔNICO v9 e usa
#   o MESMO layout de índices de ml/channels.py / sim/teragen_v9.py, permanecendo
#   independente (replica o mapa como constantes, não importa o v7):
#
#     behavior      0..10   one-hot de comportamento + cinto/confianças
#     speed         11      speed/150
#     kinematics    12,13,14 accel . brake . lateral     16  celular x velocidade
#     cognitive     15      fadiga (estado inferido)
#     environment   17,18,19,20  highway . urban . rain . night
#     traffic       21,22,23     one-hot livre/moderado/congestionado
#     acoustic(v7)  24,25,26,27,28  siren.horn.crying.ring.music
#     gaze          29      eye_gaze_off_road
#     lane          30      lane_offset_norm
#     acoustic(v8)  31,32,33     crash . tire_screech . pothole   [risco rodoviário]
#     vehicle(v9)   34,35,36     yaw_rate . steering_angle . rpm  [telemetria]
#
# IMPORTANTE — adicionar PERSONAS (inclusive rural/agrícola e transporte de carga)
#   NÃO altera a dimensão de entrada: novas personas são compostas a partir das
#   variáveis JÁ existentes (classe de veículo e tipo de via entram como
#   MODULADORES do risco e da cinemática, não como novos canais). Só haveria
#   mudança de dimensão se quiséssemos sensores físicos novos (ex.: peso da carga,
#   rugosidade do terreno), o que implicaria um v10 do x_t + re-treino do modelo.
#
# CONCEITOS
#   PERSONA  — tipo de motorista, com modificador de risco (propensão Lambda(d))
#              e classe de veículo (passeio/carga/agrícola).
#   ROTEIRO  — sequência ordenada de FASES de uma viagem (o "script" de teste).
#   FASE/STEP— trecho com comportamento dominante + contexto (via, vel., noite,
#              chuva, tráfego, cinto, sons). Cada fase é também um CENÁRIO isolado,
#              com categoria de risco ESPERADA (oráculo determinístico).
#   CENÁRIO  — unidade verificável: persona+fase+features sintéticas (37-d) +
#              risco esperado + categoria (Normal/Atenção/Alerta/Crítico).
#
# SAÍDAS: JSON (oráculo dos testes) . CSV (achatado) . NPZ (tensores X T x 37).
#
# Calibração de risco inspirada (ilustrativa) em odds ratios de naturalistic
# driving: Dingus et al. (2016) PNAS 113(10):2636-2641.
# ============================================================================

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# 0. CONSTANTES GLOBAIS — espaço v9 (37 dims)
# ----------------------------------------------------------------------------
XT_VERSION = "v9"
FEATURE_SIZE = 37          # v9: v7(31) + crash/tire/pothole(3) + yaw/steering/rpm(3)
WINDOW = 96
SEED_DEFAULT = 123

# Índices canônicos (espelham ml/channels.py — não importamos, replicamos).
IDX_SPEED = 11
IDX_ACCEL, IDX_BRAKE, IDX_LATERAL = 12, 13, 14
IDX_FATIGUE = 15
IDX_PHONE_SPEED = 16
IDX_HIGHWAY, IDX_URBAN, IDX_RAIN, IDX_NIGHT = 17, 18, 19, 20
IDX_TRAFFIC = {"livre": 21, "moderado": 22, "congestionado": 23}
IDX_ACOUSTIC = {"sirene": 24, "buzina": 25, "choro": 26, "toque": 27, "musica": 28}
IDX_GAZE, IDX_LANE = 29, 30
IDX_CRASH, IDX_TIRE, IDX_POTHOLE = 31, 32, 33      # v8 (som de risco rodoviário)
IDX_YAW, IDX_STEERING, IDX_RPM = 34, 35, 36        # v9 (dinâmica veicular)

# Bandas de categoria por risco (limiar inferior fechado, superior aberto).
CATEGORY_BANDS: List[Tuple[str, float, float]] = [
    ("Normal",  0.00, 0.15),
    ("Atencao", 0.15, 0.33),
    ("Alerta",  0.33, 0.60),
    ("Critico", 0.60, 1.01),
]


def categorize(risk: float) -> str:
    r = float(np.clip(risk, 0.0, 1.0))
    for name, lo, hi in CATEGORY_BANDS:
        if lo <= r < hi:
            return name
    return "Critico"


# ----------------------------------------------------------------------------
# 1. COMPORTAMENTOS
# ----------------------------------------------------------------------------
# Risco-base a ~60 km/h, dia, seco (calibração qualitativa por OR SHRP2).
BEHAVIOR_BASE_RISK: Dict[str, float] = {
    "normal":          0.05,
    "radio":           0.08,
    "conversa":        0.10,
    "comendo":         0.18,
    "grooming":        0.18,
    "bebendo":         0.20,
    "pegar_objeto":    0.30,
    "celular_falando": 0.32,
    "sonolento":       0.42,
    "fadiga":          0.55,
    "celular_texto":   0.58,
    "substancia":      0.72,
}

# Comportamento -> código one-hot canônico (0..6) do x_t.
#   0 normal . 1 celular . 2 radio . 3 ingestão . 4 pegar_objeto . 5 grooming . 6 conversa
BEHAVIOR_ONEHOT: Dict[str, int] = {
    "normal": 0, "radio": 2, "conversa": 6, "comendo": 3, "grooming": 5,
    "bebendo": 3, "pegar_objeto": 4, "celular_falando": 1, "celular_texto": 1,
    "sonolento": 0, "fadiga": 0, "substancia": 3,
}

PHONE_BEHAVIORS = {"celular_falando", "celular_texto"}
# Nível de fadiga (v[15]) por comportamento cognitivo.
FATIGUE_LEVEL: Dict[str, float] = {"sonolento": 0.55, "fadiga": 0.90, "substancia": 0.50}
GAZE_OFF_BEHAVIORS = {"celular_texto", "pegar_objeto", "grooming", "comendo",
                      "fadiga", "sonolento"}
LANE_DRIFT_BEHAVIORS = {"fadiga", "sonolento", "substancia"}


# ----------------------------------------------------------------------------
# 2. MULTIPLICADORES DE CONTEXTO
# ----------------------------------------------------------------------------
def speed_factor(speed_kmh: float) -> float:
    return float(np.clip(speed_kmh / 60.0, 0.30, 1.60))


TRAFFIC_MULT = {"livre": 1.00, "moderado": 1.06, "congestionado": 1.15}
NIGHT_MULT = 1.20
RAIN_MULT = 1.25
NO_SEATBELT_MULT = 1.10
# Carga cognitiva/startle de sons (variáveis ampliadas) — efeito menor.
ACOUSTIC_LOAD = {"musica": 0.05, "choro": 0.05, "sirene": 0.03,
                 "buzina": 0.02, "toque": 0.03}
# Tipo de via. "rural" = estrada não pavimentada / vicinal (terreno irregular,
# fauna, baixa sinalização). Representado nas variáveis EXISTENTES (env + cinemática),
# sem criar nova dimensão de entrada.
ROAD_MULT = {"urbano": 1.00, "rodovia": 1.00, "rural": 1.08}
# Classe do veículo. Inércia/massa e pontos cegos elevam o risco composto (carga),
# uso agrícola eleva levemente. Também modula a cinemática renderizada.
VEHICLE_RISK_MOD = {"passeio": 1.00, "carga": 1.12, "agricola": 1.06}


# ----------------------------------------------------------------------------
# 3. PERSONAS
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Persona:
    pid: str
    nome: str
    grupo: str
    risk_mod: float
    descricao: str
    vehicle: str = "passeio"      # passeio | carga | agricola


PERSONAS: Dict[str, Persona] = {
    # ---- Conjunto clássico de risco ----
    "P01": Persona("P01", "Cauteloso experiente", "risco_classico", 0.80,
                   "Condutor maduro, defensivo, antecipa riscos e mantém margem."),
    "P02": Persona("P02", "Jovem agressivo", "risco_classico", 1.28,
                   "Alta velocidade, ultrapassagens, baixa percepção de risco."),
    "P03": Persona("P03", "Profissional fatigado", "risco_classico", 1.18,
                   "Longas jornadas (carga/rodovia), privação de sono acumulada."),
    "P04": Persona("P04", "Commuter distraído", "risco_classico", 1.12,
                   "Trajeto urbano repetitivo; multitarefa por hábito/tédio."),
    "P05": Persona("P05", "Idoso", "risco_classico", 1.10,
                   "Reação mais lenta; conservador na velocidade, porém vulnerável."),
    "P06": Persona("P06", "Novato", "risco_classico", 1.22,
                   "Pouca experiência, carga cognitiva alta em tarefas simples."),
    # ---- Personas por padrão de distração ----
    "P07": Persona("P07", "Usuário de celular", "padrao_distracao", 1.16,
                   "Uso recorrente do celular (falar/digitar) ao volante."),
    "P08": Persona("P08", "Sonolento", "padrao_distracao", 1.20,
                   "Sonolência diurna/noturna; episódios de microssono."),
    "P09": Persona("P09", "Multitarefa", "padrao_distracao", 1.20,
                   "Encadeia distrações: comer, rádio, conversa, objeto."),
    "P10": Persona("P10", "Sob substância", "padrao_distracao", 1.32,
                   "Condução sob efeito de álcool/substância."),
    # ---- Frota profissional / contexto de campo (field operations) ----
    "P11": Persona("P11", "Rural/agrícola", "profissional_frota", 1.10,
                   "Picape/trator em estrada vicinal; jornada de campo, terreno "
                   "irregular, fauna e baixa sinalização.", vehicle="agricola"),
    "P12": Persona("P12", "Transporte de carga", "profissional_frota", 1.15,
                   "Caminhão em rodovia longa; alta inércia, pontos cegos, "
                   "privação de sono em viagens extensas.", vehicle="carga"),
}


# ----------------------------------------------------------------------------
# 4. FASE (STEP)
# ----------------------------------------------------------------------------
@dataclass
class Fase:
    behavior: str
    dur: int
    speed: float = 60.0
    road: str = "urbano"              # "urbano" | "rodovia" | "rural"
    traffic: str = "livre"
    night: bool = False
    rain: bool = False
    seatbelt: bool = True
    sons: Tuple[str, ...] = ()        # subset de IDX_ACOUSTIC
    note: str = ""


# Saturação suave: preserva a faixa < CRIT_KNEE (Normal/Atenção/Alerta) intacta e
# comprime a região crítica em [CRIT_KNEE, 1.0) de forma monotônica — assim os
# perfis de alto risco NÃO saturam todos em 1.00 e permanecem distinguíveis.
CRIT_KNEE = 0.60      # joelho no limiar da categoria Crítico
CRIT_K = 0.80         # suavidade da compressão (menor = comprime mais cedo)


def _soft_saturate(raw: float) -> float:
    raw = max(0.0, float(raw))
    if raw <= CRIT_KNEE:
        return raw
    return float(CRIT_KNEE + (1.0 - CRIT_KNEE) * (1.0 - np.exp(-(raw - CRIT_KNEE) / CRIT_K)))


def fase_risk(p: Persona, f: Fase) -> float:
    """Oráculo determinístico: risco esperado de uma fase para uma persona."""
    r = BEHAVIOR_BASE_RISK[f.behavior]
    r *= speed_factor(f.speed)
    r *= TRAFFIC_MULT[f.traffic]
    if f.night:
        r *= NIGHT_MULT
    if f.rain:
        r *= RAIN_MULT
    if not f.seatbelt:
        r *= NO_SEATBELT_MULT
    for s in f.sons:
        r *= (1.0 + ACOUSTIC_LOAD.get(s, 0.0))
    r *= ROAD_MULT.get(f.road, 1.0)
    r *= VEHICLE_RISK_MOD.get(p.vehicle, 1.0)
    r *= p.risk_mod
    return _soft_saturate(r)


# ----------------------------------------------------------------------------
# 5. ROTEIROS — uma viagem por persona
# ----------------------------------------------------------------------------
ROTEIROS: Dict[str, List[Fase]] = {
    "P01": [
        Fase("normal", 40, speed=95, road="rodovia", note="rodovia, fluxo livre"),
        Fase("radio", 16, speed=90, road="rodovia", sons=("musica",), note="ajusta rádio"),
        Fase("normal", 24, speed=55, traffic="moderado", note="entrada urbana"),
        Fase("conversa", 16, speed=40, traffic="moderado", note="conversa curta"),
    ],
    "P02": [
        Fase("normal", 30, speed=120, road="rodovia", note="rodovia em alta velocidade"),
        Fase("celular_texto", 22, speed=115, road="rodovia", note="digitando a 115 km/h"),
        Fase("pegar_objeto", 14, speed=110, road="rodovia", note="pega objeto no banco"),
        Fase("normal", 30, speed=118, road="rodovia", note="retoma alta velocidade"),
    ],
    "P03": [
        Fase("normal", 34, speed=85, road="rodovia", night=True, note="rodovia noturna, longa"),
        Fase("sonolento", 18, speed=82, road="rodovia", night=True, note="sonolência crescente"),
        Fase("fadiga", 16, speed=80, road="rodovia", night=True, note="microssono / desvio"),
        Fase("normal", 28, speed=78, road="rodovia", night=True, note="recupera, ainda noturno"),
    ],
    "P04": [
        Fase("normal", 30, speed=45, traffic="moderado", note="urbano de rotina"),
        Fase("celular_falando", 18, speed=40, traffic="congestionado", sons=("toque",),
             note="atende celular no congestionamento"),
        Fase("grooming", 14, speed=20, traffic="congestionado", note="se ajeita no espelho"),
        Fase("conversa", 18, speed=35, traffic="moderado", sons=("musica",), note="conversa+rádio"),
    ],
    "P05": [
        Fase("normal", 36, speed=40, traffic="moderado", note="urbano lento"),
        Fase("conversa", 18, speed=35, traffic="moderado", note="conversa"),
        Fase("pegar_objeto", 12, speed=30, traffic="moderado", note="procura algo"),
        Fase("normal", 30, speed=38, traffic="moderado", sons=("buzina",), note="segue urbano"),
    ],
    "P06": [
        Fase("normal", 30, speed=50, note="via arterial"),
        Fase("pegar_objeto", 14, speed=45, traffic="moderado", note="distração ao pegar objeto"),
        Fase("grooming", 12, speed=40, traffic="moderado", note="ajeita-se"),
        Fase("conversa", 18, speed=48, note="conversa enquanto dirige"),
    ],
    "P07": [
        Fase("normal", 26, speed=70, road="rodovia", note="início rodovia"),
        Fase("celular_falando", 18, speed=72, road="rodovia", sons=("toque",), note="falando ao celular"),
        Fase("celular_texto", 20, speed=68, road="rodovia", note="digitando mensagem"),
        Fase("normal", 28, speed=70, road="rodovia", note="retoma normal"),
    ],
    "P08": [
        Fase("normal", 28, speed=90, road="rodovia", night=True, note="rodovia noturna"),
        Fase("sonolento", 22, speed=88, road="rodovia", night=True, note="sonolência"),
        Fase("fadiga", 18, speed=85, road="rodovia", night=True, rain=True, note="microssono sob chuva"),
        Fase("normal", 24, speed=82, road="rodovia", night=True, note="recupera"),
    ],
    "P09": [
        Fase("comendo", 18, speed=55, traffic="moderado", note="comendo ao dirigir"),
        Fase("radio", 12, speed=58, sons=("musica",), note="mexe no rádio"),
        Fase("celular_falando", 18, speed=56, traffic="moderado", sons=("toque",), note="atende celular"),
        Fase("conversa", 18, speed=54, note="conversa"),
        Fase("pegar_objeto", 12, speed=50, note="pega objeto"),
    ],
    "P10": [
        Fase("normal", 26, speed=95, road="rodovia", night=True, seatbelt=False, note="rodovia noturna"),
        Fase("substancia", 24, speed=92, road="rodovia", night=True, seatbelt=False, note="condução sob efeito"),
        Fase("fadiga", 16, speed=88, road="rodovia", night=True, seatbelt=False, note="desvio cinemático"),
        Fase("normal", 24, speed=85, road="rodovia", night=True, seatbelt=False, note="segue noturno"),
    ],
    # P11 Rural/agrícola: baixa velocidade, estrada de terra, fadiga de jornada.
    "P11": [
        Fase("normal", 34, speed=35, road="rural", note="estrada vicinal de terra"),
        Fase("fadiga", 16, speed=30, road="rural", note="jornada longa no campo"),
        Fase("pegar_objeto", 14, speed=25, road="rural", note="ajusta ferramenta/carga"),
        Fase("normal", 30, speed=32, road="rural", sons=("buzina",), note="cruza com outro veículo"),
    ],
    # P12 Transporte de carga: rodovia noturna longa, fadiga, alta inércia.
    "P12": [
        Fase("normal", 30, speed=90, road="rodovia", night=True, note="rodovia, carga pesada"),
        Fase("sonolento", 20, speed=88, road="rodovia", night=True, note="sonolência na madrugada"),
        Fase("fadiga", 18, speed=85, road="rodovia", night=True, rain=True, note="microssono sob chuva"),
        Fase("normal", 28, speed=82, road="rodovia", night=True, note="segue com carga"),
    ],
}


# ----------------------------------------------------------------------------
# 6. SÍNTESE DE FRAMES (X em R^{T x 37}) — layout canônico v9
# ----------------------------------------------------------------------------
def _frame(p: Persona, f: Fase, risk: float, rng: np.random.Generator) -> np.ndarray:
    v = np.clip(rng.normal(0.0, 0.012, size=FEATURE_SIZE), 0.0, 1.0).astype(np.float32)
    # behavior one-hot (0..6) + cinto/confianças
    v[BEHAVIOR_ONEHOT[f.behavior]] = 1.0
    v[9] = float(np.clip(0.95 + rng.normal(0, 0.03), 0, 1))         # conf comportamento
    v[10] = float(np.clip((0.97 if f.seatbelt else 0.40) + rng.normal(0, 0.03), 0, 1))
    # speed / kinematics
    v[IDX_SPEED] = float(np.clip(f.speed / 150.0, 0, 1))
    v[IDX_ACCEL] = float(np.clip(0.05 + rng.normal(0, 0.03), 0, 1))
    brake = 0.02
    if p.vehicle == "carga":
        brake += 0.10            # maior inércia -> frenagens mais intensas/longas
    v[IDX_BRAKE] = float(np.clip(brake + rng.normal(0, 0.02), 0, 1))
    lat = 0.45 if f.behavior in LANE_DRIFT_BEHAVIORS else 0.05
    if f.road == "rural":
        lat += 0.18              # terreno não pavimentado / irregular
    v[IDX_LATERAL] = float(np.clip(lat + rng.normal(0, 0.04), 0, 1))
    # cognitive
    v[IDX_FATIGUE] = float(np.clip(FATIGUE_LEVEL.get(f.behavior, 0.0) + rng.normal(0, 0.04), 0, 1))
    # celular x velocidade
    if f.behavior in PHONE_BEHAVIORS:
        v[IDX_PHONE_SPEED] = float(np.clip(f.speed / 40.0, 0, 1))
    # environment (rural = nem rodovia, nem urbano: via vicinal aberta)
    v[IDX_HIGHWAY] = 1.0 if f.road == "rodovia" else 0.0
    v[IDX_URBAN] = 1.0 if f.road == "urbano" else 0.0
    v[IDX_RAIN] = 1.0 if f.rain else 0.0
    v[IDX_NIGHT] = 1.0 if f.night else 0.0
    # traffic one-hot
    v[IDX_TRAFFIC[f.traffic]] = 1.0
    # acoustic v7
    for s in f.sons:
        if s in IDX_ACOUSTIC:
            v[IDX_ACOUSTIC[s]] = 1.0
    # gaze / lane
    if f.behavior in GAZE_OFF_BEHAVIORS:
        v[IDX_GAZE] = float(np.clip(0.6 + rng.normal(0, 0.08), 0, 1))
    if f.behavior in LANE_DRIFT_BEHAVIORS:
        v[IDX_LANE] = float(np.clip(0.5 + rng.normal(0, 0.10), 0, 1))
    # v9 dinâmica veicular (derivada de speed/lateral/risco, como teragen_v9)
    sp = v[IDX_SPEED]
    v[IDX_RPM] = float(np.clip(0.35 * sp + 0.5 * v[IDX_ACCEL] + rng.normal(0, 0.03), 0, 1))
    v[IDX_YAW] = float(np.clip(0.7 * v[IDX_LATERAL] + 0.25 * risk + abs(rng.normal(0, 0.05)), 0, 1))
    v[IDX_STEERING] = float(np.clip(0.6 * v[IDX_LATERAL] + 0.20 * risk + abs(rng.normal(0, 0.05)), 0, 1))
    # pothole esparso
    if rng.random() < 0.02:
        v[IDX_POTHOLE] = float(rng.uniform(0.3, 0.7))
    return v


def synth_window(p: Persona, fases: List[Fase], window: int,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Gera X (window x 37) e a curva de risco por frame para o roteiro inteiro."""
    total = sum(f.dur for f in fases)
    scale = window / total
    frames: List[np.ndarray] = []
    risks: List[float] = []
    for f in fases:
        n = max(1, int(round(f.dur * scale)))
        r = fase_risk(p, f)
        for _ in range(n):
            if len(frames) >= window:
                break
            frames.append(_frame(p, f, r, rng))
            risks.append(r)
    while len(frames) < window:
        frames.append(frames[-1].copy())
        risks.append(risks[-1])
    X = np.stack(frames[:window], axis=0)
    y = np.array(risks[:window], dtype=np.float32)
    # crash / tire_screech ao redor do pico (som de risco rodoviário — v8)
    if y.max() > 0.5:
        peak = int(np.argmax(y))
        for k in range(max(0, peak - 1), min(window, peak + 2)):
            X[k, IDX_CRASH] = float(np.clip(0.8 + rng.normal(0, 0.05), 0, 1))
        for k in range(max(0, peak - 5), peak):
            X[k, IDX_TIRE] = float(np.clip(0.55 + rng.normal(0, 0.05), 0, 1))
    return X, y


# ----------------------------------------------------------------------------
# 7. CONSTRUÇÃO DE CENÁRIOS E ROTEIROS
# ----------------------------------------------------------------------------
@dataclass
class Cenario:
    cenario_id: str
    persona_id: str
    persona_nome: str
    grupo: str
    vehicle: str
    fase_idx: int
    behavior: str
    speed: float
    road: str
    traffic: str
    night: bool
    rain: bool
    seatbelt: bool
    sons: List[str]
    note: str
    risk_esperado: float
    categoria_esperada: str


@dataclass
class RoteiroResult:
    persona_id: str
    persona_nome: str
    grupo: str
    vehicle: str
    risk_mod: float
    descricao: str
    n_fases: int
    risk_episodio: float
    categoria_episodio: str
    categoria_dominante: str
    cenarios: List[Cenario] = field(default_factory=list)


def build_roteiro(pid: str) -> RoteiroResult:
    p = PERSONAS[pid]
    fases = ROTEIROS[pid]
    cenarios: List[Cenario] = []
    cats: List[str] = []
    risks: List[float] = []
    for i, f in enumerate(fases):
        r = fase_risk(p, f)
        c = categorize(r)
        cats.append(c)
        risks.append(r)
        cenarios.append(Cenario(
            cenario_id=f"{pid}-S{i+1:02d}",
            persona_id=pid, persona_nome=p.nome, grupo=p.grupo, vehicle=p.vehicle,
            fase_idx=i, behavior=f.behavior, speed=f.speed, road=f.road,
            traffic=f.traffic, night=f.night, rain=f.rain, seatbelt=f.seatbelt,
            sons=list(f.sons), note=f.note,
            risk_esperado=round(r, 4), categoria_esperada=c,
        ))
    pico = max(risks)
    dominante = max(set(cats), key=cats.count)
    return RoteiroResult(
        persona_id=pid, persona_nome=p.nome, grupo=p.grupo, vehicle=p.vehicle,
        risk_mod=p.risk_mod, descricao=p.descricao, n_fases=len(fases),
        risk_episodio=round(pico, 4), categoria_episodio=categorize(pico),
        categoria_dominante=dominante, cenarios=cenarios,
    )


def build_all() -> List[RoteiroResult]:
    return [build_roteiro(pid) for pid in PERSONAS]


# ----------------------------------------------------------------------------
# 8. EXPORTAÇÃO
# ----------------------------------------------------------------------------
def to_json(roteiros: List[RoteiroResult]) -> Dict[str, Any]:
    return {
        "schema": "SDR.driver_personas.v2",
        "xt_version": XT_VERSION,
        "feature_size": FEATURE_SIZE,
        "window": WINDOW,
        "category_bands": [
            {"categoria": n, "min": lo, "max": hi} for n, lo, hi in CATEGORY_BANDS
        ],
        "roteiros": [
            {
                **{k: v for k, v in asdict(r).items() if k != "cenarios"},
                "cenarios": [asdict(c) for c in r.cenarios],
            }
            for r in roteiros
        ],
    }


def save_json(path: str, roteiros: List[RoteiroResult]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(to_json(roteiros), fp, ensure_ascii=False, indent=2)


def save_csv(path: str, roteiros: List[RoteiroResult]) -> None:
    header = ["cenario_id", "persona_id", "persona_nome", "grupo", "vehicle",
              "fase_idx", "behavior", "speed", "road", "traffic", "night", "rain",
              "seatbelt", "sons", "risk_esperado", "categoria_esperada", "note"]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, delimiter=";")
        w.writerow(header)
        for r in roteiros:
            for c in r.cenarios:
                w.writerow([c.cenario_id, c.persona_id, c.persona_nome, c.grupo,
                            c.vehicle, c.fase_idx, c.behavior, c.speed, c.road,
                            c.traffic, c.night, c.rain, c.seatbelt, "|".join(c.sons),
                            c.risk_esperado, c.categoria_esperada, c.note])


def save_npz(path: str, window: int = WINDOW, seed: int = SEED_DEFAULT) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    Xs, Ys, ids = [], [], []
    for pid in PERSONAS:
        X, y = synth_window(PERSONAS[pid], ROTEIROS[pid], window, rng)
        Xs.append(X)
        Ys.append(y)
        ids.append(pid)
    np.savez_compressed(path, X=np.stack(Xs), y_frame=np.stack(Ys),
                        persona_ids=np.array(ids))
    return {"path": path, "shape": (len(ids), window, FEATURE_SIZE)}


# ----------------------------------------------------------------------------
# 9. CLI
# ----------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Gera roteiros e cenários de teste por tipo de motorista (espaço v9, 37-d).")
    ap.add_argument("--out-json", default="cenarios_personas.json")
    ap.add_argument("--out-csv", default="cenarios_personas.csv")
    ap.add_argument("--out-npz", default="", help="opcional: salva tensores X (T x 37)")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    roteiros = build_all()
    save_json(args.out_json, roteiros)
    save_csv(args.out_csv, roteiros)

    print(f"=== SDR— Roteiros/Cenários por tipo de motorista (x_t {XT_VERSION}, D={FEATURE_SIZE}) ===")
    print(f"Personas: {len(roteiros)} | Cenários: {sum(len(r.cenarios) for r in roteiros)}")
    print(f"{'Persona':24s} {'Grupo':18s} {'veículo':9s} {'pico':>5s}  {'episódio':10s} {'dominante':10s}")
    for r in roteiros:
        print(f"{r.persona_nome:24s} {r.grupo:18s} {r.vehicle:9s} {r.risk_episodio:5.2f}  "
              f"{r.categoria_episodio:10s} {r.categoria_dominante:10s}")
    print(f"\nJSON: {args.out_json}\nCSV : {args.out_csv}")
    if args.out_npz:
        info = save_npz(args.out_npz, args.window, args.seed)
        print(f"NPZ : {info['path']}  shape={info['shape']}")


if __name__ == "__main__":
    main()
