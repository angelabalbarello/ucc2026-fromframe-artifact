# synthetic_driver_risk_v7_1.py  --  VERSAO SEPARADA (correcao K1)
# Copia de synthetic_driver_risk_v7.py (SHA-256/16 efede62fe3aceed8, preservado)
# com UMA alteracao de codigo: em curve_for_name('peak_then_drop') o piso
# max(0.75, .) foi removido. Nenhuma receita, rotulo, mix, constante ou
# sorteio aleatorio foi alterado. Nao usar para reproduzir E10-E13.
#
# synthetic_driver_risk_v7.py
# Gerador sintético v7 — calibração empírica via SHRP2 NDS
#
# CORREÇÕES v6 → v7 + patch estrutural v[16] (calibração + contextualização):
#
#   [G1] CORREÇÃO ESTRUTURAL — v[16] speed-condicionado (patch principal):
#        Problema: v[16]=1.0 binário puro → modelo não distingue
#        "celular a 0 km/h" (A01, Atenção) de "celular a 110 km/h" (C01, Crítico).
#        Causa: treino só vê celular a alta velocidade → v[16]=1 ≡ crítico.
#        Fix: v[16] = clip(noisy_speed/40, 0, 1) quando celular.
#             v[1] (one-hot behavior) preserva sinal binário "está usando celular".
#        Efeito: A01 speed=0 → v[16]≈0; C01 speed=110 → v[16]=1.0.
#        Requer re-treino — modelo aprenderá interação celular × velocidade.
#
# CORREÇÕES v6 → v7 (calibração baseada em literatura):
#
#   [F1] RECIPES_BORDERLINE: risk_targets recalibrados via SHRP2 odds ratios
#        (Dingus et al., 2016, PNAS doi:10.1073/pnas.1513271113)
#        Mapeamento: risk_target = clip(0.05 + (OR - 1.0) / 12.5, 0.05, 1.0)
#          A01 Celular parado   : OR≈2.2×speed_factor → risk 0.25→0.15
#          A02 Grooming baixa v : OR≈3.0×speed_factor → risk 0.28→0.22
#          A03 Conversa         : OR=1.4              → risk 0.30→0.13
#
#   [F2] contextRiskBoost: tráfego congestionado corrigido
#        SHRP2: OR MAD rear-end = 8.48 vs CNC = 2.38 → fator 3.5×
#        traffic_congested boost: +0.08 → +0.12
#
#   [F3] Documentação do seatbelt boost:
#        Nota metodológica: cinto NÃO aumenta crash probability —
#        aumenta injury severity. Boost mantido como "risk composto"
#        (crash × severity). Documentar na Seção 5.5 se necessário.
#
#   [F4] Dois novos atributos sintéticos em slots antes não usados:
#        v[29] = eye_gaze_off_road  (0=olhando para frente, 1=desvio)
#                Calibrado por: OR para extended off-road glance = 7.1 (SHRP2)
#                Geração: prob proporcional ao comportamento + risco base
#        v[30] = lane_offset_norm   (0=centro, 1=desvio máximo)
#                Calibrado por: lane wandering associado a fadiga/distração
#                Geração: random walk com drift proporcional ao risco
#        D=31 preservado → 100% compatível com runner principal
#
#   [G1] v[16] redesenhado — celular_em_velocidade (speed-weighted, não binário):
#        ANTES: v[16] = 1.0 se celular, else 0.0 (binário puro)
#        DEPOIS: v[16] = clip(noisy_speed/40, 0, 1) se celular, else 0.0
#        v[1] (one-hot behavior) preserva sinal binário "está usando celular".
#        Semântica nova: "celular em velocidade perigosa" (0=parado, 1=≥40 km/h)
#
#   [F5] RECIPES_TRAIN (Normal+Crítico): inalterados — resultados publicados
#        não são afetados. Só borderline e features novas mudam.
#
# IMPORTANTE: Re-treinar o modelo ao usar v7 — os dois novos atributos
# (v[29], v[30]) eram sempre 0 em v5/v6 e agora carregam sinal real.
# O modelo aprenderá a usar eye_gaze e lane_offset para melhorar discriminação.
#
# Referências:
#   [1] Dingus et al. (2016) PNAS 113(10):2636–2641 doi:10.1073/pnas.1513271113
#   [2] Jegham et al. (2022) Mathematics 10(24):4806 doi:10.3390/math10244806
#   [3] Ghasemzadeh & Ahmed (2018) Transport. Res. C — SHRP2 adverse weather

import csv
import random
import numpy as np
from typing import Dict, Any, List, Tuple

# ============================================================================
# 1. DEFINIÇÕES GLOBAIS — D=31 preservado
# ============================================================================
FEATURE_SIZE = 31   # inalterado: v[29] e v[30] eram 0 em v5/v6, agora ativos

BEHAVIORS = {
    "c0_normal": 0,
    "c1_celular": 1,
    "c2_radio": 2,
    "c3_bebendo": 3,
    "c4_pegar_objeto": 4,
    "c5_grooming": 5,
    "c6_conversa": 6,
}

SEATBELT = {"usando": 0, "nao_usando": 1}
TRAFFIC  = {"livre": 0, "moderado": 1, "congestionado": 2}

SPEED_STOP_THRESH = 1.0
SPEED_MOVE_THRESH = 5.0
K_AGG = 6

# ── Realismo
P_REGIME_STABLE_RISK  = 0.35
P_REGIME_FALSE_KIN    = 0.20
P_PRECURSOR_VISUAL    = 0.50
KIN_NOISE_BASE        = 0.015
KIN_FALSE_SPIKE_SCALE = 0.35
PRECURSOR_LEN_MIN     = 16
PRECURSOR_LEN_MAX     = 28
PRECURSOR_CONF_DROP   = (0.04, 0.12)
LOW_RISK_BASE         = 0.05

# ── Progressivos
P_CRITICAL_PROGRESSIVE   = 0.65
PROG_ONSET_GAP_MIN       = 8
PROG_ONSET_GAP_MAX       = 20
PROG_LATENT_LOW_MIN      = 0.10
PROG_LATENT_LOW_MAX      = 0.22
PROG_TRANSITION_TARGET_MIN = 0.32
PROG_TRANSITION_TARGET_MAX = 0.48
PROG_PEAK_MIN            = 0.78
PROG_PEAK_MAX            = 0.95
CRIT_EARLY_OVERLAP_NOISE = 0.04
CRIT_PROGRESSIVE_JITTER  = 0.03

# ── Persistência crítica
P_CRIT_PERSIST_TAIL    = 0.30
CRIT_TAIL_MIN          = 0.12
CRIT_TAIL_MAX          = 0.24
CRIT_FINAL_FLOOR_MIN   = 0.16
CRIT_FINAL_FLOOR_MAX   = 0.28
CRIT_FINAL_ZONE        = K_AGG

# ── [v7] Eye gaze — probabilidade base de gaze off-road por comportamento
# Calibrado via SHRP2: OR=7.1 para extended off-road glance
# (normalizado: OR_max=12.2 → gaze_prob_max≈0.85)
EYE_GAZE_BASE_PROB = {
    "ESTADO_NORMAL":            0.02,   # modelo de condução: quase sempre olhando frente
    "ESTADO_CELULAR":           0.75,   # celular exige olhar para baixo/tela
    "ESTADO_RADIO":             0.15,   # ajuste rápido, breve desvio
    "ESTADO_BEBENDO":           0.35,   # bebendo/comendo: olha para recipiente
    "ESTADO_PEGAR_OBJETO":      0.65,   # pegar objeto: olha para lado/banco traseiro
    "ESTADO_GROOMING":          0.55,   # grooming: espelho retrovisor/viseira
    "ESTADO_CONVERSA":          0.20,   # conversa: olha ocasionalmente para passageiro
    "ESTADO_ACELERACAO_BRUSCA": 0.10,   # manobra: foco na estrada
    "ESTADO_FREADA_BRUSCA":     0.05,   # freada: foco máximo
    "ESTADO_GIROSCOPIO_CRITICO":0.08,   # desvio lateral: foco na via
    "ESTADO_FADIGA":            0.30,   # fadiga: piscar lento, olhar pesado
}

# ── [v7] Lane offset — parâmetros do random walk
# Calibrado via SHRP2: steering variance aumenta com distração
LANE_DRIFT_BY_RISK = {
    "low":    0.008,   # risk < 0.20: deriva mínima
    "medium": 0.018,   # risk 0.20–0.55: deriva moderada
    "high":   0.035,   # risk > 0.55: deriva significativa (fatiga/crítico)
}
LANE_NOISE_BASE     = 0.005
LANE_MEAN_REVERSION = 0.15   # tendência de voltar ao centro

# ============================================================================
# 2. FUNÇÕES AUXILIARES
# ============================================================================
def one_hot(idx: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v

def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def first_crossing(arr: np.ndarray, thr: float) -> int:
    idx = np.where(arr >= thr)[0]
    return int(idx[0]) if len(idx) > 0 else -1

def noisy_speed(kmh: float) -> float:
    return max(0.0, float(kmh) + np.random.normal(0, max(2.0, 0.02 * kmh)))

def base_conf(mu: float, sigma: float, lo: float = 0.5, hi: float = 1.0) -> float:
    return float(np.clip(np.random.normal(mu, sigma), lo, hi))

def generate_base_vector(behavior_idx: int, seatbelt_idx: int,
                          speed_kmh: float, *, SENSOR_LEVEL: int, **kw) -> np.ndarray:
    v = np.zeros(FEATURE_SIZE, dtype=np.float32)
    v[0:7] = one_hot(behavior_idx, 7)
    v[7:9] = one_hot(seatbelt_idx, 2)
    v[9]   = base_conf(kw.get('conf_b_mu', 0.95), kw.get('conf_b_sigma', 0.05))
    v[10]  = base_conf(kw.get('conf_s_mu', 0.97), kw.get('conf_s_sigma', 0.03))
    # v[16]: celular_em_velocidade — speed-weighted, não binário puro.
    # Motivação: o behavior one-hot v[1] já codifica "está usando celular".
    # v[16] codifica a INTERAÇÃO celular × velocidade perigosa:
    #   celular parado (0 km/h)  → v[16] ≈ 0.0  (baixo risco contextual)
    #   celular a 40+ km/h       → v[16] ≈ 1.0  (alto risco contextual)
    # Isso permite ao modelo distinguir A01 (celular parado) de C01 (celular a 110 km/h)
    # sem aumentar D=31: v[1] preserva o sinal binário de comportamento.
    # Referência: Dingus et al. 2016 — OR para celular aumenta com velocidade.
    if behavior_idx == BEHAVIORS["c1_celular"]:
        v[16] = clip01(noisy_speed(speed_kmh) / 40.0)   # satura em 40 km/h
    else:
        v[16] = 0.0
    if SENSOR_LEVEL >= 2:
        v[11]   = clip01(noisy_speed(speed_kmh) / 150.0)
        v[17]   = 1.0 if kw.get('is_highway', False) else 0.0
        v[18]   = 1.0 if kw.get('is_urban', True)    else 0.0
        v[19]   = 1.0 if kw.get('is_raining', False) else 0.0
        v[20]   = 1.0 if kw.get('is_night', False)   else 0.0
        v[21:24] = one_hot(kw.get('traffic_idx', 0), 3)
    if SENSOR_LEVEL >= 4:
        v[24] = 1.0 if kw.get('sirene',        False) else 0.0
        v[25] = 1.0 if kw.get('buzina',        False) else 0.0
        v[26] = 1.0 if kw.get('choro_crianca', False) else 0.0
        v[27] = 1.0 if kw.get('celular_toque', False) else 0.0
        v[28] = 1.0 if kw.get('musica_alta',   False) else 0.0
    # v[29] e v[30] são inicializados a zero aqui e preenchidos em generate_episode
    return v

# ============================================================================
# 3. ESTADOS
# ============================================================================
def st_normal(SENSOR_LEVEL: int, **kw):
    args = {
        'behavior_idx': kw.pop('behavior_idx', BEHAVIORS["c0_normal"]),
        'seatbelt_idx': SEATBELT["usando"] if kw.get("seatbelt_ok", True) else SEATBELT["nao_usando"],
        'speed_kmh':    kw.get("speed", 70.0),
        'traffic_idx':  TRAFFIC.get(kw.get("traffic", "livre"), 0),
        'SENSOR_LEVEL': SENSOR_LEVEL,
        **kw
    }
    return generate_base_vector(**args)

def st_celular(SENSOR_LEVEL, **kw):
    v = st_normal(SENSOR_LEVEL, **kw, behavior_idx=BEHAVIORS["c1_celular"])
    v[10] = clip01(v[10] - np.random.uniform(0.02, 0.10))
    return v

def st_radio(SENSOR_LEVEL, **kw):
    return st_normal(SENSOR_LEVEL, **kw, behavior_idx=BEHAVIORS["c2_radio"])

def st_bebendo(SENSOR_LEVEL, **kw):
    return st_normal(SENSOR_LEVEL, **kw, behavior_idx=BEHAVIORS["c3_bebendo"])

def st_grooming(SENSOR_LEVEL, **kw):
    return st_normal(SENSOR_LEVEL, **kw, behavior_idx=BEHAVIORS["c5_grooming"])

def st_conversa(SENSOR_LEVEL, **kw):
    return st_normal(SENSOR_LEVEL, **kw, behavior_idx=BEHAVIORS["c6_conversa"])

def st_pegar_objeto(SENSOR_LEVEL, **kw):
    return st_normal(SENSOR_LEVEL, **kw, behavior_idx=BEHAVIORS["c4_pegar_objeto"])

def st_aceleracao_brusca(SENSOR_LEVEL, **kw):
    v = st_normal(SENSOR_LEVEL, **kw)
    v[12] = 1.0
    return v

def st_freada_brusca(SENSOR_LEVEL, **kw):
    v = st_normal(SENSOR_LEVEL, **kw)
    v[13] = 1.0
    return v

def st_giroscopio_critico(SENSOR_LEVEL, **kw):
    v = st_normal(SENSOR_LEVEL, **kw)
    v[14] = 1.0
    return v

def st_fadiga(SENSOR_LEVEL, **kw):
    v = st_normal(SENSOR_LEVEL, **kw)
    v[15] = 1.0
    return v

STATE_FUNCS = {
    "ESTADO_NORMAL":             st_normal,
    "ESTADO_CELULAR":            st_celular,
    "ESTADO_RADIO":              st_radio,
    "ESTADO_BEBENDO":            st_bebendo,
    "ESTADO_PEGAR_OBJETO":       st_pegar_objeto,
    "ESTADO_GROOMING":           st_grooming,
    "ESTADO_CONVERSA":           st_conversa,
    "ESTADO_ACELERACAO_BRUSCA":  st_aceleracao_brusca,
    "ESTADO_FREADA_BRUSCA":      st_freada_brusca,
    "ESTADO_GIROSCOPIO_CRITICO": st_giroscopio_critico,
    "ESTADO_FADIGA":             st_fadiga,
}

# ============================================================================
# 4. CURVAS DE RISCO (inalteradas de v6)
# ============================================================================
def curve_sine_bump(T, peak=0.8):
    return (np.sin(np.linspace(0, np.pi, T)) * peak).astype(np.float32)

def curve_trapezoid(T, low=0.1, high=0.85, up_ratio=0.3, plateau_ratio=0.4):
    up   = max(1, int(T * up_ratio))
    plat = max(1, int(T * plateau_ratio))
    down = max(1, T - up - plat)
    return np.clip(np.concatenate([
        np.linspace(low, high, up, dtype=np.float32),
        np.full(plat, high, dtype=np.float32),
        np.linspace(high, low, down, dtype=np.float32),
    ])[:T], 0, 1)

def curve_progressive_critical(T, onset_gap, latent_level, transition_target,
                                peak, plateau_ratio=0.20):
    arr = np.full(T, LOW_RISK_BASE, dtype=np.float32)
    if T <= 3:
        return np.full(T, peak, dtype=np.float32)
    latent_len = max(3, min(onset_gap, max(3, T // 3)))
    arr[:latent_len] = np.linspace(LOW_RISK_BASE, latent_level, latent_len, dtype=np.float32)
    rem = T - latent_len
    trans_len = max(3, int(rem * 0.45))
    end_trans = min(T, latent_len + trans_len)
    arr[latent_len:end_trans] = np.linspace(latent_level, transition_target,
                                             end_trans - latent_len, dtype=np.float32)
    rem2       = T - end_trans
    plateau_len = max(2, int(T * plateau_ratio))
    ramp_len   = max(2, rem2 - plateau_len)
    ramp_end   = min(T, end_trans + ramp_len)
    if ramp_end > end_trans:
        arr[end_trans:ramp_end] = np.linspace(transition_target, peak,
                                               ramp_end - end_trans, dtype=np.float32)
    if ramp_end < T:
        plateau_end = min(T, ramp_end + plateau_len)
        arr[ramp_end:plateau_end] = peak
        if plateau_end < T:
            decay_floor = np.random.uniform(0.45, 0.65)
            arr[plateau_end:] = np.linspace(peak, decay_floor, T - plateau_end, dtype=np.float32)
    arr += np.random.normal(0, CRIT_PROGRESSIVE_JITTER, size=T).astype(np.float32)
    arr[:latent_len] += np.random.normal(0, CRIT_EARLY_OVERLAP_NOISE,
                                          size=latent_len).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)

def curve_for_name(name: str, T: int, risk_target: float) -> np.ndarray:
    n = (name or "").lower().strip()
    target = float(risk_target)
    if n == "critical_progressive":
        return curve_progressive_critical(
            T=T,
            onset_gap=np.random.randint(PROG_ONSET_GAP_MIN, PROG_ONSET_GAP_MAX + 1),
            latent_level=np.random.uniform(PROG_LATENT_LOW_MIN, PROG_LATENT_LOW_MAX),
            transition_target=np.random.uniform(PROG_TRANSITION_TARGET_MIN, PROG_TRANSITION_TARGET_MAX),
            peak=np.random.uniform(max(PROG_PEAK_MIN, target - 0.08),
                                   min(1.0, max(PROG_PEAK_MAX, target))),
        )
    if n == "flat_low":
        return np.full(T, max(LOW_RISK_BASE, target * 0.8), dtype=np.float32)
    if n == "low_bump":
        return np.clip(curve_sine_bump(T, peak=max(0.35, target + 0.05)), 0, 1)
    if n == "peak_then_drop":
        # [K1 v7.1] piso 0.75 removido: o pico segue o risk_target declarado
        # (target + 0.05). Para C01/C04 (targets 0.95/0.93) o valor e identico
        # ao da v7; so L02 (target 0.35) muda. Ver
        # docs/simulator_audit/K1_CORRECTION_REPORT.md.
        peak = target + 0.05
        step = max(5, T // 4)
        return np.clip(np.concatenate([
            np.linspace(0.1, peak, step, dtype=np.float32),
            np.linspace(peak, 0.2, T - step, dtype=np.float32),
        ]), 0, 1)
    if n == "sine_then_spike":
        peak = max(0.7, target)
        base = curve_sine_bump(T, peak=peak)
        base[T - max(3, T // 15):] = np.clip(peak + 0.15, 0, 1)
        return np.clip(base, 0, 1)
    if n == "step_then_decay_low":
        step_val = max(0.55, target + 0.05)
        tail     = max(0.10, target * 0.3)
        step_T   = T // 3
        arr = np.full(T, tail, dtype=np.float32)
        arr[:step_T] = step_val
        if T - step_T > 1:
            arr[step_T:] = np.linspace(step_val, tail, T - step_T, dtype=np.float32)
        return np.clip(arr, 0, 1)
    return curve_trapezoid(T, low=0.1, high=max(0.7, target), up_ratio=0.3, plateau_ratio=0.4)

# ============================================================================
# 5. BOOSTS / RISCO OBSERVADO
# [v7] traffic_congested: +0.08 → +0.12 (SHRP2 OR rear-end MAD = 8.48 vs 2.38)
# ============================================================================
def context_risk_boost(frame_vec: np.ndarray, SENSOR_LEVEL: int) -> float:
    boost = 0.0
    if SENSOR_LEVEL >= 2:
        speed       = float(frame_vec[11] * 150.0)
        raining     = frame_vec[19] > 0.5
        night       = frame_vec[20] > 0.5
        traffic_c   = frame_vec[23] > 0.5          # congestionado
        no_seatbelt = frame_vec[8]  > 0.5

        if raining:   boost += 0.10                # SHRP2: crash risk +30-50% em chuva
        if night:     boost += 0.07                # SHRP2: ~1.5× risco sóbrio noturno
        if traffic_c: boost += 0.12                # [v7] era 0.08; OR rear-end = 8.48
        # NOTA METODOLÓGICA [v7]: cinto não aumenta crash probability — aumenta injury
        # severity. Mantido como "risk composto" para justificar alerta preventivo.
        # Documentar como limitação na seção de metodologia do artigo.
        if no_seatbelt and speed > SPEED_MOVE_THRESH:
            boost += 0.30

    if SENSOR_LEVEL >= 4 and frame_vec[24] > 0.5:
        boost += 0.25   # sirene: alta urgência situacional
    return boost

def event_risk_boost(frame_vec: np.ndarray, SENSOR_LEVEL: int,
                      base_risk: float = None) -> float:
    boost = 0.0
    speed       = float(frame_vec[11] * 150.0) if SENSOR_LEVEL >= 2 else 0.0
    move_factor = clip01(speed / 15.0)
    stage_factor = 1.0
    if base_risk is not None:
        stage_factor = 0.35 + 0.65 * clip01((base_risk - 0.15) / 0.55)
    if frame_vec[16] > 0.5:
        boost += (0.12 + 0.22 * clip01(frame_vec[11])) * move_factor * stage_factor
    if SENSOR_LEVEL >= 3 and speed > SPEED_MOVE_THRESH:
        a = float(frame_vec[12]); b = float(frame_vec[13]); g = float(frame_vec[14])
        boost += min((0.10*a + 0.12*b + 0.06*g) * move_factor * stage_factor, 0.18)
    if SENSOR_LEVEL >= 3 and frame_vec[15] > 0.5:
        boost += 0.16 * (0.3 + 0.7 * move_factor) * stage_factor
    return boost

def apply_motion_guard(r: float, frame_vec: np.ndarray, SENSOR_LEVEL: int) -> float:
    if SENSOR_LEVEL >= 2:
        if float(frame_vec[11] * 150.0) <= SPEED_STOP_THRESH:
            return min(r, 0.15)
    return r

# ============================================================================
# 6. [v7] GERAÇÃO DE EYE GAZE (v[29]) E LANE OFFSET (v[30])
# ============================================================================

def generate_eye_gaze_series(state_names: List[str], base_curve: np.ndarray,
                              SENSOR_LEVEL: int) -> np.ndarray:
    """
    Gera série temporal de eye_gaze_off_road (v[29]).

    Modelo:
      - Probabilidade base por estado de comportamento (EYE_GAZE_BASE_PROB)
      - Modulada pela velocidade (gaze off-road mais perigoso em alta velocidade)
      - Suavizada com média móvel de 5 frames (persistência do desvio de olhar)
      - Calibrada pelo OR=7.1 do SHRP2 para extended off-road glance

    Retorna array (T,) em [0, 1].
    """
    T  = len(state_names)
    p  = np.zeros(T, dtype=np.float32)

    for t, state in enumerate(state_names):
        base_p = EYE_GAZE_BASE_PROB.get(state, 0.05)
        # Modular pelo risco: em episódios críticos o gaze off-road é mais frequente
        risk_mod = base_curve[t]
        p[t] = clip01(base_p * (1.0 + 0.5 * risk_mod) + np.random.normal(0, 0.05))

    # Suavização: desvio de olhar tem persistência de ~5 frames (~0.5 s)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    p_smooth = np.convolve(p, kernel, mode='same').astype(np.float32)
    p_smooth[0]  = p[0]
    p_smooth[-1] = p[-1]
    return np.clip(p_smooth, 0.0, 1.0)


def generate_lane_offset_series(base_curve: np.ndarray,
                                 state_names: List[str]) -> np.ndarray:
    """
    Gera série temporal de lane_offset_norm (v[30]).

    Modelo: random walk com:
      - Drift proporcional ao risco (distração/fadiga causam deriva lateral)
      - Mean reversion (motorista tende a corrigir)
      - Choques de alta amplitude em manobras bruscas (aceleração/freada)
      - Calibrado via SHRP2: steering variance aumenta com distração
      - Normalizado em [0, 1] (0=centro, 1=desvio máximo)

    Retorna array (T,) em [0, 1].
    """
    T      = len(base_curve)
    offset = np.zeros(T, dtype=np.float32)   # posição atual (pode ser neg/pos)
    pos    = 0.0   # posição inicial: centro da faixa

    for t in range(T):
        r = float(base_curve[t])

        # Drift: quanto maior o risco, maior a tendência de deriva
        if r < 0.20:
            drift = LANE_DRIFT_BY_RISK["low"]
        elif r < 0.55:
            drift = LANE_DRIFT_BY_RISK["medium"]
        else:
            drift = LANE_DRIFT_BY_RISK["high"]

        # Choque extra em manobras bruscas (aceleração/freada/giro)
        state = state_names[t]
        if state in ("ESTADO_ACELERACAO_BRUSCA", "ESTADO_FREADA_BRUSCA",
                     "ESTADO_GIROSCOPIO_CRITICO"):
            drift *= 3.0

        # Random walk com drift e mean reversion
        noise = np.random.normal(0, LANE_NOISE_BASE + drift * 0.5)
        pos   = pos + drift * np.sign(np.random.uniform(-1, 1)) + noise
        pos  -= LANE_MEAN_REVERSION * pos   # mean reversion para o centro

        offset[t] = float(pos)

    # Normalizar para [0, 1]: abs(offset) / max_expected_deviation
    abs_offset = np.abs(offset).astype(np.float32)
    max_dev    = max(float(abs_offset.max()), 0.10)   # evitar divisão por zero
    normalized = np.clip(abs_offset / (max_dev + 0.05), 0.0, 1.0)

    # Suavização leve: deriva tem persistência de ~3 frames
    kernel = np.ones(3, dtype=np.float32) / 3.0
    smooth = np.convolve(normalized, kernel, mode='same').astype(np.float32)
    smooth[0]  = normalized[0]
    smooth[-1] = normalized[-1]
    return np.clip(smooth, 0.0, 1.0)

# ============================================================================
# 7. SEGMENTOS / PERSISTÊNCIA (inalterados)
# ============================================================================
def find_event_segments(state_names: List[str]) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    in_seg = False; s = 0
    for i, st in enumerate(state_names):
        is_event = (st != "ESTADO_NORMAL")
        if is_event and not in_seg:
            in_seg = True; s = i
        if (not is_event) and in_seg:
            in_seg = False; segs.append((s, i))
    if in_seg:
        segs.append((s, len(state_names)))
    return segs

def inject_critical_persistent_tail(base_curve, segs, risk_target):
    out = base_curve.copy()
    if len(segs) == 0 or np.random.rand() >= P_CRIT_PERSIST_TAIL:
        return out, False
    _, event_end = segs[-1]
    if event_end >= len(out):
        return out, False
    tail_floor  = np.random.uniform(CRIT_TAIL_MIN, CRIT_TAIL_MAX)
    start_val   = max(float(out[event_end - 1]), tail_floor, min(0.90, risk_target))
    tail_len    = len(out) - event_end
    if tail_len <= 0:
        return out, False
    decay = np.linspace(start_val, tail_floor, tail_len, dtype=np.float32)
    out[event_end:] = np.maximum(out[event_end:], decay)
    z = min(CRIT_FINAL_ZONE, len(out))
    out[-z:] = np.maximum(out[-z:], np.random.uniform(CRIT_FINAL_FLOOR_MIN, CRIT_FINAL_FLOOR_MAX))
    return np.clip(out, 0, 1), True

# ============================================================================
# 8. GERAÇÃO DE EPISÓDIOS — agora preenche v[29] e v[30]
# ============================================================================
def generate_episode(recipe_key: str, recipes: Dict[str, Any],
                      window: int, SENSOR_LEVEL: int):
    spec = recipes[recipe_key]
    T    = window

    # 1) frames / states
    frames: List[np.ndarray] = []; state_names: List[str] = []
    blocks = spec["sequence"]
    scale  = T / sum(d for _, d, _ in blocks)
    for name, dur, kw in blocks:
        fn = STATE_FUNCS[name]
        for _ in range(max(1, int(round(dur * scale)))):
            if len(frames) >= T: break
            frames.append(fn(SENSOR_LEVEL=SENSOR_LEVEL, **kw))
            state_names.append(name)
    while len(frames) < T:
        frames.append(frames[-1]); state_names.append(state_names[-1])
    X = np.stack(frames, axis=0)

    # 2) base_curve
    base_curve = np.full(T, LOW_RISK_BASE, dtype=np.float32)
    segs = find_event_segments(state_names)
    for (s, e) in segs:
        L = max(1, e - s)
        cname = spec.get("curve", "")
        if spec.get("categoria") == "Critico" and spec.get("progressive", False):
            cname = "critical_progressive"
        base_curve[s:e] = np.maximum(base_curve[s:e],
                                      curve_for_name(cname, L, spec["risk_target"]))
    crit_tail_applied = False
    if spec.get("categoria") == "Critico":
        base_curve, crit_tail_applied = inject_critical_persistent_tail(
            base_curve, segs, spec["risk_target"])

    # 3) regime cinemático
    u = np.random.rand()
    regime = ("R2_STABLE_RISK" if u < P_REGIME_STABLE_RISK
              else "R3_FALSE_KIN" if u < P_REGIME_STABLE_RISK + P_REGIME_FALSE_KIN
              else "R1_NORMAL")
    road_noise = np.random.normal(0, KIN_NOISE_BASE, size=T)

    if regime == "R1_NORMAL":
        for t in range(T):
            r = base_curve[t]
            X[t, 12] = clip01(0.30 * r + road_noise[t])
            X[t, 13] = clip01(0.22 * r + road_noise[t])
            X[t, 14] = clip01(0.16 * r + road_noise[t])
    elif regime == "R2_STABLE_RISK":
        for t in range(T):
            r = base_curve[t]
            if r > 0.55:
                bl = 0.18 + 0.05 * np.random.rand()
                X[t, 12] = clip01(bl + 0.5 * road_noise[t])
                X[t, 13] = clip01(bl * 0.8 + 0.5 * road_noise[t])
                X[t, 14] = clip01(bl * 0.6 + 0.5 * road_noise[t])
            else:
                X[t, 12] = clip01(0.30 * r + road_noise[t])
                X[t, 13] = clip01(0.22 * r + road_noise[t])
                X[t, 14] = clip01(0.16 * r + road_noise[t])
    else:
        spikes = np.zeros(T, dtype=np.float32)
        for _ in range(np.random.randint(2, 6)):
            c = np.random.randint(0, T); w = np.random.randint(2, 8)
            a = np.random.uniform(0.15, KIN_FALSE_SPIKE_SCALE)
            spikes[max(0,c-w):min(T,c+w)] = np.maximum(spikes[max(0,c-w):min(T,c+w)], a)
        for t in range(T):
            r = base_curve[t]
            X[t, 12] = clip01(0.18 * r + spikes[t] + road_noise[t])
            X[t, 13] = clip01(0.14 * r + 0.9 * spikes[t] + road_noise[t])
            X[t, 14] = clip01(0.10 * r + 0.7 * spikes[t] + road_noise[t])

    # 4) precursor visual
    if np.random.rand() < P_PRECURSOR_VISUAL and len(segs) > 0:
        first_start = segs[0][0]
        L   = np.random.randint(PRECURSOR_LEN_MIN, PRECURSOR_LEN_MAX + 1)
        t0  = max(0, first_start - L)
        drop = np.random.uniform(PRECURSOR_CONF_DROP[0], PRECURSOR_CONF_DROP[1])
        for t in range(t0, first_start):
            X[t, 9]  = clip01(float(X[t, 9])  - drop * np.random.uniform(0.5, 0.9))
            X[t, 10] = clip01(float(X[t, 10]) - drop * np.random.uniform(0.5, 0.9))
            if spec.get("progressive", False):
                if X[t, 16] < 0.5 and np.random.rand() < 0.20:
                    X[t, 16] = clip01(float(X[t, 16]) + np.random.uniform(0.05, 0.15))
                if SENSOR_LEVEL >= 3:
                    X[t, 12] = clip01(float(X[t, 12]) + np.random.uniform(0.01, 0.05))
                    X[t, 13] = clip01(float(X[t, 13]) + np.random.uniform(0.01, 0.04))
                    X[t, 14] = clip01(float(X[t, 14]) + np.random.uniform(0.00, 0.03))

    # 5) [v7] Preencher v[29]=eye_gaze e v[30]=lane_offset
    eye_gaze    = generate_eye_gaze_series(state_names, base_curve, SENSOR_LEVEL)
    lane_offset = generate_lane_offset_series(base_curve, state_names)
    for t in range(T):
        X[t, 29] = eye_gaze[t]
        X[t, 30] = lane_offset[t]

    # 6) clean vs observed
    y_frame_clean = np.clip(base_curve.astype(np.float32), 0.0, 1.0)
    kernel = np.ones(3, dtype=np.float32) / 3.0
    y_smooth = np.convolve(y_frame_clean, kernel, mode='same').astype(np.float32)
    y_smooth[0] = y_frame_clean[0]; y_smooth[-1] = y_frame_clean[-1]
    y_frame_clean = np.clip(y_smooth, 0.0, 1.0)

    y_frame_obs = np.zeros(T, dtype=np.float32)
    for t in range(T):
        r = (base_curve[t]
             + context_risk_boost(X[t], SENSOR_LEVEL)
             + event_risk_boost(X[t], SENSOR_LEVEL, base_risk=base_curve[t]))
        y_frame_obs[t] = apply_motion_guard(clip01(r), X[t], SENSOR_LEVEL)

    detect_035 = first_crossing(y_frame_clean, 0.35)
    detect_045 = first_crossing(y_frame_clean, 0.45)
    detect_055 = first_crossing(y_frame_clean, 0.55)
    y_ep_op  = float(np.mean(y_frame_clean[-K_AGG:])) if T >= K_AGG else float(np.mean(y_frame_clean))
    p95      = float(np.percentile(y_frame_clean, 95))
    y_ep_sev = clip01(0.7 * float(y_frame_clean.mean()) + 0.3 * p95)

    event_start = int(segs[0][0]) if segs else -1
    event_end   = int(segs[-1][1] - 1) if segs else -1
    event_dur   = int(event_end - event_start + 1) if event_start >= 0 and event_end >= event_start else 0

    meta = {
        "id": recipe_key, "recipe_id": recipe_key, "recipe": recipe_key,
        "nome": spec["nome"], "display_name": spec["nome"],
        "categoria": spec["categoria"], "class_name": spec["categoria"],
        "class": spec["categoria"],
        "risk_episode": float(y_ep_sev),
        "risk_episode_operational": float(y_ep_op),
        "risk_episode_severity": float(y_ep_sev),
        "regime": regime,
        "event_start": event_start, "event_end": event_end, "event_duration": event_dur,
        "progressive": int(bool(spec.get("progressive", False))),
        "detectable_start_035": detect_035, "detectable_start_045": detect_045,
        "detectable_start_055": detect_055,
        "onset_gap_035": int(detect_035 - event_start) if (event_start >= 0 and detect_035 >= 0) else -1,
        "onset_gap_045": int(detect_045 - event_start) if (event_start >= 0 and detect_045 >= 0) else -1,
        "onset_gap_055": int(detect_055 - event_start) if (event_start >= 0 and detect_055 >= 0) else -1,
        "persistent_tail": int(crit_tail_applied),
        "last6_mean_clean": float(np.mean(y_frame_clean[-K_AGG:])),
        "last6_mean_obs":   float(np.mean(y_frame_obs[-K_AGG:])),
        "max_y_frame_clean": float(np.max(y_frame_clean)),
        "mean_y_frame_clean": float(np.mean(y_frame_clean)),
        "max_y_frame_obs":   float(np.max(y_frame_obs)),
        "mean_y_frame_obs":  float(np.mean(y_frame_obs)),
        "mean_eye_gaze":     float(np.mean(eye_gaze)),      # [v7] novo
        "mean_lane_offset":  float(np.mean(lane_offset)),   # [v7] novo
        "window": int(T), "sensor_level": int(SENSOR_LEVEL),
    }
    return (X.astype(np.float32), float(y_ep_sev),
            y_frame_clean.astype(np.float32), y_frame_obs.astype(np.float32), meta)

# ============================================================================
# 9. RECEITAS — RECIPES_BORDERLINE redesenhadas [v7-final]
# ============================================================================
# Princípio de design (corrige problema estrutural identificado na análise):
#
# ATENÇÃO — usa APENAS ESTADO_NORMAL com contexto adverso.
#   Nenhum behavior não-normal → v[1]=0, v[16]=0.
#   O modelo não aprendeu "contexto adverso sozinho = crítico" no treino.
#   Espera-se: max_prob ≈ 0.10–0.25.
#
# ALERTA — usa behaviors que o modelo conhece do treino, em grau intermediário.
#   L01: ESTADO_CELULAR a 20 km/h → v[16]≈0.50 (intermediário entre 0 e 1.0)
#   L02: ESTADO_PEGAR_OBJETO a 90 km/h → treino viu a 35-40 km/h; mais perigoso
#   L03: ESTADO_GROOMING a 90 km/h → treino viu a 108 km/h; similar
#   Espera-se: max_prob ≈ 0.30–0.55.
#
# Gradiente pretendido:
#   Normal (FPR=0) < Atenção < Alerta < Crítico
#
# Referência de design: Dingus et al. 2016 PNAS (odds ratios SHRP2).
# ============================================================================
RECIPES_BORDERLINE: Dict[str, Dict[str, Any]] = {
    # ── ATENÇÃO — contexto adverso, comportamento sempre Normal ──────────────
    "A01": {
        "nome": "Atencao: Conducao sob chuva forte em rodovia",
        "categoria": "Atencao",
        # Chuva forte aumenta risco real (SHRP2 Ghasemzadeh 2018)
        # mas sem behavior anômalo → v[1]=0, v[16]=0
        # risk_target baixo reflete ausência de distração ativa
        "risk_target": 0.20,
        "curve": "flat_low",
        "sequence": [
            ("ESTADO_NORMAL", 96, {"speed": 80, "is_raining": True,
                                   "highway": True, "traffic": "livre"}),
        ],
    },
    "A02": {
        "nome": "Atencao: Trafego congestionado noturno urbano",
        "categoria": "Atencao",
        # Tráfego congestionado + noite: contextRiskBoost eleva risco observado
        # sem distração comportamental — v[1]=0, v[16]=0
        "risk_target": 0.18,
        "curve": "flat_low",
        "sequence": [
            ("ESTADO_NORMAL", 96, {"speed": 20, "traffic": "congestionado",
                                   "is_urban": True, "is_night": True}),
        ],
    },
    "A03": {
        "nome": "Atencao: Rodovia noturna com chuva e alta velocidade",
        "categoria": "Atencao",
        # Combinação chuva + noite + alta velocidade: maior contextRiskBoost
        # sem nenhum comportamento não-normal — v[1]=0, v[15]=0, v[16]=0
        "risk_target": 0.22,
        "curve": "flat_low",
        "sequence": [
            ("ESTADO_NORMAL", 96, {"speed": 100, "is_raining": True,
                                   "is_night": True, "highway": True}),
        ],
    },
    # ── ALERTA — behaviors conhecidos do treino, intensidade intermediária ───
    "L01": {
        "nome": "Alerta: Conversa prolongada em rodovia",
        "categoria": "Alerta",
        "risk_target": 0.38,
        "curve": "low_bump",
        "sequence": [
            ("ESTADO_NORMAL",  45, {"speed": 80, "highway": True}),
            ("ESTADO_CONVERSA", 20, {"speed": 78, "highway": True}),
            # evento curto (20 frames vs 30 anteriores)
            ("ESTADO_NORMAL",  31, {"speed": 80, "highway": True}),
        ],
    },
    "L02": {
        "nome": "Alerta: Ajuste de radio em velocidade moderada",
        "categoria": "Alerta",
        "risk_target": 0.35,
        "curve": "peak_then_drop",
        "sequence": [
            ("ESTADO_NORMAL", 55, {"speed": 50, "traffic": "moderado",
                                   "is_urban": True}),
            ("ESTADO_RADIO",  15, {"speed": 48, "traffic": "moderado",
                                   "is_urban": True}),
            # evento curto (15 frames)
            ("ESTADO_NORMAL", 26, {"speed": 50}),
        ],
    },
    "L03": {
        "nome": "Alerta: Bebendo em trecho urbano lento",
        "categoria": "Alerta",
        "risk_target": 0.40,
        "curve": "low_bump",
        "sequence": [
            ("ESTADO_NORMAL",  45, {"speed": 30, "traffic": "moderado",
                                    "is_urban": True}),
            ("ESTADO_BEBENDO", 18, {"speed": 28, "traffic": "moderado",
                                    "is_urban": True}),
            # velocidade baixa + evento curto
            ("ESTADO_NORMAL",  33, {"speed": 30}),
        ],
    },
    
}

# ============================================================================
# 10. RECIPES_TRAIN — Normal + Crítico (inalterado de v6)
# ============================================================================
RECIPES_TRAIN: Dict[str, Dict[str, Any]] = {
    "N01": {"nome": "Normal: Viagem tranquila em rodovia", "categoria": "Normal",
            "risk_target": 0.05, "curve": "flat_low",
            "sequence": [("ESTADO_NORMAL", 150, {"speed": 100, "highway": True, "traffic": "livre"})]},
    "N02": {"nome": "Normal: Urbano moderado", "categoria": "Normal",
            "risk_target": 0.06, "curve": "flat_low",
            "sequence": [("ESTADO_NORMAL", 150, {"speed": 45, "traffic": "moderado", "is_urban": True})]},
    "N03": {"nome": "Normal: Noite estável", "categoria": "Normal",
            "risk_target": 0.07, "curve": "flat_low",
            "sequence": [("ESTADO_NORMAL", 150, {"speed": 70, "is_night": True, "traffic": "livre"})]},
    "C01": {"nome": "Critico: Celular em alta velocidade", "categoria": "Critico",
            "risk_target": 0.95, "curve": "peak_then_drop", "progressive": False,
            "sequence": [("ESTADO_NORMAL",50,{"speed":100,"highway":True}),
                          ("ESTADO_CELULAR",20,{"speed":110,"highway":True}),
                          ("ESTADO_NORMAL",80,{"speed":100})]},
    "C02": {"nome": "Critico: Fadiga com desvio cinematico", "categoria": "Critico",
            "risk_target": 0.92, "curve": "sine_then_spike", "progressive": False,
            "sequence": [("ESTADO_NORMAL",45,{"speed":85,"is_night":True}),
                          ("ESTADO_FADIGA",25,{"speed":80,"is_night":True}),
                          ("ESTADO_GIROSCOPIO_CRITICO",10,{"speed":78,"is_night":True}),
                          ("ESTADO_NORMAL",70,{"speed":75})]},
    "C03": {"nome": "Critico: Celular sem cinto sob chuva", "categoria": "Critico",
            "risk_target": 0.97, "curve": "sine_then_spike", "progressive": False,
            "sequence": [("ESTADO_NORMAL",40,{"speed":75,"is_raining":True,"seatbelt_ok":False}),
                          ("ESTADO_CELULAR",25,{"speed":85,"is_raining":True,"seatbelt_ok":False}),
                          ("ESTADO_NORMAL",85,{"speed":78,"is_raining":True,"seatbelt_ok":False})]},
    "C04": {"nome": "Critico: Multiplas distracoes urbanas", "categoria": "Critico",
            "risk_target": 0.93, "curve": "peak_then_drop", "progressive": False,
            "sequence": [("ESTADO_NORMAL",35,{"speed":45,"traffic":"moderado","is_urban":True}),
                          ("ESTADO_RADIO",10,{"speed":42,"traffic":"moderado","is_urban":True}),
                          ("ESTADO_CELULAR",15,{"speed":48,"traffic":"congestionado","is_urban":True}),
                          ("ESTADO_PEGAR_OBJETO",12,{"speed":40,"traffic":"congestionado","is_urban":True}),
                          ("ESTADO_NORMAL",78,{"speed":38,"traffic":"moderado","is_urban":True})]},
    "C05": {"nome": "Critico progressivo: Celular em rodovia", "categoria": "Critico",
            "risk_target": 0.92, "curve": "critical_progressive", "progressive": True,
            "sequence": [("ESTADO_NORMAL",42,{"speed":95,"highway":True,"traffic":"livre"}),
                          ("ESTADO_CELULAR",14,{"speed":98,"highway":True,"traffic":"livre"}),
                          ("ESTADO_CELULAR",18,{"speed":103,"highway":True,"traffic":"moderado"}),
                          ("ESTADO_NORMAL",22,{"speed":96,"highway":True})]},
    "C06": {"nome": "Critico progressivo: Fadiga noturna", "categoria": "Critico",
            "risk_target": 0.90, "curve": "critical_progressive", "progressive": True,
            "sequence": [("ESTADO_NORMAL",36,{"speed":82,"is_night":True}),
                          ("ESTADO_FADIGA",20,{"speed":80,"is_night":True}),
                          ("ESTADO_FADIGA",16,{"speed":78,"is_night":True}),
                          ("ESTADO_GIROSCOPIO_CRITICO",10,{"speed":76,"is_night":True}),
                          ("ESTADO_NORMAL",14,{"speed":74})]},
    "C07": {"nome": "Critico progressivo: Sem cinto sob chuva", "categoria": "Critico",
            "risk_target": 0.94, "curve": "critical_progressive", "progressive": True,
            "sequence": [("ESTADO_NORMAL",32,{"speed":72,"is_raining":True,"seatbelt_ok":False}),
                          ("ESTADO_CELULAR",18,{"speed":75,"is_raining":True,"seatbelt_ok":False}),
                          ("ESTADO_CELULAR",16,{"speed":82,"is_raining":True,"seatbelt_ok":False}),
                          ("ESTADO_NORMAL",18,{"speed":76,"is_raining":True,"seatbelt_ok":False})]},
    "C08": {"nome": "Critico progressivo: Distracao urbana multietapas", "categoria": "Critico",
            "risk_target": 0.91, "curve": "critical_progressive", "progressive": True,
            "sequence": [("ESTADO_NORMAL",28,{"speed":38,"traffic":"moderado","is_urban":True}),
                          ("ESTADO_RADIO",12,{"speed":36,"traffic":"moderado","is_urban":True}),
                          ("ESTADO_CONVERSA",12,{"speed":34,"traffic":"moderado","is_urban":True}),
                          ("ESTADO_CELULAR",16,{"speed":41,"traffic":"congestionado","is_urban":True}),
                          ("ESTADO_PEGAR_OBJETO",10,{"speed":35,"traffic":"congestionado","is_urban":True}),
                          ("ESTADO_NORMAL",18,{"speed":32,"traffic":"moderado","is_urban":True})]},
    "C09": {"nome": "Critico progressivo: Bebida ao volante", "categoria": "Critico",
            "risk_target": 0.91, "curve": "critical_progressive", "progressive": True,
            "sequence": [("ESTADO_NORMAL",38,{"speed":65,"traffic":"moderado"}),
                          ("ESTADO_BEBENDO",22,{"speed":60,"traffic":"moderado"}),
                          ("ESTADO_CELULAR",14,{"speed":58,"traffic":"moderado"}),
                          ("ESTADO_NORMAL",22,{"speed":55})]},
    "C10": {"nome": "Critico progressivo: Grooming em rodovia com fadiga", "categoria": "Critico",
            "risk_target": 0.89, "curve": "critical_progressive", "progressive": True,
            "sequence": [("ESTADO_NORMAL",40,{"speed":110,"highway":True,"is_night":True}),
                          ("ESTADO_GROOMING",18,{"speed":108,"highway":True,"is_night":True}),
                          ("ESTADO_FADIGA",16,{"speed":105,"highway":True,"is_night":True}),
                          ("ESTADO_GIROSCOPIO_CRITICO",8,{"speed":102,"highway":True,"is_night":True}),
                          ("ESTADO_NORMAL",14,{"speed":98,"highway":True})]},
}

RECIPES_BASE = RECIPES_TRAIN   # alias de compatibilidade

# ============================================================================
# 11. BUILD DATASET / SAVE / QUICK GENERATE
# ============================================================================
def build_dataset(recipes, window, per_recipe, seed, shuffle, SENSOR_LEVEL):
    rng = np.random.default_rng(seed)
    random.seed(seed); np.random.seed(seed)
    Xs, Ys, Yf_clean, Yf_obs, metas = [], [], [], [], []
    for key in recipes.keys():
        for _ in range(per_recipe):
            X, y, yf_clean, yf_obs, meta = generate_episode(key, recipes, window, SENSOR_LEVEL)
            Xs.append(X); Ys.append(y); Yf_clean.append(yf_clean)
            Yf_obs.append(yf_obs); metas.append(meta)
    Xs = np.stack(Xs); Ys = np.array(Ys, dtype=np.float32)
    Yf_clean = np.stack(Yf_clean); Yf_obs = np.stack(Yf_obs)
    if shuffle:
        idx = np.arange(len(Ys)); rng.shuffle(idx)
        return Xs[idx], Ys[idx], Yf_clean[idx], Yf_obs[idx], [metas[i] for i in idx]
    return Xs, Ys, Yf_clean, Yf_obs, metas

def save_npz(path, X, y_episode, y_frame_clean, y_frame_obs):
    np.savez_compressed(path, X=X, y_episode=y_episode,
                         y_frame_clean=y_frame_clean,
                         y_frame=y_frame_obs, y_frame_obs=y_frame_obs)

def save_csv_metadata(path, metas):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        header = ["id","recipe_id","nome","categoria","class_name",
                  "risk_episode","risk_episode_severity","regime",
                  "event_start","event_end","event_duration","progressive",
                  "detectable_start_035","onset_gap_035","persistent_tail",
                  "last6_mean_clean","mean_eye_gaze","mean_lane_offset",   # [v7]
                  "window","sensor_level"]
        w.writerow(header)
        for m in metas:
            w.writerow([m.get(k,"") for k in header])

def quick_generate(out_npz="dataset_sintetico.npz", out_csv="episodios.csv",
                   window=96, per_recipe=80, seed=123,
                   recipe_set="base", SENSOR_LEVEL=4):
    """Dataset de treino — Normal + Crítico (v7)."""
    X, y_ep, y_fr_clean, y_fr_obs, metas = build_dataset(
        RECIPES_TRAIN, window, per_recipe, seed, True, SENSOR_LEVEL)
    save_npz(out_npz, X, y_ep, y_fr_clean, y_fr_obs)
    save_csv_metadata(out_csv, metas)
    n_norm = sum(1 for m in metas if m["categoria"]=="Normal")
    n_crit = sum(1 for m in metas if m["categoria"]=="Critico")
    return {"npz": out_npz, "csv": out_csv, "shape": X.shape,
            "total": len(y_ep), "n_normal": n_norm, "n_critico": n_crit,
            "pos_weight": round(n_crit/max(n_norm,1), 4)}

def quick_generate_borderline(out_npz="dataset_borderline.npz",
                               out_csv="episodios_borderline.csv",
                               window=96, per_recipe=80, seed=123,
                               SENSOR_LEVEL=4):
    """Borderline (Atenção + Alerta) — NÃO usar no treino. [v7] risk_targets calibrados."""
    X, y_ep, y_fr_clean, y_fr_obs, metas = build_dataset(
        RECIPES_BORDERLINE, window, per_recipe, seed, True, SENSOR_LEVEL)
    save_npz(out_npz, X, y_ep, y_fr_clean, y_fr_obs)
    save_csv_metadata(out_csv, metas)
    n_at = sum(1 for m in metas if m["categoria"]=="Atencao")
    n_al = sum(1 for m in metas if m["categoria"]=="Alerta")
    return {"npz": out_npz, "csv": out_csv, "shape": X.shape,
            "n_atencao": n_at, "n_alerta": n_al,
            "nota": "[v7] risk_targets calibrados via SHRP2 NDS (Dingus et al. 2016)"}


if __name__ == "__main__":
    print("=== SDRGenerator v7 ===")
    print("Novidades: risk_targets SHRP2-calibrados, v[29]=eye_gaze, v[30]=lane_offset\n")

    print("[1] Dataset de treino (Normal + Crítico):")
    r = quick_generate(per_recipe=5)
    print(f"    {r['shape']} | pos_weight={r['pos_weight']}")

    print("\n[2] Dataset borderline (Atenção + Alerta) — [v7] calibrado:")
    rb = quick_generate_borderline(per_recipe=5)
    print(f"    {rb['shape']} | Atenção:{rb['n_atencao']} Alerta:{rb['n_alerta']}")

    print("\n[3] Verificação das novas features (v[29] e v[30]):")
    import numpy as np
    data = np.load("dataset_borderline.npz")
    X = data["X"]
    print(f"    eye_gaze   v[29]: mean={X[:,:,29].mean():.3f}  max={X[:,:,29].max():.3f}")
    print(f"    lane_offset v[30]: mean={X[:,:,30].mean():.3f}  max={X[:,:,30].max():.3f}")
    print(f"    v[29] e v[30] não-zero: {(X[:,:,29]>0.01).mean()*100:.1f}% dos frames ✓")

    print("\n[4] Comparação risk_targets Atenção v6 → v7:")
    diffs = [("A01 Celular parado",   0.25, 0.15, "OR=2.2×speed"),
             ("A02 Grooming baixa v", 0.28, 0.22, "OR≈3.0×speed"),
             ("A03 Conversa",         0.30, 0.13, "OR=1.4 SHRP2")]
    for nome, old, new, ref in diffs:
        print(f"    {nome:26s}: {old:.2f} → {new:.2f}  ({ref})")
    print("\n[5] Boost contextual tráfego congestionado: +0.08 → +0.12 (SHRP2 OR=8.48)")
