"""E10 · arms — os quatro braços sobre o MESMO stream de frames (§2).

Todos compartilham o MHEG (edge_runtime.MHEGScheduler, mesmos τΔ/τH/max_skip)
e leem `p_obs[t]` somente quando computam. A única diferença é a camada de
decisão de alerta:

    A0 legado    — limiar frame-level + cooldown (frames) + histerese
    A1 EGA       — FSM de evento (event_guard, C8 de E9) sobre p_t corrente
    A2 EAML      — edge_runtime completo (janela W, R_E, OSS_t, force_compute)
    A3 EAML−OSSL — ablação: frames reutilizados entram como se fossem
                   computados (sem OSS_t, sem decaimento, sem force_compute)

Conversão de unidades E9→frames: 1 unidade (episódio-decisão do UCC) ≈ 1 s
≈ UNIT_FRAMES=10 frames (Δt = 10/96 s). Declarado no protocolo/spec.

A resposta do motorista usa o sorteio pré-amostrado do episódio e
q = clip(responsividade − 0,4·fadiga, 0,05, 1); a MESMA regra em todos os
braços com guarda de evento. SAR (stale-alert rate) é medido EXTERNAMENTE
para todos os braços com a mesma janela (fração de frames reutilizados nos
últimos SAR_WINDOW frames no instante da entrega) — para A3 isso captura o
que o próprio braço não consegue ver.
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from edge_runtime import EdgePolicy, EdgeRuntime, MHEGScheduler, EAML, ZState
from edge_runtime.event_guard import EventAwareGovernance

from .scenarios import Episode, SeedData, WINDOW_T, response_probability

UNIT_FRAMES = 10      # 1 unidade E9 ≈ 1 s ≈ 10 frames
SAR_WINDOW = 12       # janela p/ reused_ratio no instante do alerta (uniforme)
CONFIDENCE = 0.95     # confiança do "modelo" quando computa (§4)

_LEVELS = EdgePolicy()


def h_bin(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _mheg_policy(mheg_cfg: Dict, **eaml_kw) -> EdgePolicy:
    return EdgePolicy(
        tau_delta=mheg_cfg["tau_delta"], tau_h=mheg_cfg["tau_h"],
        max_skip=mheg_cfg["max_skip"], **eaml_kw)


def _episode_record(ep: Episode, alerts, stale, redundant, computed,
                    force_compute=0):
    return {
        "persona": ep.persona, "trip": ep.trip, "slot": ep.slot,
        "recipe": ep.recipe, "categoria": ep.categoria,
        "progressive": ep.progressive, "onset": ep.onset,
        "alerts": list(alerts), "n_alerts": len(alerts),
        "stale_alerts": stale, "redundant_alerts": redundant,
        "computed": computed, "reused": WINDOW_T - computed,
        "force_compute": force_compute,
    }


def _respond(ep: Episode, persona_alerts: int) -> bool:
    q = response_probability(ep.persona, persona_alerts)
    return ep.response_draw < q


# ---------------------------------------------------------------- A0 legado
def run_a0(seed_data: SeedData, cfg: Dict, mheg_cfg: Dict) -> List[Dict]:
    """Limiar frame-level + cooldown/histerese (lógica legada, nível de frame)."""
    theta, cooldown, hyst = cfg["theta_alert"], cfg["cooldown_frames"], cfg["hysteresis"]
    out = []
    for _, _, episodes in seed_data.trips():
        since_last, armed = 10_000, True
        for ep in episodes:
            mheg = MHEGScheduler(_mheg_policy(mheg_cfg))
            p_last, h_last = 0.0, 1.0
            win = deque(maxlen=SAR_WINDOW)
            alerts, stale, computed = [], 0, 0
            for t in range(WINDOW_T):
                d = mheg.decide(ep.dx[t], h_last)
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last)
                    computed += 1
                win.append(0 if d.compute else 1)
                since_last += 1
                if not armed and p_last < theta - hyst:
                    armed = True
                if p_last >= theta and armed and since_last >= cooldown:
                    alerts.append(t)
                    if sum(win) / len(win) > 0.60:
                        stale += 1
                    armed = False
                    since_last = 0
            out.append(_episode_record(ep, alerts, stale, 0, computed))
    return out


# ---------------------------------------------------------------- A1 EGA
def run_a1(seed_data: SeedData, cand: Dict, mheg_cfg: Dict) -> List[Dict]:
    """EGA mínima (E9/C8) por frame, guarda persistente por viagem."""
    out = []
    for persona, _, episodes in seed_data.trips():
        guard = EventAwareGovernance(
            theta_off=cand["theta_off"],
            min_realert_gap=cand["min_realert_gap"] * UNIT_FRAMES,
            max_alerts_per_trip=cand["max_alerts_per_trip"],
            escalation_delta=cand["escalation_delta"],
            end_persistence=cand["end_persistence"] * UNIT_FRAMES,
        )
        trip_alerts = 0
        for ep in episodes:
            mheg = MHEGScheduler(_mheg_policy(mheg_cfg))
            p_last, h_last = 0.0, 1.0
            win = deque(maxlen=SAR_WINDOW)
            alerts, stale, redundant, computed = [], 0, 0, 0
            for t in range(WINDOW_T):
                d = mheg.decide(ep.dx[t], h_last)
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last)
                    computed += 1
                win.append(0 if d.compute else 1)
                alert, red = guard.step(p_last, cand["theta_on"],
                                        _LEVELS.level_of(p_last))
                if alert:
                    alerts.append(t); redundant += red
                    if sum(win) / len(win) > 0.60:
                        stale += 1
                    guard.observe_response(int(_respond(ep, trip_alerts)))
                    trip_alerts += 1
            out.append(_episode_record(ep, alerts, stale, redundant, computed))
    return out


# ---------------------------------------------------------------- A2 EAML
def build_a2_policy(a2_cfg: Dict, mheg_cfg: Dict) -> EdgePolicy:
    g = a2_cfg["guard"]
    return _mheg_policy(
        mheg_cfg,
        window_w=a2_cfg["window_w"], m_persistence=a2_cfg["m_persistence"],
        theta_episode=a2_cfg["theta_episode"],
        theta_on=g["theta_on"], theta_off=g["theta_off"],
        min_realert_gap=g["min_realert_gap"] * UNIT_FRAMES,
        max_alerts_per_trip=g["max_alerts_per_trip"],
        escalation_delta=g["escalation_delta"],
        end_persistence=g["end_persistence"] * UNIT_FRAMES,
        theta_conf=0.30,
    )


def run_a2(seed_data: SeedData, a2_cfg: Dict, mheg_cfg: Dict) -> List[Dict]:
    """EAML completa via edge_runtime; guarda de evento persistente por viagem."""
    policy = build_a2_policy(a2_cfg, mheg_cfg)
    out = []
    for persona, _, episodes in seed_data.trips():
        trip_guard = EventAwareGovernance(
            theta_off=policy.theta_off,
            min_realert_gap=policy.min_realert_gap,
            max_alerts_per_trip=policy.max_alerts_per_trip,
            escalation_delta=policy.escalation_delta,
            end_persistence=policy.end_persistence,
        )
        trip_alerts = 0
        for ep in episodes:
            computed_box = [0]

            def infer_fn(t, _ep=ep, _box=computed_box):
                _box[0] += 1
                p = float(_ep.p_obs[t])
                return p, h_bin(p), CONFIDENCE

            rt = EdgeRuntime(policy, infer_fn)
            rt.eaml.guard = trip_guard
            alerts, stale, redundant, force = [], 0, 0, 0
            for t in range(WINDOW_T):
                dec = rt.process_frame(delta_x=float(ep.dx[t]))
                if dec.action == "force_compute_before_alert":
                    force += 1
                elif dec.action == "alert":
                    alerts.append(t); redundant += int(dec.redundant)
                    if dec.reused_ratio > 0.60:
                        stale += 1
                    rt.observe_response(_respond(ep, trip_alerts))
                    trip_alerts += 1
            out.append(_episode_record(ep, alerts, stale, redundant,
                                       computed_box[0], force))
    return out


# ---------------------------------------------------------------- A3 ablação
def run_a3(seed_data: SeedData, a2_cfg: Dict, mheg_cfg: Dict) -> List[Dict]:
    """EAML sem OSSL: frame reutilizado entra na janela COMO SE computado
    (p carregado vira ZState com confiança cheia; sem qualidade de supressão,
    sem decaimento, sem force_compute). O SAR é medido por fora."""
    policy = build_a2_policy(a2_cfg, mheg_cfg)
    out = []
    for persona, _, episodes in seed_data.trips():
        trip_guard = EventAwareGovernance(
            theta_off=policy.theta_off,
            min_realert_gap=policy.min_realert_gap,
            max_alerts_per_trip=policy.max_alerts_per_trip,
            escalation_delta=policy.escalation_delta,
            end_persistence=policy.end_persistence,
        )
        trip_alerts = 0
        for ep in episodes:
            mheg = MHEGScheduler(policy)
            eaml = EAML(policy, mheg=None)   # sem canal de force_compute
            eaml.guard = trip_guard
            p_last, h_last = 0.0, 1.0
            win = deque(maxlen=SAR_WINDOW)
            alerts, stale, redundant, computed = [], 0, 0, 0
            for t in range(WINDOW_T):
                d = mheg.decide(ep.dx[t], h_last)
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last)
                    computed += 1
                win.append(0 if d.compute else 1)
                # supressão invisível: sempre ZState (este é o ponto da ablação)
                dec = eaml.step(ZState(
                    t=t, p=p_last, level=policy.level_of(p_last),
                    entropy=h_last, delta_x=float(ep.dx[t]),
                    trigger_reason=d.trigger_reason, confidence=CONFIDENCE,
                    policy_id=policy.policy_id))
                if dec.action == "alert":
                    alerts.append(t); redundant += int(dec.redundant)
                    if sum(win) / len(win) > 0.60:   # SAR externo (verdadeiro)
                        stale += 1
                    eaml.observe_response(_respond(ep, trip_alerts))
                    trip_alerts += 1
            out.append(_episode_record(ep, alerts, stale, redundant, computed))
    return out


ARM_RUNNERS = {"A0": run_a0, "A1": run_a1, "A2": run_a2, "A3": run_a3}
