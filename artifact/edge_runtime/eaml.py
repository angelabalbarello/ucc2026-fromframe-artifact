"""Nível 3 · EAML — Episodic Alert Mediation Layer. ÚNICA origem da HMI.

Mantém o buffer episódico na borda (janela W) agregando z_t e OSS_t, calcula o
risco episódico R_E e decide a comunicação ao motorista (Seção 6 da
apresentação):

    Trigger_HMI = (RISCO) ∧ (GUARDA)

    RISCO (qualquer):
        pico(p) >= θ_peak  |  mean_top_k(p) >= θ_episode
        ℓ ∈ {Alert, Critical} por m frames consecutivos
        escalation_slope >= θ_slope
    GUARDA (todas):
        entrega orientada a eventos (EGA: gap, orçamento, escalada/não-resposta)
        confiança da janela >= θ_conf
        suppress_alert_noise == False
        qualidade da evidência suficiente

Proteção contra evidência reutilizada em excesso (force_compute_before_alert):
    se RISCO e reused_ratio > reused_ratio_max e last_compute_age > max_skip_safe:
        a EAML PEDE computação ao MHEG e NÃO alerta neste frame.

Invariantes:
  * nenhum alerta fora da EAML;
  * um frame Critical isolado não satisfaz, sozinho, o RISCO;
  * a EAML pode pedir computação, mas quem computa é o MHEG.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

from .event_guard import EventAwareGovernance
from .mheg import MHEGScheduler
from .ossl import OSSummary, ZState
from .policy_state import EdgePolicy

Evidence = Union[ZState, OSSummary]


def _p_of(ev: Evidence) -> float:
    return ev.p if isinstance(ev, ZState) else ev.carried_p


def _conf_of(ev: Evidence) -> float:
    return ev.confidence if isinstance(ev, ZState) else ev.confidence_decay


@dataclass(frozen=True)
class AlertDecision:
    """Saída da EAML para um frame. `action` é o único canal de HMI do sistema."""
    t: int
    action: str                    # "none" | "hold" | "force_compute_before_alert" | "alert"
    episode_risk: float            # R_E
    peak: float
    mean_top_k: float
    persistence: int
    escalation_slope: float
    reused_ratio: float
    last_compute_age: int
    confidence: float
    evidence_quality: str          # "sufficient" | "insufficient"
    risk_reasons: tuple = ()       # quais cláusulas de RISCO dispararam
    guard_blocks: tuple = ()       # quais guardas bloquearam (quando action != alert)
    redundant: bool = False
    episode_id: int = 0


class EpisodicBuffer:
    """Janela deslizante W de evidências (z_t | OSS_t). Nenhum frame é descartado
    antes de entrar aqui; a janela apenas desliza."""

    def __init__(self, window_w: int):
        self._buf: deque = deque(maxlen=window_w)

    def push(self, ev: Evidence) -> None:
        self._buf.append(ev)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def items(self) -> list:
        return list(self._buf)

    # ------------------------------------------------------------- métricas
    def peak(self) -> float:
        return max((_p_of(e) for e in self._buf), default=0.0)

    def mean_top_k(self, k: int) -> float:
        ps = sorted((_p_of(e) for e in self._buf), reverse=True)[:k]
        return sum(ps) / len(ps) if ps else 0.0

    def persistence(self, policy: EdgePolicy) -> int:
        """Frames consecutivos mais recentes com nível Alert/Critical."""
        count = 0
        for ev in reversed(self._buf):
            level = policy.level_of(_p_of(ev))
            if level in ("Alert", "Critical"):
                count += 1
            else:
                break
        return count

    def escalation_slope(self) -> float:
        """Inclinação OLS de p ao longo da janela (Δp por frame)."""
        ps = [_p_of(e) for e in self._buf]
        n = len(ps)
        if n < 2:
            return 0.0
        xm = (n - 1) / 2.0
        ym = sum(ps) / n
        num = sum((i - xm) * (p - ym) for i, p in enumerate(ps))
        den = sum((i - xm) ** 2 for i in range(n))
        return num / den if den else 0.0

    def reused_ratio(self) -> float:
        if not self._buf:
            return 0.0
        reused = sum(1 for e in self._buf if isinstance(e, OSSummary))
        return reused / len(self._buf)

    def last_compute_age(self) -> int:
        """Idade (frames) da última computação efetiva na janela; W se não houver."""
        for age, ev in enumerate(reversed(self._buf)):
            if isinstance(ev, ZState):
                return age
        return len(self._buf)

    def mean_confidence(self) -> float:
        if not self._buf:
            return 0.0
        return sum(_conf_of(e) for e in self._buf) / len(self._buf)

    def has_unsafe_suppression(self) -> bool:
        return any(
            isinstance(e, OSSummary) and e.suppression_quality == "unsafe"
            for e in self._buf
        )


class EAML:
    """Mediação episódica de alerta. Consome evidências da OSSL; fala com o
    MHEG apenas via ``request_force_compute``; é a única origem da HMI."""

    def __init__(self, policy: EdgePolicy, mheg: Optional[MHEGScheduler] = None):
        self.policy = policy
        self.mheg = mheg
        self.buffer = EpisodicBuffer(policy.window_w)
        self.guard = EventAwareGovernance(
            theta_off=policy.theta_off,
            min_realert_gap=policy.min_realert_gap,
            max_alerts_per_trip=policy.max_alerts_per_trip,
            escalation_delta=policy.escalation_delta,
            end_persistence=policy.end_persistence,
        )
        self.suppress_alert_noise = False   # sinalizador externo de ruído de HMI
        self._episode_id = 0
        self._guard_was_active = False

    # ------------------------------------------------------------------ API
    def observe_response(self, responded: bool) -> None:
        """Acopla a resposta do motorista ao alerta mais recente (via EGA)."""
        self.guard.observe_response(int(responded))

    def step(self, ev: Evidence) -> AlertDecision:
        pol = self.policy
        self.buffer.push(ev)

        peak = self.buffer.peak()
        top_k = self.buffer.mean_top_k(pol.top_k)
        persistence = self.buffer.persistence(pol)
        slope = self.buffer.escalation_slope()
        reused = self.buffer.reused_ratio()
        age = self.buffer.last_compute_age()
        confidence = self.buffer.mean_confidence()
        episode_risk = 0.7 * top_k + 0.3 * peak

        quality = "sufficient"
        if self.buffer.has_unsafe_suppression() or reused > pol.reused_ratio_max:
            quality = "insufficient"

        # ---------------- RISCO (qualquer) --------------------------------
        risk_reasons = []
        if peak >= pol.theta_peak:
            risk_reasons.append("peak")
        if top_k >= pol.theta_episode:
            risk_reasons.append("top_k")
        if persistence >= pol.m_persistence:
            risk_reasons.append("persistence")
        if len(self.buffer) >= 3 and slope >= pol.theta_slope:
            risk_reasons.append("escalation")

        base = dict(
            t=ev.t, episode_risk=episode_risk, peak=peak, mean_top_k=top_k,
            persistence=persistence, escalation_slope=slope, reused_ratio=reused,
            last_compute_age=age, confidence=confidence, evidence_quality=quality,
            risk_reasons=tuple(risk_reasons), episode_id=self._episode_id,
        )

        if not risk_reasons:
            return AlertDecision(action="none", **base)

        # ------- force_compute_before_alert (evidência requentada) --------
        if reused > pol.reused_ratio_max and age > pol.max_skip_safe:
            if self.mheg is not None:
                self.mheg.request_force_compute()
            return AlertDecision(
                action="force_compute_before_alert",
                guard_blocks=("stale_evidence",),
                **base,
            )

        # ---------------- GUARDA (todas) -----------------------------------
        blocks = []
        if confidence < pol.theta_conf:
            blocks.append("low_confidence")
        if self.suppress_alert_noise:
            blocks.append("alert_noise")
        if quality != "sufficient":
            blocks.append("evidence_quality")
        if blocks:
            return AlertDecision(action="hold", guard_blocks=tuple(blocks), **base)

        # ---------------- entrega orientada a eventos (EGA) ----------------
        dominant = pol.level_of(peak)
        delivered, redundant = self.guard.step(
            score=episode_risk, theta=pol.theta_on, rclass=dominant,
        )
        if self.guard._active and not self._guard_was_active:
            self._episode_id += 1
            base["episode_id"] = self._episode_id
        self._guard_was_active = self.guard._active

        if delivered:
            return AlertDecision(action="alert", redundant=bool(redundant), **base)
        return AlertDecision(action="hold", guard_blocks=("event_guard",), **base)
