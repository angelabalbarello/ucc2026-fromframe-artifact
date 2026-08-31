"""Guarda de entrega orientada a eventos (EGA) — vendorizada do artigo UCC2026.

Origem: ucc2026_tobe/SDR-experiments/src/edge/event_governance.py
(experimento E9 — companion work, anonymized for review). Validada em holdout congelado
de 20 sementes: risco −8,1×10⁻³ (IC 95% [−8,4; −7,8]×10⁻³), recall +0,013,
custo +0,48 alertas/viagem sob orçamento B=2,5.

Papel neste pacote: núcleo da GUARDA da EAML. O controlador legado acoplava um
limiar de disparo mais baixo a um limiar de rearme mais baixo; um alerta
precoce marginal podia suprimir um alerta posterior mais útil. Esta guarda
representa o evento operacional explicitamente: abre no cruzamento do limiar,
fecha após recuperação persistente e permite um realerta limitado apenas após
não-resposta ou escalada material de severidade.

Diferenças em relação ao original: nenhuma na lógica; apenas docstring e
comentários. Mantida byte-compatível no comportamento para preservar a
validade experimental.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EventAwareGovernance:
    theta_off: float
    min_realert_gap: int = 2
    max_alerts_per_trip: int = 2
    escalation_delta: float = 0.10
    end_persistence: int = 1
    _active: bool = False
    _below_count: int = 0
    _since_last: int = 999
    _alerts: int = 0
    _last_alert_score: float = 0.0
    _last_response: bool = True
    _last_class: str = ""

    def step(self, score: float, theta: float, rclass: str,
             governed: bool = True) -> tuple[int, int]:
        """Return ``(delivered_alert, redundant_alert)`` for one episode."""
        self._since_last += 1
        raw_trigger = score >= theta
        if not governed:
            alert = int(raw_trigger)
            redundant = int(bool(alert) and rclass == self._last_class)
            if alert:
                self._last_class = rclass
            return alert, redundant

        if self._active:
            if score < self.theta_off:
                self._below_count += 1
                if self._below_count >= self.end_persistence:
                    self._active = False
                    self._below_count = 0
            else:
                self._below_count = 0

        can_open = raw_trigger and not self._active and self._since_last >= self.min_realert_gap
        escalated = score >= self._last_alert_score + self.escalation_delta
        unanswered = not self._last_response
        can_realert = (
            raw_trigger
            and self._active
            and self._since_last >= self.min_realert_gap
            and (unanswered or escalated)
        )
        alert = bool(
            self._alerts < self.max_alerts_per_trip
            and (can_open or can_realert)
        )
        redundant = 0
        if alert:
            redundant = int(rclass == self._last_class)
            self._active = True
            self._since_last = 0
            self._alerts += 1
            self._last_alert_score = float(score)
            self._last_response = False
            self._last_class = rclass
        return int(alert), redundant

    def observe_response(self, responded: int) -> None:
        """Attach the response observed after the most recent delivered alert."""
        if self._since_last == 0:
            self._last_response = bool(responded)
