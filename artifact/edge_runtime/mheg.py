"""Nível 1 · MHEG — decide QUANDO COMPUTAR; nunca decide, sozinho, quando alertar.

Regra (idêntica à validada no ecossistema TERA e em pipeline_completo/
tera_pipeline/inference/tera_infer.py):

    se Δx_t >= τΔ  ou  H(p_{t-1}) >= τH  ->  computar (p_t, h_t)
    caso contrário                        ->  reutilizar p_{t-1}, h_{t-1}

Extensões em relação ao gate mínimo:
  * ``max_skip`` — teto de reutilizações consecutivas (política da nuvem);
  * ``request_force_compute()`` — canal pelo qual a EAML exige uma computação
    antes de alertar (force_compute_before_alert). A EAML PEDE; quem computa
    é o MHEG — invariante da apresentação.
"""
from __future__ import annotations

from dataclasses import dataclass

from .policy_state import EdgePolicy


@dataclass(frozen=True)
class MHEGDecision:
    compute: bool
    trigger_reason: str      # "delta" | "entropy" | "max_skip" | "force_compute" | "reuse"
    delta_x: float
    entropy_prev: float
    suppression_count: int   # reutilizações consecutivas até este frame


class MHEGScheduler:
    """Runtime scheduler do Nível 1. Estado: contador de supressões e pedido de força."""

    def __init__(self, policy: EdgePolicy):
        self.policy = policy
        self._suppression_count = 0
        self._force_requested = False

    def request_force_compute(self) -> None:
        """Chamado pela EAML: computar no próximo frame antes de alertar."""
        self._force_requested = True

    def decide(self, delta_x: float, entropy_prev: float) -> MHEGDecision:
        pol = self.policy
        if self._force_requested:
            reason = "force_compute"
        elif delta_x >= pol.tau_delta:
            reason = "delta"
        elif entropy_prev >= pol.tau_h:
            reason = "entropy"
        elif self._suppression_count >= pol.max_skip:
            reason = "max_skip"
        else:
            reason = "reuse"

        compute = reason != "reuse"
        decision = MHEGDecision(
            compute=compute,
            trigger_reason=reason,
            delta_x=float(delta_x),
            entropy_prev=float(entropy_prev),
            suppression_count=self._suppression_count,
        )
        if compute:
            self._suppression_count = 0
            self._force_requested = False
        else:
            self._suppression_count += 1
        return decision
