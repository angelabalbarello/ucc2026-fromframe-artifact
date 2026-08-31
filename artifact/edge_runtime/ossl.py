"""Nível 2 · OSSL — Operational State & Suppression Layer.

Materializa a decisão do MHEG em dois objetos de evidência (Seção 5 da
apresentação): o estado operacional z_t quando o frame é COMPUTADO e o resumo
operacional OSS_t quando o frame é SUPRIMIDO.

Invariantes desta camada:
  (1) OSS_t não produz novo p_t/h_t — carrega o p reutilizado apenas como
      referência (`carried_p`), nunca como inferência nova;
  (2) OSS_t é de baixa complexidade (O(1), aritmética elementar);
  (3) nenhum frame é descartado — todo frame vira z_t ou OSS_t;
  (4) a linguagem é "observado e sumarizado", jamais "inferido"
      (`infer_status="reuse"`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .mheg import MHEGDecision
from .policy_state import EdgePolicy


@dataclass(frozen=True)
class ZState:
    """z_t — evidência de frame COMPUTADO."""
    t: int
    p: float                       # probabilidade de risco inferida neste frame
    level: str                     # ℓ_t ∈ {Normal, Attention, Alert, Critical}
    entropy: float                 # H_t
    delta_x: float
    edge_decision: str = "compute"
    trigger_reason: str = "delta"
    confidence: float = 1.0
    context: dict = field(default_factory=dict)
    policy_id: str = ""


@dataclass(frozen=True)
class OSSummary:
    """OSS_t — resumo de frame SUPRIMIDO. Observado e sumarizado, nunca inferido."""
    t: int
    infer_status: str              # sempre "reuse"
    reused_from: int               # t da última computação efetiva
    elapsed_since_compute: int
    delta_x: float
    max_delta_since_last_compute: float
    entropy_carried: float         # H da última computação (carregada, não nova)
    carried_p: float               # p reutilizado (referência; NÃO é inferência nova)
    confidence_decay: float        # confiança decaída da última computação
    cheap_risk_proxy: float        # proxy barato: carried_p ajustado por Δx
    trigger_margin_delta: float    # distância de Δx até τΔ (negativo = folga)
    trigger_margin_entropy: float  # distância de H até τH
    suppression_count: int
    suppression_quality: str       # {"safe", "watch", "unsafe"}
    context: dict = field(default_factory=dict)
    policy_id: str = ""


Evidence = ZState  # alias semântico; janelas da EAML carregam ZState | OSSummary


class OSSLayer:
    """Produz z_t ou OSS_t a partir da decisão do MHEG.

    Mantém apenas o estado mínimo entre frames: última computação efetiva
    (t, p, H, confiança) e o máximo de Δx observado durante a supressão.
    """

    def __init__(self, policy: EdgePolicy):
        self.policy = policy
        self._last_compute_t: int = -1
        self._last_p: float = 0.0
        self._last_entropy: float = 0.0
        self._last_confidence: float = 1.0
        self._max_delta_in_skip: float = 0.0

    # ------------------------------------------------------------------ API
    def materialize(
        self,
        t: int,
        decision: MHEGDecision,
        p_t: Optional[float] = None,
        entropy_t: Optional[float] = None,
        confidence_t: float = 1.0,
        context: Optional[dict] = None,
    ):
        """Retorna ZState (compute) ou OSSummary (reuse) para o frame t.

        Quando ``decision.compute`` é True, ``p_t`` e ``entropy_t`` são
        obrigatórios (vêm do núcleo causal). Quando é False, eles DEVEM ser
        omitidos: a OSSL nunca aceita inferência nova para frame suprimido.
        """
        context = context or {}
        if decision.compute:
            if p_t is None or entropy_t is None:
                raise ValueError("frame computado exige p_t e entropy_t do núcleo causal")
            z = ZState(
                t=t,
                p=float(p_t),
                level=self.policy.level_of(float(p_t)),
                entropy=float(entropy_t),
                delta_x=decision.delta_x,
                trigger_reason=decision.trigger_reason,
                confidence=float(confidence_t),
                context=context,
                policy_id=self.policy.policy_id,
            )
            self._last_compute_t = t
            self._last_p = float(p_t)
            self._last_entropy = float(entropy_t)
            self._last_confidence = float(confidence_t)
            self._max_delta_in_skip = 0.0
            return z

        if p_t is not None or entropy_t is not None:
            raise ValueError(
                "frame suprimido não aceita p_t/entropy_t: OSS_t é observado e "
                "sumarizado, jamais inferido (invariante OSSL-1)"
            )
        if self._last_compute_t < 0:
            raise ValueError("não há computação anterior para reutilizar (frame 0 deve computar)")

        pol = self.policy
        self._max_delta_in_skip = max(self._max_delta_in_skip, decision.delta_x)
        elapsed = t - self._last_compute_t
        decay = pol.confidence_decay_rate ** elapsed
        confidence_decay = self._last_confidence * decay
        proxy = min(1.0, self._last_p + 0.5 * decision.delta_x)

        return OSSummary(
            t=t,
            infer_status="reuse",
            reused_from=self._last_compute_t,
            elapsed_since_compute=elapsed,
            delta_x=decision.delta_x,
            max_delta_since_last_compute=self._max_delta_in_skip,
            entropy_carried=self._last_entropy,
            carried_p=self._last_p,
            confidence_decay=confidence_decay,
            cheap_risk_proxy=proxy,
            trigger_margin_delta=decision.delta_x - pol.tau_delta,
            trigger_margin_entropy=self._last_entropy - pol.tau_h,
            suppression_count=decision.suppression_count + 1,
            suppression_quality=self._quality(decision.delta_x, elapsed),
            context=context,
            policy_id=pol.policy_id,
        )

    # ------------------------------------------------------------ internos
    def _quality(self, delta_x: float, elapsed: int) -> str:
        """Selo de qualidade da supressão: safe | watch | unsafe.

        unsafe: a entrada variou tanto quanto o gatilho (deveria ter computado)
                ou a reutilização já ultrapassou o teto seguro de idade;
        watch:  Δx se aproxima do gatilho ou a idade se aproxima do teto;
        safe:   folga clara em ambas as dimensões.
        """
        pol = self.policy
        if delta_x >= pol.tau_delta or elapsed > pol.max_skip * pol.unsafe_elapsed_frac:
            return "unsafe"
        if delta_x >= pol.watch_margin * pol.tau_delta or elapsed > pol.max_skip_safe:
            return "watch"
        return "safe"
