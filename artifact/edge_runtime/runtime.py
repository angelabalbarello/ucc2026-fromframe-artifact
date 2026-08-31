"""EdgeRuntime — encadeia MHEG → OSSL → EAML por frame.

O runtime é agnóstico ao modelo: recebe uma função ``infer_fn(t) -> (p,
entropy, confidence)`` que executa o núcleo causal (LSTM destilada, mock de
teste, etc.). Ela SÓ é chamada quando o MHEG decide computar — inclusive
quando a computação foi forçada pela EAML (force_compute_before_alert).

Fluxo por frame:
    1. MHEG decide computar/reutilizar (Δx_t, H carregada, max_skip, força);
    2. OSSL materializa z_t (com inferência) ou OSS_t (sem inferência);
    3. EAML agrega no buffer episódico e emite a AlertDecision — única HMI.

Invariante I1: este módulo não contém nenhum caminho de alerta próprio;
`AlertDecision` da EAML é o único canal.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

from .eaml import EAML, AlertDecision
from .mheg import MHEGScheduler
from .ossl import OSSLayer
from .policy_state import EdgePolicy

InferFn = Callable[[int], Tuple[float, float, float]]  # t -> (p, entropy, confidence)


class EdgeRuntime:
    def __init__(self, policy: EdgePolicy, infer_fn: InferFn):
        self.policy = policy
        self.infer_fn = infer_fn
        self.mheg = MHEGScheduler(policy)
        self.ossl = OSSLayer(policy)
        self.eaml = EAML(policy, mheg=self.mheg)
        self._t = -1
        self._last_entropy = 1.0   # frame 0 computa: entropia máxima a priori

    # ---------------------------------------------------------------- API
    def process_frame(self, delta_x: float, context: Optional[dict] = None) -> AlertDecision:
        """Processa um frame de sensoriamento e retorna a decisão da EAML."""
        self._t += 1
        t = self._t

        decision = self.mheg.decide(delta_x=delta_x, entropy_prev=self._last_entropy)

        if decision.compute:
            p, entropy, confidence = self.infer_fn(t)
            self._last_entropy = float(entropy)
            evidence = self.ossl.materialize(
                t, decision, p_t=p, entropy_t=entropy,
                confidence_t=confidence, context=context,
            )
        else:
            evidence = self.ossl.materialize(t, decision, context=context)

        return self.eaml.step(evidence)

    def observe_response(self, responded: bool) -> None:
        self.eaml.observe_response(responded)
