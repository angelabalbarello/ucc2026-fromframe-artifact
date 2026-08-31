"""SDRedge_runtime — os quatro níveis da arquitetura cognitiva na borda.

Pacote canônico que materializa, em código executável, a separação de decisões
defendida na apresentação "Do frame ao episódio":

    Nível 1 · MHEG  — computar ou reutilizar?        (mheg.py)
    Nível 2 · OSSL  — que evidência o frame gera?    (ossl.py: z_t | OSS_t)
    Nível 3 · EAML  — a evidência justifica alertar? (eaml.py; única origem da HMI)
    Nível 4 · Nuvem — recalibração de políticas      (app/services/promotion.py)

Invariantes globais (cobertos em tests/test_edge_runtime.py):
    I1. Nenhum alerta sai fora da EAML.
    I2. Um frame Critical isolado nunca dispara alerta sozinho.
    I3. OSS_t não produz novo p_t/h_t — frames suprimidos são
        "observados e sumarizados", jamais "inferidos".
    I4. Nenhum frame é descartado: todo frame vira z_t ou OSS_t.
    I5. Com R_E alto, reutilização excessiva e computação antiga, a EAML
        exige force_compute_before_alert antes de comunicar risco.
"""
from .policy_state import EdgePolicy
from .mheg import MHEGScheduler, MHEGDecision
from .ossl import OSSLayer, ZState, OSSummary
from .event_guard import EventAwareGovernance
from .eaml import EAML, EpisodicBuffer, AlertDecision
from .payload import build_payload
from .runtime import EdgeRuntime

__all__ = [
    "EdgePolicy",
    "MHEGScheduler", "MHEGDecision",
    "OSSLayer", "ZState", "OSSummary",
    "EventAwareGovernance",
    "EAML", "EpisodicBuffer", "AlertDecision",
    "build_payload",
    "EdgeRuntime",
]
