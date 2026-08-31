"""Política vigente na borda — u = Π(Λ(d), c), recebida da nuvem.

Agrupa em um único objeto versionado todos os parâmetros que a análise
longitudinal pode recalibrar (Seção "Nuvem" da apresentação): limiares do
MHEG, limites de supressão da OSSL e limiares/guardas da EAML.

A borda nunca altera estes valores por conta própria; eles só mudam quando a
nuvem promove uma política nova via portões pós-salvaguarda
(app/services/promotion.py). Isso preserva a lição do artigo UCC2026: nenhum
parâmetro de disparo é recalibrado sem auditar a ação entregue.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EdgePolicy:
    """Parâmetros operacionais vigentes na borda (imutável; troca por versão)."""

    policy_id: str = "policy-default-v1"

    # ---- Nível 1 · MHEG (quando computar) --------------------------------
    tau_delta: float = 0.50        # limiar de variação cinemática |Δx_t|
    tau_h: float = 0.60            # limiar de entropia H(p_{t-1})
    max_skip: int = 8              # máximo de frames reutilizados em sequência

    # ---- Nível 2 · OSSL (qualidade da supressão) -------------------------
    confidence_decay_rate: float = 0.97   # conf_t = conf_last * rate^elapsed
    watch_margin: float = 0.80            # fração de tau_delta que rebaixa p/ watch
    unsafe_elapsed_frac: float = 1.0      # elapsed > max_skip*frac => unsafe

    # ---- Nível 3 · EAML (quando alertar) ---------------------------------
    window_w: int = 12             # janela episódica W (frames)
    theta_peak: float = 0.85       # RISCO: pico
    theta_episode: float = 0.70    # RISCO: média dos top-k
    top_k: int = 3
    m_persistence: int = 3         # RISCO: nível Alert/Critical por m frames
    theta_slope: float = 0.04      # RISCO: escalada (Δp por frame)
    theta_conf: float = 0.55       # GUARDA: confiança mínima da janela
    reused_ratio_max: float = 0.60 # GUARDA/força: reutilização excessiva
    max_skip_safe: int = 4         # força: idade máxima da última computação

    # ---- Guarda de entrega (EGA vendorizada do UCC2026) ------------------
    theta_on: float = 0.55         # abertura do evento (sobre R_E)
    theta_off: float = 0.52        # fechamento persistente do evento
    min_realert_gap: int = 2       # frames mínimos entre alertas
    max_alerts_per_trip: int = 3   # orçamento de alertas por viagem
    escalation_delta: float = 0.10 # realerta só se R_E escalar isto
    end_persistence: int = 1       # frames abaixo de theta_off p/ fechar

    # ---- Classificação de nível de risco por frame -----------------------
    level_bins: tuple = field(default=(0.35, 0.55, 0.75))  # N/A/Al/Cr

    def level_of(self, p: float) -> str:
        b1, b2, b3 = self.level_bins
        if p < b1:
            return "Normal"
        if p < b2:
            return "Attention"
        if p < b3:
            return "Alert"
        return "Critical"
