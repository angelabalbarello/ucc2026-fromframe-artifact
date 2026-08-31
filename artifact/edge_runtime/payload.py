"""Payload operacional TERA-aware e episode-aware — cinco blocos (A–E).

Carrega risco E a trilha de decisão (computacional e episódica) para a nuvem,
conforme Seção 7 da apresentação. É o objeto que o worker
(app/workers/redis_stream_worker.py) persiste e que a camada longitudinal
consolida como BRE.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from .eaml import EAML, AlertDecision
from .ossl import OSSummary, ZState
from .policy_state import EdgePolicy


def build_payload(
    eaml: EAML,
    decision: AlertDecision,
    policy: EdgePolicy,
    driver_pseudonym: str,
    context: Optional[dict] = None,
    hmi_action: Optional[str] = None,
) -> dict:
    """Monta o payload A–E a partir do estado episódico corrente."""
    context = context or {}
    items = eaml.buffer.items
    z_items = [e for e in items if isinstance(e, ZState)]
    oss_items = [e for e in items if isinstance(e, OSSummary)]

    payload = {
        # ---- A · Identificação & contexto --------------------------------
        "A_context": {
            "payload_id": None,  # preenchido abaixo (hash do conteúdo)
            "episode_id": decision.episode_id,
            "driver": driver_pseudonym,
            "road": context.get("road"),
            "traffic": context.get("traffic"),
            "night": context.get("night"),
            "rain": context.get("rain"),
            "risk_level": policy.level_of(decision.peak),
            "operational_state_summary": {
                "frames_in_window": len(items),
                "computed": len(z_items),
                "reused": len(oss_items),
            },
        },
        # ---- B · Trilha MHEG ---------------------------------------------
        "B_mheg_trail": {
            "edge_scheduler": "MHEG",
            "decision_sequence": [
                "compute" if isinstance(e, ZState) else "reuse" for e in items
            ],
            "compute_count": len(z_items),
            "reuse_count": len(oss_items),
            "suppression_rate": decision.reused_ratio,
            "max_skip": max((e.suppression_count for e in oss_items), default=0),
            "trigger_reasons": sorted({e.trigger_reason for e in z_items}),
            "tau_delta": policy.tau_delta,
            "tau_h": policy.tau_h,
        },
        # ---- C · Resumo de supressão --------------------------------------
        "C_suppression": {
            "reused_ratio": decision.reused_ratio,
            "max_delta_during_skip": max(
                (e.max_delta_since_last_compute for e in oss_items), default=0.0
            ),
            "mean_confidence_decay": (
                sum(e.confidence_decay for e in oss_items) / len(oss_items)
                if oss_items else None
            ),
            "suppression_quality_worst": _worst_quality(oss_items),
            "force_compute_events": int(
                decision.action == "force_compute_before_alert"
            ),
        },
        # ---- D · Estado episódico ------------------------------------------
        "D_episode": {
            "episode_risk": decision.episode_risk,
            "peak_risk": decision.peak,
            "mean_top_k_risk": decision.mean_top_k,
            "persistence_frames": decision.persistence,
            "escalation_slope": decision.escalation_slope,
            "alert_decision": decision.action,
            "evidence_quality": decision.evidence_quality,
            "hmi_action": hmi_action,
        },
        # ---- E · Nuvem & política ------------------------------------------
        "E_cloud_policy": {
            "policy_id": policy.policy_id,
            "candidate_brs": None,          # atribuído pela nuvem (BRS pipeline)
            "upload_priority": _upload_priority(decision),
            "policy_update_candidate": decision.action in (
                "alert", "force_compute_before_alert"
            ),
            "requires_human_review": decision.evidence_quality != "sufficient"
            and decision.action == "alert",
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    payload["A_context"]["payload_id"] = f"BRE_{digest}"
    return payload


def _worst_quality(oss_items) -> Optional[str]:
    order = {"safe": 0, "watch": 1, "unsafe": 2}
    if not oss_items:
        return None
    return max((e.suppression_quality for e in oss_items), key=order.get)


def _upload_priority(decision: AlertDecision) -> str:
    if decision.action == "alert":
        return "immediate"
    if decision.action in ("force_compute_before_alert", "hold"):
        return "high"
    return "normal"
