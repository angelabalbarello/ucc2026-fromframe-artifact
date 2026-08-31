"""Demo executável: um episódio crítico abrupto atravessando os quatro níveis.

Uso:  python -m edge_runtime.demo

Simula um episódio no estilo da receita C02 (crítico abrupto): fase normal
estável (o MHEG economiza computação), onset abrupto (Δx dispara), escalada de
risco e mediação episódica do alerta. Imprime a trilha de decisão frame a
frame e o payload A–E final.
"""
from __future__ import annotations

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # consoles Windows cp1252

from .eaml import AlertDecision
from .ossl import OSSummary
from .payload import build_payload
from .policy_state import EdgePolicy
from .runtime import EdgeRuntime

# (delta_x, p_se_computado) por frame — fase estável, onset em t=8, escalada
EPISODIO_C02 = [
    (0.05, 0.12), (0.04, 0.12), (0.03, 0.13), (0.05, 0.12),  # estável: reuso
    (0.04, 0.13), (0.06, 0.12), (0.05, 0.13), (0.04, 0.12),
    (0.72, 0.55), (0.65, 0.70), (0.58, 0.80), (0.55, 0.88),  # onset + escalada
    (0.50, 0.90), (0.30, 0.86), (0.20, 0.82), (0.10, 0.60),  # recuperação
    (0.08, 0.40), (0.05, 0.25),
]


def main() -> None:
    policy = EdgePolicy(m_persistence=3, theta_conf=0.3, max_skip=6)
    frame_p = [p for _, p in EPISODIO_C02]

    def infer_fn(t: int):
        p = frame_p[min(t, len(frame_p) - 1)]
        entropy = 4 * p * (1 - p) * 0.5 + 0.2   # proxy simples de H
        return p, entropy, 0.95

    rt = EdgeRuntime(policy, infer_fn)
    ultimo: AlertDecision | None = None

    print(f"{'t':>2} {'Δx':>5} {'decisão':<14} {'evid.':<6} {'p/carr':>6} "
          f"{'R_E':>5} {'reuso':>5} {'ação EAML':<26} motivo")
    print("-" * 96)
    for delta_x, _ in EPISODIO_C02:
        dec = rt.process_frame(delta_x=delta_x,
                               context={"night": True, "road": "highway"})
        ev = rt.eaml.buffer.items[-1]
        is_oss = isinstance(ev, OSSummary)
        p_shown = ev.carried_p if is_oss else ev.p
        reasons = ",".join(dec.risk_reasons) or "-"
        blocks = ",".join(dec.guard_blocks)
        detail = f"[{reasons}]" + (f" bloqueio={blocks}" if blocks else "")
        print(f"{dec.t:>2} {delta_x:>5.2f} "
              f"{'reuse' if is_oss else 'compute':<14} "
              f"{'OSS_t' if is_oss else 'z_t':<6} {p_shown:>6.2f} "
              f"{dec.episode_risk:>5.2f} {dec.reused_ratio:>5.2f} "
              f"{dec.action:<26} {detail}")
        if dec.action == "alert":
            rt.observe_response(True)   # motorista respondeu ao alerta
        ultimo = dec

    print("\nPayload A–E do episódio (para a nuvem consolidar como BRE):")
    payload = build_payload(rt.eaml, ultimo, policy, driver_pseudonym="P12",
                            context={"night": True, "road": "highway"})
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
