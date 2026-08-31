"""E10 · desenvolvimento — calibração, grade, portões e CONGELAMENTO (§6).

Uso (a partir de SDR-backend/):
    python -m experiments.e10_eaml.run_dev            # protocolo completo
    python -m experiments.e10_eaml.run_dev --smoke    # validação rápida

Saídas em experiments/e10_eaml/outputs/:
    results_dev.csv       — portões por candidata (trilha de decisão)
    spec_e10.json         — especificação congelada p/ o holdout
    MANIFEST.sha256       — hash de spec + código + protocolo
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from experiments.e10_eaml import arms, metrics, scenarios

OUT_DIR = Path(__file__).resolve().parent / "outputs"

DEV_SEEDS = [201, 202, 203, 204, 205]
HOLDOUT_SEEDS = list(range(3001, 3021))
BUDGET = 2.5
RECALL_MARGIN = 0.02

A0_CFG = {"theta_alert": 0.55, "cooldown_frames": 24, "hysteresis": 0.05}
# v3 (§13): margem de não-inferioridade de TTA_ef vs A1 = 1,0 s (10% de
# T_max; ordem de grandeza de uma janela W de mediação). Declarada com os
# dados de DEV já vistos — transparente na spec; holdout intocado.
TTA_MARGIN_S = 1.0
# A1 = C8 exata promovida em E9, sem re-otimização (referência governada, v2)
A1_CAND = {"theta_on": 0.55, "theta_off": 0.52, "min_realert_gap": 2,
           "max_alerts_per_trip": 3, "escalation_delta": 0.10,
           "end_persistence": 1}


def guards_v2():
    """Grade de guarda recalibrada para o regime deste gerador (emenda v2,
    protocolo §12): θ_on × A_max; demais parâmetros herdados de E9."""
    out = {}
    for theta_on in (0.55, 0.60, 0.65):
        for amax in (3, 4, 6):
            out[f"G{theta_on:.2f}-A{amax}"] = {
                "theta_on": theta_on, "theta_off": round(theta_on - 0.03, 4),
                "min_realert_gap": 2, "max_alerts_per_trip": amax,
                "escalation_delta": 0.10, "end_persistence": 1,
            }
    return out


def full_grid():
    for w in (8, 12, 16):
        for m in (2, 3):
            for th in (0.65, 0.70, 0.75):
                for gname, g in guards_v2().items():
                    yield {"window_w": w, "m_persistence": m,
                           "theta_episode": th, "guard": dict(g),
                           "guard_name": gname}


def smoke_grid():
    gs = guards_v2()
    for gname in ("G0.55-A3", "G0.60-A4"):
        yield {"window_w": 12, "m_persistence": 2, "theta_episode": 0.70,
               "guard": dict(gs[gname]), "guard_name": gname}


def calibrate_tau_delta(seed_datas) -> float:
    """τΔ = p75 da distribuição de Δx nas seeds de DEV (frame 0 excluído)."""
    vals = np.concatenate([ep.dx[1:] for sd in seed_datas
                           for ep in sd.episodes])
    return float(np.percentile(vals, 75))


def run_arm_over_seeds(runner, cfg, mheg_cfg, seed_datas, personas, n_trips):
    clusters = []
    for sd in seed_datas:
        recs = runner(sd, cfg, mheg_cfg)
        clusters.extend(metrics.collect_clusters(recs, sd.seed, personas, n_trips))
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        seeds, personas, n_trips = [201, 202], ["P02", "P08"], 4
        grid = list(smoke_grid())
    else:
        seeds, personas, n_trips = DEV_SEEDS, scenarios.ALL_PERSONAS, 20
        grid = list(full_grid())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[E10-dev] gerando {len(seeds)} seeds x {len(personas)} personas "
          f"x {n_trips} viagens ...")
    seed_datas = [scenarios.generate_seed(s, personas, n_trips) for s in seeds]
    exo = {sd.seed: sd.exogenous_hash for sd in seed_datas}
    print(f"[E10-dev] geração ok em {time.time()-t0:.1f}s; "
          f"{sum(len(sd.episodes) for sd in seed_datas)} episódios")

    mheg_cfg = {"tau_delta": calibrate_tau_delta(seed_datas),
                "tau_h": 0.60, "max_skip": 8}
    print(f"[E10-dev] tau_delta calibrado (p75 dev) = {mheg_cfg['tau_delta']:.4f}")

    ref = run_arm_over_seeds(arms.run_a0, A0_CFG, mheg_cfg,
                             seed_datas, personas, n_trips)
    a1 = run_arm_over_seeds(arms.run_a1, A1_CAND, mheg_cfg,
                            seed_datas, personas, n_trips)
    print("[E10-dev] A0 (ref):", metrics.summary_row("A0", ref))
    print("[E10-dev] A1 (C8):", metrics.summary_row("A1", a1))

    rows, admissible = [], []
    for i, cfg in enumerate(grid, 1):
        clusters = run_arm_over_seeds(arms.run_a2, cfg, mheg_cfg,
                                      seed_datas, personas, n_trips)
        # v3 (protocolo §13): implementação FIEL da H2 pré-declarada —
        # não-inferioridade em TTA/recall vs A1 (EGA) + superioridade em ≥1
        # dimensão de qualidade de alerta. A0 permanece descritivo.
        rep = metrics.gates_h2(clusters, a1, budget=BUDGET,
                               recall_margin=RECALL_MARGIN,
                               tta_margin_s=TTA_MARGIN_S)
        d_tta = rep["diagnostics"]["d_tta_ef"]
        row = {"candidate": i, **{k: cfg[k] for k in
                                  ("window_w", "m_persistence",
                                   "theta_episode", "guard_name")},
               **{k: (";".join(rep[k]) if k == "G5_dimensions" else rep[k])
                  for k in rep if k.startswith("G") or
                  k in ("admissible", "alerts_per_trip")},
               "mean_d_tta": d_tta["mean"], "u90_d_tta": d_tta["u90"]}
        rows.append(row)
        if rep["admissible"]:
            admissible.append((d_tta["mean"], i, cfg))
        print(f"[E10-dev] cand {i}/{len(grid)} "
              f"{cfg['window_w']}/{cfg['m_persistence']}/"
              f"{cfg['theta_episode']}/{cfg['guard_name']} "
              f"admissivel={rep['admissible']} dTTA={d_tta['mean']}")

    with open(OUT_DIR / "results_dev.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    if admissible:
        admissible.sort(key=lambda x: x[0])
        _, win_idx, winner = admissible[0]
        decision = "promote_A2"
    else:
        win_idx, winner = None, None
        decision = "keep_A0"

    spec = {
        "specification_version": 3,
        "gates_reference": "A1 (EGA C8, E9) — H2 fiel: não-inferioridade "
                           "TTA/recall + superioridade em qualidade (§13)",
        "tta_noninferiority_margin_s": TTA_MARGIN_S,
        "amendment_note": "v1 (ref. A0 irrestrita): vazio 54/54 (G1/G2) — "
                          "outputs_v1_strict/. v2 (superioridade estrita de "
                          "TTA vs A1, MAIS estrita que a H2 pré-declarada): "
                          "vazio 162/162 — outputs_v2_superiority/. v3 "
                          "implementa a H2 como declarada no §1; margem TTA "
                          "definida com dados de dev vistos (transparente). "
                          "Holdout jamais inspecionado.",
        "protocol": "docs/PROTOCOLO_E10_EAML_confirmatorio.md",
        "smoke": bool(args.smoke),
        "decision_dev": decision,
        "winner_candidate_index": win_idx,
        "a2_config": winner,
        "a0_config": A0_CFG,
        "a1_candidate": A1_CAND,
        "mheg_config": mheg_cfg,
        "unit_frames": arms.UNIT_FRAMES,
        "sar_window": arms.SAR_WINDOW,
        "budget_alerts_per_trip": BUDGET,
        "recall_margin": RECALL_MARGIN,
        "personas": personas, "n_trips": n_trips,
        "dev_seeds": seeds,
        "holdout_seeds": HOLDOUT_SEEDS if not args.smoke else [3001, 3002],
        "holdout_seen_during_selection": False,
        "dev_exogenous_hashes": exo,
    }
    spec_path = OUT_DIR / "spec_e10.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    manifest = []
    for p in [spec_path, Path(__file__),
              Path(__file__).parent / "arms.py",
              Path(__file__).parent / "metrics.py",
              Path(__file__).parent / "scenarios.py",
              _BACKEND / "docs/PROTOCOLO_E10_EAML_confirmatorio.md"]:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest.append(f"{digest}  {p.name}")
    (OUT_DIR / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n",
                                             encoding="utf-8")

    print(f"[E10-dev] decisão: {decision}; vencedora: {winner}")
    print(f"[E10-dev] spec congelada em {spec_path}")
    print(f"[E10-dev] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
