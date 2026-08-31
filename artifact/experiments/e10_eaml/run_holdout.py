"""E10 · holdout — execução ÚNICA da especificação congelada (§6–7).

Recusa-se a rodar sem `spec_e10.json`. Não re-otimiza nada: os quatro braços
rodam exatamente com as configurações congeladas; os portões G1–G4 (vs A0),
G4b (A2 vs A1) e as checagens de H3 (A3 vs A2) são reportados por inteiro,
inclusive falhas.

Uso (a partir de SDR-backend/):
    python -m experiments.e10_eaml.run_holdout
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from experiments.e10_eaml import arms, metrics, scenarios

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def main():
    spec_path = OUT_DIR / "spec_e10.json"
    if not spec_path.exists():
        raise SystemExit("spec_e10.json ausente — rode run_dev primeiro "
                         "(o holdout só executa especificação congelada).")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["a2_config"] is None:
        print("[E10-holdout] decisão de dev foi keep_A0 (conjunto vazio); "
              "o holdout verificará apenas A0/A1/ablação não se aplica.")

    personas, n_trips = spec["personas"], spec["n_trips"]
    seeds = spec["holdout_seeds"]
    mheg_cfg = spec["mheg_config"]
    t0 = time.time()

    clusters = {a: [] for a in ("A0", "A1", "A2", "A3")}
    for seed in seeds:
        sd = scenarios.generate_seed(seed, personas, n_trips)
        runs = {
            "A0": arms.run_a0(sd, spec["a0_config"], mheg_cfg),
            "A1": arms.run_a1(sd, spec["a1_candidate"], mheg_cfg),
        }
        if spec["a2_config"]:
            runs["A2"] = arms.run_a2(sd, spec["a2_config"], mheg_cfg)
            runs["A3"] = arms.run_a3(sd, spec["a2_config"], mheg_cfg)
        for arm, recs in runs.items():
            clusters[arm].extend(
                metrics.collect_clusters(recs, seed, personas, n_trips))
        print(f"[E10-holdout] seed {seed} ok ({time.time()-t0:.0f}s)")

    # ---------------- resultados por cluster (CSV) -----------------------
    rows = []
    for arm, cls in clusters.items():
        for c in cls:
            rows.append({"arm": arm, **c})
    if rows:
        with open(OUT_DIR / "results_holdout.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    # ---------------- portões e hipóteses (spec v2: referência A1) -------
    summary = {"seeds": seeds,
               "specification_version": spec.get("specification_version", 1),
               "gates_reference": spec.get("gates_reference", "A0"),
               "arms_summary": [metrics.summary_row(a, c)
                                for a, c in clusters.items() if c]}
    a0, a1 = clusters["A0"], clusters["A1"]
    # descritivo: quanto custa a liberdade irrestrita do legado
    summary["desc_A1_vs_A0"] = metrics.gates(
        a1, a0, spec["budget_alerts_per_trip"], spec["recall_margin"])
    if clusters["A2"]:
        # PRIMÁRIO (H2 fiel, v3): não-inferioridade + superioridade vs A1
        summary["H1p_A2_vs_A1"] = metrics.gates_h2(
            clusters["A2"], a1, spec["budget_alerts_per_trip"],
            spec["recall_margin"],
            tta_margin_s=spec.get("tta_noninferiority_margin_s", 1.0))
        # descritivo: A2 vs legado irrestrito (reportado integralmente)
        summary["desc_A2_vs_A0"] = metrics.gates(
            clusters["A2"], a0, spec["budget_alerts_per_trip"],
            spec["recall_margin"])
        summary["H2p_G4b_sar_A2_vs_A1"] = metrics.gate_g4b_sar(
            clusters["A2"], a1)
        # H3 — ablação: A3 vs A2
        summary["H3_sar_A3_vs_A2"] = metrics.paired_diff(
            clusters["A3"], clusters["A2"], "sar")
        summary["H3_tta_A3_vs_A2"] = metrics.paired_diff(
            clusters["A3"], clusters["A2"], "tta_ef")
        summary["H3_gates_A3_vs_A1"] = metrics.gates_h2(
            clusters["A3"], a1, spec["budget_alerts_per_trip"],
            spec["recall_margin"],
            tta_margin_s=spec.get("tta_noninferiority_margin_s", 1.0))

        h1 = summary["H1p_A2_vs_A1"]["admissible"]
        h2 = h1 and summary["H2p_G4b_sar_A2_vs_A1"]["pass"]
        d_sar = summary["H3_sar_A3_vs_A2"]
        h3 = (h2 and ((not summary["H3_gates_A3_vs_A1"]["admissible"])
                      or (d_sar["mean"] is not None and d_sar["mean"] > 0
                          and d_sar["ci95"][0] > 0)))
        summary["hypotheses"] = {"H1_prime": bool(h1), "H2_prime": bool(h2),
                                 "H3": bool(h3),
                                 "note": "hierárquicas (§8); referência "
                                         "governada A1 (emenda v2, §12)"}

    out = OUT_DIR / "summary_holdout.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                              default=str), encoding="utf-8")
    print(f"[E10-holdout] resumo em {out} ({time.time()-t0:.0f}s)")
    if "hypotheses" in summary:
        print("[E10-holdout] hipóteses:", summary["hypotheses"])


if __name__ == "__main__":
    main()
