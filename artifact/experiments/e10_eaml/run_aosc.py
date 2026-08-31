"""E10 · A-OSC — análise adicional pré-declarada com receitas oscilantes (§12).

Executa os MESMOS braços congelados em `spec_e10.json` (nenhuma re-seleção)
sobre o mix oscilante (TRIP_SLOTS_OSC: 2×L04-osc + 1×C11-flutter por viagem).
Mede a sensibilidade das conclusões ao regime temporal — é aqui que a
patologia da fadiga de alarme (cruzamentos transitórios) existe de verdade.

Reporta sumários e diferenças pareadas (sem portões — não é confirmação):
    A2 vs A1, A2 vs A0, A3 vs A2, A1 vs A0.

Uso (a partir de SDR-backend/, APÓS o holdout v2):
    python -m experiments.e10_eaml.run_aosc
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
AOSC_SEEDS = list(range(5001, 5011))     # 10 seeds, disjuntas de dev/holdout


def main():
    spec_path = OUT_DIR / "spec_e10.json"
    if not spec_path.exists():
        raise SystemExit("spec_e10.json ausente — A-OSC só roda com braços "
                         "congelados (rode run_dev e, idealmente, o holdout antes).")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    personas, n_trips = spec["personas"], spec["n_trips"]
    mheg_cfg = spec["mheg_config"]
    t0 = time.time()

    clusters = {a: [] for a in ("A0", "A1", "A2", "A3")}
    for seed in AOSC_SEEDS:
        sd = scenarios.generate_seed(seed, personas, n_trips, osc_mix=True)
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
        print(f"[E10-AOSC] seed {seed} ok ({time.time()-t0:.0f}s)")

    rows = [{"arm": a, **c} for a, cls in clusters.items() for c in cls]
    if rows:
        with open(OUT_DIR / "results_aosc.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    summary = {
        "analysis": "A-OSC (adicional, sem portões — protocolo §12)",
        "seeds": AOSC_SEEDS,
        "arms_summary": [metrics.summary_row(a, c)
                         for a, c in clusters.items() if c],
        "paired": {},
    }
    pairs = [("A1", "A0"), ("A2", "A1"), ("A2", "A0"), ("A3", "A2")]
    fields = ("tta_ef", "recall", "far_per_trip", "alerts_per_trip", "sar")
    for cand, ref in pairs:
        if clusters[cand] and clusters[ref]:
            summary["paired"][f"{cand}_vs_{ref}"] = {
                f: metrics.paired_diff(clusters[cand], clusters[ref], f)
                for f in fields}

    out = OUT_DIR / "summary_aosc.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                              default=str), encoding="utf-8")
    print(f"[E10-AOSC] resumo em {out} ({time.time()-t0:.0f}s)")
    for row in summary["arms_summary"]:
        print("[E10-AOSC]", row)


if __name__ == "__main__":
    main()
