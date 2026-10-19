# Provenance: paper item → artifact file

Paths are relative to the repository root. Classes:

- **frozen**: E10 result, reported as run;
- **re-analysis**: post-acceptance re-analysis of the frozen code or outputs;
- **definition**: the text or equation was corrected, and the number is unchanged.

| Paper item | Source | Class |
|---|---|---|
| Table III: recall, TTA_ef, FAR, alerts, redundant re-alerts | `artifact/experiments/e10_eaml/outputs/results_holdout.csv`, `summary_holdout.json` | frozen |
| Table III: SAR (uniform 12-frame window) and Inf./tr. | `artifact/experiments/e11_camera_ready/outputs/cost_by_arm.csv` | re-analysis |
| Table IV (onset regimes), Table V (gates) | `e10_eaml/outputs/summary_holdout.json`; G5 SAR row from `e11_camera_ready/outputs/summary_e11.json` | frozen + re-analysis |
| Eq. (1), TTA_ef as executed | `artifact/experiments/e10_eaml/metrics.py` (`episode_outcomes`); alternatives in `e11_camera_ready/outputs/latency_alternatives.json` | definition |
| Algorithm 1, Table II, invariant E2 | `artifact/edge_runtime/{mheg,ossl,eaml,event_guard,runtime}.py`; `e10_eaml/arms.py` (`run_a2`); `e11_camera_ready/outputs/e2_check.json`, `refresh_followup.json` | code |
| Table VI: A4, S3 | `e10_eaml/outputs/strongaccept/strongaccept_result.json`; uniform-window SAR in `e11_camera_ready/outputs/s3_uniform_sar.json` and `summary_e11.json` | frozen + re-analysis |
| Table VI: SAR accounting threshold (0.50/0.60/0.70) | `e11_camera_ready/outputs/raw_holdout.json` (per-alert reuse fractions; pooled shares), `summary_e11.json` (`sar12_*`, cluster means), `sar_trace_counters.json` (25 of 11,907 A2 alerts) | re-analysis |
| Table VI: A5, A1+F, A1 with max_skip 4/2 | `e11_camera_ready/outputs/cost_by_arm.csv` | re-analysis |
| Table VI: A2T / A1C (guard time base) | `e11_camera_ready/outputs/egs_audit.json`, `raw_egs_audit.json` | re-analysis |
| Table VII (sensitivity, seeds 7001–7010; all 25 variants) | `e11_camera_ready/outputs/sensitivity.csv`, `raw_sens.json`, `table_sensitivity.tex` | re-analysis |
| Table VIII (v7 → v7.1) | `artifact/experiments/k1_generator_correction/outputs/summary_k1.json`; `docs/GENERATOR_SPECIFICATION_NOTE.md` | re-analysis |
| Table IX (microbenchmark) | `e10_eaml/outputs/bench_runtime_cost.json` | frozen (hardware-dependent) |
| Fig. 2 (A-OSC) | `e10_eaml/outputs/summary_aosc.json`; drawn by `figures/make_fig_aosc.py` | frozen |
| Persona parameter ranges | `artifact/sim/driver_personas.py` | code |
| 31-attribute inventory (24 vary) | `e11_camera_ready/outputs/feature_inventory.csv` | re-analysis |
| Amendment trail (v1, v2, v3) | `e10_eaml/outputs_v1_strict/`, `outputs_v2_superiority/`, `outputs/spec_e10.json`, `docs/PROTOCOLO_E10_EAML_confirmatorio.md` | frozen |

Consolidated values and their classes are listed in `docs/RESULTS_MATRIX.md`.
