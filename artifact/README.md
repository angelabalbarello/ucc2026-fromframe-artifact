# Anonymized artifact — "From Frame to Episode" (UCC 2026)

Supplementary material for double-blind review. All author-,
institution-, and machine-identifying strings were removed.

## Layout
- `edge_runtime/` — MHEG (gate), OSSL (Suppression Summary),
  EAML (episodic alert mediation), event guard (EGS), policy.
- `experiments/e10_eaml/` — protocol code: `scenarios.py`
  (episode generator wrapper, personas, exogenous pre-sampling,
  CRN hash), `arms.py` (A0-A3), `metrics.py` (metrics, cluster
  aggregation, gates; normal-approximation CIs, U90 = mean +
  1.645 SE), `run_dev.py`, `run_holdout.py`, `run_aosc.py`,
  `run_strongaccept.py` (A4, SAR-threshold sweep, re-optimized
  EGS), `bench_runtime_cost.py` (microbenchmark).
- `experiments/e10_eaml/outputs/` — frozen results:
  `spec_e10.json` (frozen v3 spec), `results_holdout.csv` and
  `summary_holdout.json` (Tables II-IV), `results_aosc.csv` and
  `summary_aosc.json` (Fig. 2), `strongaccept/` (Table V),
  `bench_runtime_cost.json` (Table VI), `MANIFEST.sha256`.
- `outputs_v1_strict/`, `outputs_v2_superiority/` — the two
  earlier specifications with empty admissible sets (54/54 and
  162/162 rejections), preserved as reported in Sec. IV.
- `pipeline_completo/synthetic_driver_risk_v7.py` — behavioral
  recipe generator. `sim/driver_personas.py` — the 12 personas.

## Reproduction
From the artifact root (Python 3.12, numpy):
`python -m experiments.e10_eaml.run_holdout`,
`python -m experiments.e10_eaml.run_aosc`,
`python -m experiments.e10_eaml.run_strongaccept`,
`python experiments/e10_eaml/bench_runtime_cost.py`.
Seeds: development 201-205, holdout 3001-3020, A-OSC 5001-5010,
re-optimized-EGS evaluation 6001-6010.

`MANIFEST.sha256` at the artifact root covers every file.
