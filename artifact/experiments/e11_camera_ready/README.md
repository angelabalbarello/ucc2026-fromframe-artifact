# E11 — camera-ready post-hoc analyses (post-acceptance, exploratory)

Nothing in `experiments/e10_eaml/` is modified; this package imports it.
All results here are **post-acceptance** and must be labelled as such.

| Key | Arm |
|---|---|
| A0–A3 | frozen E10 arms (`arms.run_a0..a3`); A2 re-run as `run_a2x` with uniform 12-frame SAR |
| A2_frozen | frozen `arms.run_a2` (SAR on the EAML window W=8, as in E10) |
| A4 | A2 without the force channel (same as E10 `run_a4`, uniform SAR) |
| A5 | A2 with OSS reduced to provenance + age (no decay, quality=safe) |
| A1F | A1 + reuse/age counter (8 frames) + forced refresh |
| A1_maxskipN | A1 with max_skip = N (cost-increasing controls) |

Reproduce from `artifact/` (Python ≥ 3.10, numpy):

```
python -m experiments.e11_camera_ready.posthoc holdout   # ~8 min, 2 processes
python -m experiments.e11_camera_ready.posthoc aosc      # ~3 min
python -m experiments.e11_camera_ready.posthoc sens      # ~25 min (fresh seeds 7001-7010)
python -m experiments.e11_camera_ready.summarize
python -m experiments.e11_camera_ready.make_tables
python -m experiments.e11_camera_ready.latency_alt
python -m experiments.e11_camera_ready.sar_trace          # counters + traces A-E
python -m experiments.e11_camera_ready.refresh_followup
python -m experiments.e11_camera_ready.trace_e2 3001
```

`posthoc holdout` also re-runs the frozen A0–A3 and reproduces `results_holdout.csv` exactly (0 mismatches).
