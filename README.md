# Artifact — From Frame to Episode: Evidence-Preserving Alert Mediation for Selective Inference in Vehicular Edge Systems

Frozen artifact for the paper (UCC 2026 submission). All data is synthetic;
episodes are generated deterministically by the included generator and
personas, and every reported number comes from the frozen outputs in
`artifact/experiments/e10_eaml/outputs*`.

## Layout

```
artifact/                the reviewer artifact (unchanged from the version
                         attached to the submission; integrity covered by
                         artifact/MANIFEST.sha256)
  edge_runtime/          MHEG scheduler, OSSL suppression layer, EAML mediation,
                         event guard, runtime
  experiments/e10_eaml/  protocol code (scenarios, arms A0-A4, metrics, runners)
                         + frozen outputs: spec_e10.json, results/summary for the
                         20-seed holdout, A-OSC stress, development grid,
                         strong-accept post-hoc analyses, runtime microbenchmark,
                         and the two earlier specifications that produced empty
                         admissible sets (outputs_v1_strict, outputs_v2_superiority)
  pipeline_completo/     synthetic risk generator (behavioral recipes)
  sim/                   the twelve driver personas
paper/                   submitted manuscript source, figure and PDF
docs/                    the confirmatory protocol with its v1/v2/v3 amendment
                         trail (in Portuguese; the amendment history is also
                         machine-readable in spec_e10.json)
```

## Reproducing

From `artifact/`, with Python 3.12 and numpy:

```
python -m experiments.e10_eaml.run_dev
python -m experiments.e10_eaml.run_holdout
python -m experiments.e10_eaml.run_aosc
python -m experiments.e10_eaml.run_strongaccept
python -m experiments.e10_eaml.bench_runtime_cost
```

## Table map (paper → frozen output)

| Paper item | Source file (under artifact/experiments/e10_eaml/) |
|---|---|
| Table III (frozen holdout) | outputs/summary_holdout.json, outputs/results_holdout.csv |
| Table IV (stratified by onset) | outputs/summary_holdout.json (abrupt/progressive fields) |
| Table V (admissibility gates) | outputs/summary_holdout.json (gate booleans, U90) |
| Table VI (post-hoc mechanism/baseline checks) | outputs/strongaccept/strongaccept_result.json |
| Table VII (runtime microbenchmark) | outputs/bench_runtime_cost.json |
| Fig. 2 (A-OSC stress) | outputs/summary_aosc.json, outputs/results_aosc.csv |
| Empty admissible sets (Sec. IV) | outputs_v1_strict/, outputs_v2_superiority/ |

## Notes

- The frozen configuration and seed lists live in
  `artifact/experiments/e10_eaml/outputs/spec_e10.json` (development seeds
  201–205, holdout seeds, amendment notes, per-seed exogenous hashes for the
  common-random-numbers integrity check).
- The tables were transcribed from the JSON summaries; there is no LaTeX
  table generator in the artifact.
- `artifact/MANIFEST.sha256` covers the 37 files of the reviewer artifact
  exactly as submitted; the `paper/` and `docs/` folders were added afterwards
  and are not part of that manifest.
