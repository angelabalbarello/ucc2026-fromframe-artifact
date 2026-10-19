# From Frame to Episode: artifact (v1.0.0)

Artifact for **"From Frame to Episode: Evidence-Preserving Alert Mediation for
Selective Inference in Vehicular Edge Systems"**, IEEE/ACM UCC 2026.

Authors: Angela Albarello, Edna Dias Canedo, Vinícius P. Gonçalves, Leandro A.
Villas, Roger Kreutz Immich, Rodolfo I. Meneguette, Geraldo P. Rocha Filho.

Release `v1.0.0` corresponds to the camera-ready version of the paper. The
submission-time artifact is the history of this repository before that release.

All data is synthetic. Episodes are generated deterministically by the included
generator and personas. No learned detector, real driving data or personal
data is included.

## Layout

```
artifact/
  edge_runtime/                 MHEG gate, OSSL Suppression Summary, EAML mediation,
                                event guard (EGS), runtime, policy
  pipeline_completo/            episode generator v7 (frozen run) and v7.1 (Sec. V-E)
  sim/                          the 12 driver personas
  experiments/e10_eaml/         frozen protocol code and outputs: spec v3, development
                                grid, 20-seed holdout, A-OSC, A4/S2/S3, microbenchmark,
                                empty-set specifications v1/v2
  experiments/e11_camera_ready/ post-acceptance re-analyses (uniform-window SAR, cost per
                                arm, A5, A1+F, max_skip controls, A2T/A1C, sensitivity,
                                latency alternatives, SAR traces, E2 check)
  experiments/k1_generator_correction/  paired re-run under v7 and v7.1
  tests/                        integrity, mechanism, metric, reproduction, time-base and
                                generator-isolation tests
figures/                        Fig. 2 script and PDF
docs/                           protocol and amendment trail, PROVENANCE.md (paper item -> file),
                                RESULTS_MATRIX.md, GENERATOR_SPECIFICATION_NOTE.md
paper/                          manuscript as submitted for review (anonymized; not the
                                camera-ready version, see paper/README.md)
```

Post-acceptance analyses (E11, v7.1) reuse the frozen code and configurations
and select nothing. The paper labels them as post hoc. See `docs/PROVENANCE.md`
for the source of every table and figure.

## Requirements

Python ≥ 3.10 and numpy. The tests use pytest. Figure 2 uses matplotlib.

## Verify integrity

Run from `artifact/`:

```
tr -d '\r' < MANIFEST.sha256 | sha256sum -c      # 33/33 frozen files
python -m pytest -q tests/test_integrity.py
```

The frozen files keep the CRLF line endings they had when the manifests were
written; `.gitattributes` preserves them on checkout. The freeze-time manifest
`experiments/e10_eaml/outputs/MANIFEST.sha256` matches the spec byte for byte
and `arms.py`, `metrics.py`, `scenarios.py` and the protocol after CRLF→LF
normalization, which `test_integrity.py` checks. `run_dev.py` differs from its
freeze-time entry. The holdout path does not import it; the paper's threats
section discloses this.

The file `RELEASE_MANIFEST.sha256` at the root lists every file of the release
commit.

## Reproduce

Run from `artifact/`:

```
python -m pytest -q tests                                   # 37 tests, ~1.5 min
python -m experiments.e10_eaml.run_holdout                  # frozen holdout
python -m experiments.e10_eaml.run_aosc
python -m experiments.e10_eaml.run_strongaccept
python -m experiments.e11_camera_ready.posthoc holdout      # see e11_camera_ready/README.md
python -m experiments.e11_camera_ready.posthoc aosc
python -m experiments.e11_camera_ready.posthoc sens
python -m experiments.e11_camera_ready.egs_audit
python -m experiments.k1_generator_correction.run_k1 holdout  # also: aosc, s3, sens, resp
python -m experiments.e10_eaml.bench_runtime_cost           # hardware-dependent
```

Figure 2 (from the frozen summary; no experiment is run), from the repository root:

```
python figures/make_fig_aosc.py artifact/experiments/e10_eaml/outputs/summary_aosc.json \
       artifact/experiments/e10_eaml/outputs/spec_e10.json figures/UCC2026C_fig_aosc.pdf
```

## Notes

- `artifact/README.md` and the comments in some frozen files still refer to the
  anonymized review version. They are kept unchanged so that the frozen
  manifests remain valid.
- Some code comments are in Portuguese.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
