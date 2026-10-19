# Generator specification sensitivity (v7 and v7.1)

This note documents Sec. V-E ("Generator specification") and Table VIII of the paper.

## The difference

In generator v7 (`artifact/pipeline_completo/synthetic_driver_risk_v7.py`,
SHA-256/16 `efede62fe3aceed8`), borderline recipe L02 (declared risk target 0.35)
uses the `peak_then_drop` curve shared with the abrupt-critical recipes C01 and
C04. That curve sets `peak = max(0.75, target + 0.05)`, so L02's latent peak
lies in the critical band (median 0.85).

Generator v7.1 (`synthetic_driver_risk_v7_1.py`, SHA-256/16 `fd9bd15e5299404e`)
is a copy of v7 with a header comment and **one changed code line**:
`peak = target + 0.05`. Only L02 is affected; its median latent peak becomes 0.46.

The paper does not state which specification was intended. It reports both,
with v7 as the version of the frozen run.

## Isolation

`artifact/tests/test_k1_generator.py` (7 tests) checks the following:

- **Changed code.** The only code-line difference between v7 and v7.1 is the
  floor line, and the original file hash is intact.
- **Recipes other than L02.** Every recipe except L02 produces bit-identical
  inputs, latent risk, observed risk and metadata.
- **L02 itself.** L02 keeps its label, onset and response draw.
- **Exogenous hash.** Switching back restores the original exogenous hash.

`artifact/experiments/k1_generator_correction/genswitch.py` swaps the
generator without modifying any E10 or E11 file.

## Paired re-run

`run_k1.py` re-runs the v7-frozen arms on the same seeds (holdout 3001–3020,
A-OSC 5001–5010, S3 6001–6010, sensitivity 7001–7010), with no re-selection.
The re-run under v7 reproduces the frozen E10 and E11 outputs with zero
mismatches (`outputs/summary_k1.json`, fields `reproduction`). Table VIII of
the paper is taken from `outputs/summary_k1.json`.

The MHEG threshold τΔ was calibrated on v7 development data and kept frozen.
This is a sensitivity analysis, not an independent confirmation.
