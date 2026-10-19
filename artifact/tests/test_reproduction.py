"""Reproduction of frozen results and of the E11 corrections (slow)."""
import csv
import json
from pathlib import Path
import pytest

ART = Path(__file__).resolve().parents[1]
OUT = ART / "experiments" / "e10_eaml" / "outputs"


@pytest.fixture(scope="module")
def seed3001():
    from experiments.e11_camera_ready import posthoc as P
    return P.generate(3001)


@pytest.mark.slow
def test_dev_seed_exogenous_hash_matches_spec():
    from experiments.e11_camera_ready import posthoc as P
    spec = json.loads((OUT / "spec_e10.json").read_text(encoding="utf-8"))
    assert P.generate(201).exogenous_hash == spec["dev_exogenous_hashes"]["201"]


@pytest.mark.slow
def test_frozen_arms_reproduce_results_holdout_seed3001(seed3001):
    from experiments.e10_eaml import arms, metrics
    from experiments.e11_camera_ready import posthoc as P
    fro = {(r["arm"], r["persona"]): r for r in csv.DictReader(open(OUT / "results_holdout.csv"))
           if r["seed"] == "3001"}
    cfg = {"A0": P.SPEC["a0_config"], "A1": P.A1C, "A2": P.A2C, "A3": P.A2C}
    for arm, cfgv in cfg.items():
        cl = metrics.collect_clusters(arms.ARM_RUNNERS[arm](seed3001, cfgv, P.MH), 3001, P.PERS, P.NTR)
        for c in cl:
            f = fro[(arm, c["persona"])]
            for k in ("recall", "tta_ef", "far_per_trip", "alerts_per_trip", "sar",
                      "redundant_per_trip", "skip_pct"):
                assert abs(float(f[k]) - c[k]) < 1e-12, (arm, c["persona"], k)


@pytest.mark.slow
def test_uniform_sar_correction_and_a5_equivalence(seed3001):
    from experiments.e10_eaml import metrics
    from experiments.e11_camera_ready import posthoc as P
    a2 = P.clusters(P.run_a2x(seed3001, P.A2C, P.MH), 3001)
    a5 = P.clusters(P.run_a2x(seed3001, P.A2C, P.MH, minimal_oss=True), 3001)
    assert abs(metrics.summary_row("A2", a2)["sar"] - 0.0035) < 1e-4    # uniform window
    assert all(c["sar_int"] == 0.0 for c in a2)                         # own window: 0
    for x, y in zip(a2, a5):
        for k in ("recall", "tta_ef", "far_per_trip", "alerts_per_trip", "sar", "exec_per_trip"):
            assert x[k] == y[k]


@pytest.mark.slow
def test_inference_counter_equals_compute_decisions(seed3001):
    from experiments.e11_camera_ready import posthoc as P
    recs = P.run_a2x(seed3001, P.A2C, P.MH)
    assert all(0 < r["computed"] <= 96 for r in recs)
    a4 = P.run_a2x(seed3001, P.A2C, P.MH, force=False)
    extra = sum(r["computed"] for r in recs) - sum(r["computed"] for r in a4)
    assert 0 <= extra <= sum(r["force_compute"] for r in recs)
