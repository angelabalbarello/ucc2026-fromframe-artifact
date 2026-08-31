"""Post-hoc mechanism and fairness analyses (Paper C, UCC 2026).

Three analyses, all executed AFTER the frozen confirmatory protocol and
labeled as post-hoc; none alters the pre-declared promotion decision.

S1 · A4 ablation (OSSL kept, force channel disabled) — separates SEEING
    staleness (OSS_t lets the guard withhold stale alerts passively) from
    ACTING on it (force_compute_before_alert restores freshness). Runs on
    the 20 holdout seeds and on the 10 A-OSC seeds with the frozen A2
    config; A1/A2/A3 cluster results are reused from the frozen outputs
    (the generator is deterministic per seed, so pairing is exact).

S2 · SAR threshold sensitivity — re-runs A1 and A2 on the holdout seeds
    capturing the per-alert reused ratio rho, then reports SAR at
    rho > {0.50, 0.60, 0.70}. Delivery behavior is unchanged (the guard
    still uses 0.60); only the accounting threshold varies.

S3 · Fairness check: re-optimized EGS — re-optimizes the EGS baseline on
    the SAME development budget as A2 (guard grid theta_on x A_max from
    the v2 amendment, dev seeds 201-205, selection: max recall subject to
    alerts/trip <= 2.5, tie-break lower FAR), freezes it, and evaluates
    the re-optimized EGS against the frozen A2 on FRESH seeds 6001-6010.
    The original holdout is not reused (it has been observed).
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from edge_runtime import EdgePolicy, MHEGScheduler, EAML
from edge_runtime.ossl import OSSLayer
from edge_runtime.event_guard import EventAwareGovernance

from experiments.e10_eaml import arms, metrics, scenarios
from experiments.e10_eaml.arms import (
    ARM_RUNNERS, SAR_WINDOW, UNIT_FRAMES, CONFIDENCE, _LEVELS,
    _episode_record, _respond, _mheg_policy, build_a2_policy, h_bin)
from experiments.e10_eaml.scenarios import WINDOW_T

OUT_DIR = Path(__file__).resolve().parent / "outputs"
SA_DIR = OUT_DIR / "strongaccept"
FRESH_SEEDS_S3 = list(range(6001, 6011))


# ---------------------------------------------------------------- S1 · A4
def run_a4(seed_data, a2_cfg: Dict, mheg_cfg: Dict) -> List[Dict]:
    """OSSL + EAML with the force channel DISABLED (mheg=None).

    Real z_t/OSS_t evidence flows into the window (unlike A3), so the
    guard still sees staleness and withholds stale alerts; it just cannot
    request fresh computation. force-intent decisions are counted."""
    policy = build_a2_policy(a2_cfg, mheg_cfg)
    out = []
    for persona, _, episodes in seed_data.trips():
        trip_guard = EventAwareGovernance(
            theta_off=policy.theta_off,
            min_realert_gap=policy.min_realert_gap,
            max_alerts_per_trip=policy.max_alerts_per_trip,
            escalation_delta=policy.escalation_delta,
            end_persistence=policy.end_persistence,
        )
        trip_alerts = 0
        for ep in episodes:
            mheg = MHEGScheduler(policy)
            ossl = OSSLayer(policy)
            eaml = EAML(policy, mheg=None)   # forca desativada
            eaml.guard = trip_guard
            p_last, h_last = 0.0, 1.0
            win = deque(maxlen=SAR_WINDOW)
            alerts, stale, redundant, computed, force = [], 0, 0, 0, 0
            for t in range(WINDOW_T):
                d = mheg.decide(float(ep.dx[t]), h_last)
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last)
                    computed += 1
                    ev = ossl.materialize(t, d, p_t=p_last, entropy_t=h_last,
                                          confidence_t=CONFIDENCE)
                else:
                    ev = ossl.materialize(t, d)
                win.append(0 if d.compute else 1)
                dec = eaml.step(ev)
                if dec.action == "force_compute_before_alert":
                    force += 1          # intencao sem canal (nao computa)
                elif dec.action == "alert":
                    alerts.append(t); redundant += int(dec.redundant)
                    if sum(win) / len(win) > 0.60:
                        stale += 1
                    eaml.observe_response(_respond(ep, trip_alerts))
                    trip_alerts += 1
            out.append(_episode_record(ep, alerts, stale, redundant,
                                       computed, force))
    return out


def s1_a4(spec: Dict) -> Dict:
    personas, n_trips = spec["personas"], spec["n_trips"]
    mheg_cfg, a2_cfg = spec["mheg_config"], spec["a2_config"]
    result = {}
    for label, seeds, osc in (("holdout", spec["holdout_seeds"], False),
                              ("aosc", list(range(5001, 5011)), True)):
        clusters = []
        for seed in seeds:
            sd = scenarios.generate_seed(seed, personas, n_trips,
                                         osc_mix=osc)
            recs = run_a4(sd, a2_cfg, mheg_cfg)
            clusters.extend(metrics.collect_clusters(recs, seed, personas,
                                                     n_trips))
            print(f"[S1-A4] {label} seed {seed} ok", flush=True)
        # pair against frozen A1/A2 clusters from the existing outputs
        src = OUT_DIR / ("results_holdout.csv" if not osc
                         else "results_aosc.csv")
        import csv as _csv
        ref = {"A1": [], "A2": []}
        with open(src, encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                if row["arm"] in ref:
                    ref[row["arm"]].append({
                        k: (None if v == "" else
                            (v if k in ("arm", "persona") else float(v)))
                        for k, v in row.items()})
        for r in ref.values():
            for c in r:
                c["seed"] = int(c["seed"])
        summary = {"clusters_mean": {
            k: (sum(c[k] for c in clusters if c[k] is not None)
                / max(1, sum(1 for c in clusters if c[k] is not None)))
            for k in ("recall", "tta_ef", "far_per_trip", "alerts_per_trip",
                      "sar", "redundant_per_trip", "force_per_trip")}}
        for other, rows in ref.items():
            summary[f"A4_vs_{other}"] = {
                f: metrics.paired_diff(clusters, rows, f)
                for f in ("recall", "tta_ef", "far_per_trip",
                          "alerts_per_trip", "sar")}
        result[label] = summary
    return result


# ------------------------------------------------- S2 · SAR sensitivity
def _run_a1_rho(seed_data, cand: Dict, mheg_cfg: Dict) -> List[float]:
    rhos = []
    for persona, _, episodes in seed_data.trips():
        guard = EventAwareGovernance(
            theta_off=cand["theta_off"],
            min_realert_gap=cand["min_realert_gap"] * UNIT_FRAMES,
            max_alerts_per_trip=cand["max_alerts_per_trip"],
            escalation_delta=cand["escalation_delta"],
            end_persistence=cand["end_persistence"] * UNIT_FRAMES,
        )
        trip_alerts = 0
        for ep in episodes:
            mheg = MHEGScheduler(_mheg_policy(mheg_cfg))
            p_last, h_last = 0.0, 1.0
            win = deque(maxlen=SAR_WINDOW)
            for t in range(WINDOW_T):
                d = mheg.decide(float(ep.dx[t]), h_last)
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last)
                win.append(0 if d.compute else 1)
                alert, _ = guard.step(p_last, cand["theta_on"],
                                      _LEVELS.level_of(p_last))
                if alert:
                    rhos.append(sum(win) / len(win))
                    guard.observe_response(int(_respond(ep, trip_alerts)))
                    trip_alerts += 1
    return rhos


def _run_a2_rho(seed_data, a2_cfg: Dict, mheg_cfg: Dict) -> List[float]:
    from edge_runtime import EdgeRuntime
    policy = build_a2_policy(a2_cfg, mheg_cfg)
    rhos = []
    for persona, _, episodes in seed_data.trips():
        trip_guard = EventAwareGovernance(
            theta_off=policy.theta_off,
            min_realert_gap=policy.min_realert_gap,
            max_alerts_per_trip=policy.max_alerts_per_trip,
            escalation_delta=policy.escalation_delta,
            end_persistence=policy.end_persistence,
        )
        trip_alerts = 0
        for ep in episodes:
            def infer_fn(t, _ep=ep):
                p = float(_ep.p_obs[t])
                return p, h_bin(p), CONFIDENCE
            rt = arms.EdgeRuntime(policy, infer_fn)
            rt.eaml.guard = trip_guard
            for t in range(WINDOW_T):
                dec = rt.process_frame(delta_x=float(ep.dx[t]))
                if dec.action == "alert":
                    rhos.append(float(dec.reused_ratio))
                    rt.observe_response(_respond(ep, trip_alerts))
                    trip_alerts += 1
    return rhos


def s2_sar_sensitivity(spec: Dict) -> Dict:
    personas, n_trips = spec["personas"], spec["n_trips"]
    mheg_cfg = spec["mheg_config"]
    rhos = {"A1": [], "A2": []}
    for seed in spec["holdout_seeds"]:
        sd = scenarios.generate_seed(seed, personas, n_trips)
        rhos["A1"].extend(_run_a1_rho(sd, spec["a1_candidate"], mheg_cfg))
        rhos["A2"].extend(_run_a2_rho(sd, spec["a2_config"], mheg_cfg))
        print(f"[S2-SAR] seed {seed} ok", flush=True)
    out = {}
    for arm, vals in rhos.items():
        n = len(vals)
        out[arm] = {"n_alerts": n}
        for thr in (0.50, 0.60, 0.70):
            out[arm][f"sar_at_{thr:.2f}"] = (
                sum(1 for r in vals if r > thr) / n if n else 0.0)
    return out


# ------------------------------------------------- S3 · fair EGS re-opt
def s3_fair_egs(spec: Dict) -> Dict:
    personas, n_trips = spec["personas"], spec["n_trips"]
    mheg_cfg, a2_cfg = spec["mheg_config"], spec["a2_config"]
    base = spec["a1_candidate"]
    grid = [dict(base, theta_on=on, theta_off=round(on - 0.03, 2),
                 max_alerts_per_trip=amax)
            for on in (0.55, 0.60, 0.65) for amax in (3, 4, 6)]

    dev_seeds = (201, 202, 203, 204, 205)
    dev_stats = []
    for gi, cand in enumerate(grid):
        clusters = []
        for seed in dev_seeds:
            sd = scenarios.generate_seed(seed, personas, n_trips)
            recs = arms.run_a1(sd, cand, mheg_cfg)
            clusters.extend(metrics.collect_clusters(recs, seed, personas,
                                                     n_trips))
        def _m(k):
            vals = [c[k] for c in clusters if c[k] is not None]
            return sum(vals) / len(vals)
        dev_stats.append({"idx": gi, "cand": cand, "recall": _m("recall"),
                          "far": _m("far_per_trip"),
                          "alerts": _m("alerts_per_trip")})
        print(f"[S3-dev] guard {gi+1}/9 alerts={dev_stats[-1]['alerts']:.2f} "
              f"recall={dev_stats[-1]['recall']:.3f}", flush=True)

    admissible = [s for s in dev_stats if s["alerts"] <= 2.5]
    pool = admissible if admissible else dev_stats
    winner = sorted(pool, key=lambda s: (-s["recall"], s["far"]))[0]

    fresh = {"A1R": [], "A2": []}
    for seed in FRESH_SEEDS_S3:
        sd = scenarios.generate_seed(seed, personas, n_trips)
        fresh["A1R"].extend(metrics.collect_clusters(
            arms.run_a1(sd, winner["cand"], mheg_cfg), seed, personas,
            n_trips))
        fresh["A2"].extend(metrics.collect_clusters(
            arms.run_a2(sd, a2_cfg, mheg_cfg), seed, personas, n_trips))
        print(f"[S3-fresh] seed {seed} ok", flush=True)

    def _m(rows, k):
        vals = [c[k] for c in rows if c[k] is not None]
        return sum(vals) / len(vals)

    return {
        "grid_dev": [{k: v for k, v in s.items() if k != "idx"}
                     for s in dev_stats],
        "winner": winner["cand"],
        "winner_dev": {k: winner[k] for k in ("recall", "far", "alerts")},
        "fresh_seeds": FRESH_SEEDS_S3,
        "fresh_means": {
            arm: {k: _m(rows, k) for k in
                  ("recall", "tta_ef", "far_per_trip", "alerts_per_trip",
                   "sar", "redundant_per_trip")}
            for arm, rows in fresh.items()},
        "A2_vs_A1R": {
            f: metrics.paired_diff(fresh["A2"], fresh["A1R"], f)
            for f in ("recall", "tta_ef", "far_per_trip", "alerts_per_trip",
                      "sar")},
    }


def main():
    SA_DIR.mkdir(exist_ok=True)
    spec = json.loads((OUT_DIR / "spec_e10.json").read_text(encoding="utf-8"))
    t0 = time.time()

    res = {"analysis": "post-hoc mechanism/fairness (S1 A4, S2 SAR, S3 EGS)",
           "note": "frozen A2 config; no reselection; original holdout not "
                   "treated as new confirmation"}
    res["S1_a4"] = s1_a4(spec)
    print(f"[S1] done {time.time()-t0:.0f}s", flush=True)
    res["S2_sar_sensitivity"] = s2_sar_sensitivity(spec)
    print(f"[S2] done {time.time()-t0:.0f}s", flush=True)
    res["S3_fair_egs"] = s3_fair_egs(spec)
    print(f"[S3] done {time.time()-t0:.0f}s", flush=True)

    out = SA_DIR / "strongaccept_result.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2)[:4000])


if __name__ == "__main__":
    main()
