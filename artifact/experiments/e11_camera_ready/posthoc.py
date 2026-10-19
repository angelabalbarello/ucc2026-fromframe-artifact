"""E11 · camera-ready post-hoc analyses (UCC 2026, reviewer responses).

All analyses are POST-HOC. The frozen E10 configuration (spec_e10.json) is
never re-selected; the frozen E10 code is imported, not modified.

  R1  reproduction of the frozen holdout (A0-A3) from the artifact code;
  R2  uniform stale-alert accounting: rho measured for EVERY arm over the
      same external 12-frame window (E10 measured A2 over its own W=8 window);
  R3  effective-latency decomposition (detection delay vs. miss penalty);
  R4  computational cost: detector executions per trip, forced refreshes;
  R5  compact comparators: A5 = A2 with OSS reduced to reuse flag + age,
      A1F = frame-level EGS (A1) + reuse/age counter + forced refresh,
      A1 with smaller max_skip (more detector executions);
  R6  one-at-a-time sensitivity of W, m, k, theta_ep, MHEG thresholds and
      the observation-noise model (including AR(1) temporally correlated
      noise) on FRESH seeds 7001-7010;
  R7  per-frame decision trace of one episode.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random as _random
import sys
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from edge_runtime import EAML, MHEGScheduler, EventAwareGovernance
from edge_runtime.ossl import OSSLayer, OSSummary
from experiments.e10_eaml import arms, metrics, scenarios
from experiments.e10_eaml.arms import (CONFIDENCE, SAR_WINDOW, UNIT_FRAMES,
                                       _LEVELS, _episode_record, _mheg_policy,
                                       _respond, build_a2_policy, h_bin)
from experiments.e10_eaml.run_strongaccept import run_a4
from experiments.e10_eaml.scenarios import DT_S, WINDOW_T

E10 = _BACKEND / "experiments" / "e10_eaml" / "outputs"
OUT = Path(__file__).resolve().parent / "outputs"
SPEC = json.loads((E10 / "spec_e10.json").read_text(encoding="utf-8"))
PERS, NTR = SPEC["personas"], SPEC["n_trips"]
FRESH = list(range(7001, 7011))
AOSC = list(range(5001, 5011))


# ------------------------------------------------------------ generation
def generate(seed, osc=False, noise_gain=0.10, ar_rho=0.0):
    """scenarios.generate_seed with an optional observation-noise model.
    noise_gain=0.10, ar_rho=0 reproduces E10 bit-for-bit."""
    if noise_gain == 0.10 and ar_rho == 0.0:
        return scenarios.generate_seed(seed, PERS, NTR, osc_mix=osc)
    old = scenarios.OBS_NOISE_GAIN
    scenarios.OBS_NOISE_GAIN = noise_gain
    try:
        sd = scenarios.generate_seed(seed, PERS, NTR, osc_mix=osc)
    finally:
        scenarios.OBS_NOISE_GAIN = old
    if ar_rho:
        # replace white noise by AR(1) noise with the same marginal variance;
        # p_true is recovered exactly from a zero-noise regeneration.
        scenarios.OBS_NOISE_GAIN = 0.0
        try:
            clean = scenarios.generate_seed(seed, PERS, NTR, osc_mix=osc)
        finally:
            scenarios.OBS_NOISE_GAIN = old
        rng = np.random.default_rng([seed, 99])
        for ep, ec in zip(sd.episodes, clean.episodes):
            obs = scenarios.PERSONA_PROFILES[ep.persona][0]
            e = np.empty(WINDOW_T); e[0] = rng.standard_normal()
            s = np.sqrt(1 - ar_rho ** 2)
            for t in range(1, WINDOW_T):
                e[t] = ar_rho * e[t - 1] + s * rng.standard_normal()
            ep.p_obs = np.clip(ec.p_obs + e * noise_gain * (1.2 - obs), 0, 1)
    return sd


# ------------------------------------------------------------ helpers
def _guard(policy):
    return EventAwareGovernance(
        theta_off=policy.theta_off, min_realert_gap=policy.min_realert_gap,
        max_alerts_per_trip=policy.max_alerts_per_trip,
        escalation_delta=policy.escalation_delta,
        end_persistence=policy.end_persistence)


def _rec(ep, alerts, rho12, rho8, red, computed, force):
    r = _episode_record(ep, alerts, sum(1 for x in rho12 if x > 0.60), red,
                        computed, force)
    r["rho12"], r["rho_int"] = list(rho12), list(rho8)
    return r


# ------------------------------------------------------------ A2 instrumented
def run_a2x(sd, a2_cfg, mheg_cfg, minimal_oss=False, force=True,
            trace_key=None, **pol_over):
    """A2 with uniform external SAR window. Decision logic identical to
    arms.run_a2 (same MHEG/OSSL/EAML objects). minimal_oss=True replaces
    OSS_t by a record carrying only provenance (reuse flag) and the reused
    p: no confidence decay, quality always 'safe'."""
    policy = build_a2_policy(a2_cfg, mheg_cfg)
    if pol_over:
        policy = dataclasses.replace(policy, **pol_over)
    out, trace = [], []
    for persona, trip, episodes in sd.trips():
        guard = _guard(policy); trip_alerts = 0
        for ep in episodes:
            mheg = MHEGScheduler(policy); ossl = OSSLayer(policy)
            eaml = EAML(policy, mheg=mheg if force else None); eaml.guard = guard
            p_last, h_last = 0.0, 1.0
            win = deque(maxlen=SAR_WINDOW)
            alerts, r12, r8, red, comp, fc = [], [], [], 0, 0, 0
            for t in range(WINDOW_T):
                d = mheg.decide(float(ep.dx[t]), h_last)
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last); comp += 1
                    ev = ossl.materialize(t, d, p_t=p_last, entropy_t=h_last,
                                          confidence_t=CONFIDENCE)
                else:
                    ev = ossl.materialize(t, d)
                    if minimal_oss:
                        ev = dataclasses.replace(
                            ev, confidence_decay=CONFIDENCE,
                            suppression_quality="safe", entropy_carried=0.0,
                            cheap_risk_proxy=0.0, max_delta_since_last_compute=0.0,
                            trigger_margin_delta=0.0, trigger_margin_entropy=0.0)
                win.append(0 if d.compute else 1)
                dec = eaml.step(ev)
                if trace_key == (persona, trip, ep.slot):
                    trace.append(dict(t=t, mheg=d.trigger_reason,
                        p_obs=round(float(ep.p_obs[t]), 3), p_used=round(p_last, 3),
                        R_E=round(dec.episode_risk, 3), top_k=round(dec.mean_top_k, 3),
                        peak=round(dec.peak, 3), persistence=dec.persistence,
                        rho_W=round(dec.reused_ratio, 3), age=dec.last_compute_age,
                        rho12=round(sum(win) / len(win), 3),
                        risk="+".join(dec.risk_reasons), action=dec.action,
                        blocks="+".join(dec.guard_blocks), onset=ep.onset))
                if dec.action == "force_compute_before_alert":
                    fc += 1
                elif dec.action == "alert":
                    alerts.append(t); red += int(dec.redundant)
                    r12.append(sum(win) / len(win)); r8.append(dec.reused_ratio)
                    eaml.observe_response(_respond(ep, trip_alerts)); trip_alerts += 1
            out.append(_rec(ep, alerts, r12, r8, red, comp, fc))
    return (out, trace) if trace_key else out


# ------------------------------------------------------------ A1 family
def run_a1x(sd, cand, mheg_cfg, fresh=False, w_f=8, rho_max=0.60, age_max=4):
    """A1 (frozen EGS) with uniform SAR window; fresh=True adds the compact
    freshness comparator A1F: a reuse/age counter over the last w_f frames
    and the same forced-refresh rule as the EAML (hold + force when the
    guard's raw trigger fires on reuse-dominated evidence)."""
    pol = _mheg_policy(mheg_cfg)
    out = []
    for persona, _, episodes in sd.trips():
        guard = EventAwareGovernance(
            theta_off=cand["theta_off"],
            min_realert_gap=cand["min_realert_gap"] * UNIT_FRAMES,
            max_alerts_per_trip=cand["max_alerts_per_trip"],
            escalation_delta=cand["escalation_delta"],
            end_persistence=cand["end_persistence"] * UNIT_FRAMES)
        trip_alerts = 0
        for ep in episodes:
            mheg = MHEGScheduler(pol)
            p_last, h_last, age = 0.0, 1.0, 0
            win = deque(maxlen=SAR_WINDOW); wf = deque(maxlen=w_f)
            alerts, r12, r8, red, comp, fc = [], [], [], 0, 0, 0
            for t in range(WINDOW_T):
                d = mheg.decide(float(ep.dx[t]), h_last)
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last)
                    comp += 1; age = 0
                else:
                    age += 1
                win.append(0 if d.compute else 1); wf.append(0 if d.compute else 1)
                if fresh and p_last >= cand["theta_on"] and sum(wf) / len(wf) > rho_max:
                    if age > age_max:
                        mheg.request_force_compute(); fc += 1
                    continue                      # hold (guard not stepped)
                alert, rd = guard.step(p_last, cand["theta_on"], _LEVELS.level_of(p_last))
                if alert:
                    alerts.append(t); red += rd
                    r12.append(sum(win) / len(win)); r8.append(sum(wf) / len(wf))
                    guard.observe_response(int(_respond(ep, trip_alerts))); trip_alerts += 1
            out.append(_rec(ep, alerts, r12, r8, red, comp, fc))
    return out


def run_a3x(sd, a2_cfg, mheg_cfg):
    """Frozen A3 + per-alert rho12 (A3 already uses the 12-frame window)."""
    recs = arms.run_a3(sd, a2_cfg, mheg_cfg)
    return recs


# ------------------------------------------------------------ metrics
def clusters(recs, seed):
    cl = metrics.collect_clusters(recs, seed, PERS, NTR)
    by = {}
    for r in recs:
        by.setdefault(r["persona"], []).append(r)
    for c in cl:
        rs = by[c["persona"]]
        c["exec_per_trip"] = sum(r["computed"] for r in rs) / NTR
        det, miss = [], 0
        for r in rs:
            o = metrics.episode_outcomes(r)
            if o["critical"]:
                ta = metrics._first_alert_from_onset(r)
                if ta is None:
                    miss += 1
                else:
                    det.append((ta - r["onset"]) * DT_S)
        ncrit = len(det) + miss
        c["delay_detected"] = sum(det) / len(det) if det else None
        c["tta_flat10"] = (sum(det) + 10.0 * miss) / ncrit if ncrit else None
        rho12 = [x for r in rs for x in r.get("rho12", [])]
        rhoi = [x for r in rs for x in r.get("rho_int", [])]
        n = len(rho12)
        for thr in (0.50, 0.60, 0.70):
            c[f"sar12_{thr:.2f}"] = (sum(x > thr for x in rho12) / n) if n else 0.0
        c["sar_int"] = (sum(x > 0.60 for x in rhoi) / len(rhoi)) if rhoi else 0.0
        c["_rho12"] = rho12
    return cl


# ------------------------------------------------------------ jobs
A1C, A2C, MH = SPEC["a1_candidate"], SPEC["a2_config"], SPEC["mheg_config"]

SENS_EAML = ([("W", dict(window_w=w)) for w in (6, 10, 12)] +
             [("m", dict(m_persistence=m)) for m in (2, 4)] +
             [("k", dict(top_k=k)) for k in (2, 4)] +
             [("theta_ep", dict(theta_episode=x)) for x in (0.65, 0.75)] +
             [("theta_on", dict(theta_on=x, theta_off=round(x - 0.03, 2)))
              for x in (0.60, 0.70)] +
             [("rho_max", dict(reused_ratio_max=x)) for x in (0.50, 0.70)] +
             [("age_max", dict(max_skip_safe=x)) for x in (2, 6)])
SENS_MHEG = ([("tau_delta", dict(tau_delta=MH["tau_delta"] * f)) for f in (0.5, 2.0)] +
             [("tau_H", dict(tau_h=x)) for x in (0.5, 0.7)] +
             [("max_skip", dict(max_skip=x)) for x in (4, 12)])
SENS_NOISE = [("noise", dict(noise_gain=0.05)), ("noise", dict(noise_gain=0.20)),
              ("noise_AR1", dict(ar_rho=0.8)), ("noise_AR1", dict(ar_rho=0.95))]


def job_holdout(seed):
    sd = generate(seed)
    res = {"seed": seed, "hash": sd.exogenous_hash}
    res["A0"] = clusters(arms.run_a0(sd, SPEC["a0_config"], MH), seed)
    res["A1"] = clusters(run_a1x(sd, A1C, MH), seed)
    res["A2_frozen"] = clusters(arms.run_a2(sd, A2C, MH), seed)
    res["A2"] = clusters(run_a2x(sd, A2C, MH), seed)
    res["A5"] = clusters(run_a2x(sd, A2C, MH, minimal_oss=True), seed)  # A5 = A2 with age+reuse only
    res["A3"] = clusters(arms.run_a3(sd, A2C, MH), seed)
    res["A4"] = clusters(run_a2x(sd, A2C, MH, force=False), seed)
    res["A1F"] = clusters(run_a1x(sd, A1C, MH, fresh=True), seed)
    for ms in (6, 4, 2):
        res[f"A1_maxskip{ms}"] = clusters(run_a1x(sd, A1C, dict(MH, max_skip=ms)), seed)
    return res


def job_aosc(seed):
    sd = generate(seed, osc=True)
    res = {"seed": seed}
    res["A0"] = clusters(arms.run_a0(sd, SPEC["a0_config"], MH), seed)
    res["A1"] = clusters(run_a1x(sd, A1C, MH), seed)
    res["A2"] = clusters(run_a2x(sd, A2C, MH), seed)
    res["A3"] = clusters(arms.run_a3(sd, A2C, MH), seed)
    res["A4"] = clusters(run_a2x(sd, A2C, MH, force=False), seed)
    res["A1F"] = clusters(run_a1x(sd, A1C, MH, fresh=True), seed)
    return res


def job_sens(seed):
    res = {"seed": seed}
    sd = generate(seed)
    res["base|A1"] = clusters(run_a1x(sd, A1C, MH), seed)
    res["base|A2"] = clusters(run_a2x(sd, A2C, MH), seed)
    for name, over in SENS_EAML:
        key = f"{name}={list(over.values())[0]}"
        res[f"{key}|A2"] = clusters(run_a2x(sd, A2C, MH, **over), seed)
    for name, over in SENS_MHEG:
        mh = dict(MH, **over); key = f"{name}={list(over.values())[0]:.4g}"
        res[f"{key}|A1"] = clusters(run_a1x(sd, A1C, mh), seed)
        res[f"{key}|A2"] = clusters(run_a2x(sd, A2C, mh), seed)
    for name, over in SENS_NOISE:
        sdn = generate(seed, **over); key = f"{name}={list(over.values())[0]}"
        res[f"{key}|A1"] = clusters(run_a1x(sdn, A1C, MH), seed)
        res[f"{key}|A2"] = clusters(run_a2x(sdn, A2C, MH), seed)
    return res


def main(which):
    OUT.mkdir(exist_ok=True)
    seeds = {"holdout": SPEC["holdout_seeds"], "aosc": AOSC, "sens": FRESH}[which]
    fn = {"holdout": job_holdout, "aosc": job_aosc, "sens": job_sens}[which]
    t0 = time.time()
    with Pool(2) as pool:
        res = []
        for r in pool.imap_unordered(fn, seeds):
            res.append(r); print(f"[{which}] seed {r['seed']} {time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda r: r["seed"])
    (OUT / f"raw_{which}.json").write_text(json.dumps(res), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
