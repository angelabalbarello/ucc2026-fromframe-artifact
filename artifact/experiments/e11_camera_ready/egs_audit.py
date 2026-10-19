"""E11 · EGS temporal-advancement audit (post-acceptance, exploratory).

Question: the same EventAwareGovernance (EGS) is stepped on EVERY frame in
A1 but only on EAML *candidate* frames in A2 (RISK true and not held). Its
clocks (min_realert_gap, end_persistence) and its event-closing logic
therefore advance on different time bases. How much of A2's FAR/alert-load
advantage depends on that?

Instrumentation (frozen classes are subclassed, never edited):
  * InstrumentedGuard records every step: opened event, re-alert, blocked;
    whether an event was already active when an episode starts (carry-over
    across episodes); wall-clock frames between consecutive alerts.
  * TickingGuard adds `tick(score)`, which advances the clocks and the
    closing logic WITHOUT the possibility of opening/delivering. It is
    called on every frame where the EAML did not step the guard.

Arms (holdout seeds, CRN with all other arms):
  A1      frozen semantics (guard stepped every frame on p_t)
  A2      frozen semantics (guard stepped on candidate frames only)
  A2T     A2 + wall-clock guard time (tick on non-candidate frames)
  A1C     A1 whose guard is stepped only on frames with p_t >= theta_on
          (candidate-only time base transplanted into A1; ticks elsewhere
          are NOT applied) -- the converse contrast
"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections import Counter, deque
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from edge_runtime import EAML, MHEGScheduler, EventAwareGovernance
from edge_runtime.ossl import OSSLayer
from experiments.e10_eaml import metrics
from experiments.e10_eaml.arms import (CONFIDENCE, SAR_WINDOW, UNIT_FRAMES, _LEVELS,
                                       _mheg_policy, _respond, build_a2_policy, h_bin)
from experiments.e10_eaml.scenarios import WINDOW_T
from experiments.e11_camera_ready import posthoc as P


class InstrumentedGuard(EventAwareGovernance):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.C = Counter()
        self.wall = 0                 # wall-clock frame counter (set by runner)
        self.last_alert_wall = None
        self.gaps = []

    def step(self, score, theta, rclass, governed=True):
        was_active = self._active
        self.C["steps"] += 1
        alert, red = super().step(score, theta, rclass, governed)
        if alert:
            self.C["alert_open" if not was_active else "alert_realert"] += 1
            if self.last_alert_wall is not None:
                self.gaps.append(self.wall - self.last_alert_wall)
            self.last_alert_wall = self.wall
        elif score >= theta:
            if self._alerts >= self.max_alerts_per_trip:
                self.C["blocked_budget"] += 1
            elif self._active:
                self.C["blocked_active_event"] += 1
            else:
                self.C["blocked_gap"] += 1
        return alert, red


class TickingGuard(InstrumentedGuard):
    def tick(self, score):
        """Advance clocks/closing as step() would, but never open/deliver."""
        self.C["ticks"] += 1
        self._since_last += 1
        if self._active:
            if score < self.theta_off:
                self._below_count += 1
                if self._below_count >= self.end_persistence:
                    self._active = False
                    self._below_count = 0
            else:
                self._below_count = 0


def _mk(cls, theta_off, gap, amax, esc, endp):
    return cls(theta_off=theta_off, min_realert_gap=gap, max_alerts_per_trip=amax,
               escalation_delta=esc, end_persistence=endp)


def run_a2_variant(sd, tick=False):
    pol = build_a2_policy(P.A2C, P.MH)
    out, C, gaps = [], Counter(), []
    for persona, trip, episodes in sd.trips():
        g = _mk(TickingGuard if tick else InstrumentedGuard, pol.theta_off,
                pol.min_realert_gap, pol.max_alerts_per_trip, pol.escalation_delta,
                pol.end_persistence)
        ta = 0
        for ep in episodes:
            C["episodes"] += 1; C["episode_starts_with_active_event"] += int(g._active)
            mheg = MHEGScheduler(pol); ossl = OSSLayer(pol)
            eaml = EAML(pol, mheg=mheg); eaml.guard = g
            p_last, h_last = 0.0, 1.0; win = deque(maxlen=SAR_WINDOW)
            alerts, r12, red, comp, fc = [], [], 0, 0, 0
            for t in range(WINDOW_T):
                g.wall += 1
                d = mheg.decide(float(ep.dx[t]), h_last)
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last); comp += 1
                    ev = ossl.materialize(t, d, p_t=p_last, entropy_t=h_last, confidence_t=CONFIDENCE)
                else:
                    ev = ossl.materialize(t, d)
                win.append(0 if d.compute else 1)
                steps_before = g.C["steps"]
                dec = eaml.step(ev)
                if tick and g.C["steps"] == steps_before:
                    g.tick(dec.episode_risk)
                if dec.action == "force_compute_before_alert":
                    fc += 1
                elif dec.action == "alert":
                    alerts.append(t); red += int(dec.redundant)
                    r12.append(sum(win) / len(win))
                    eaml.observe_response(_respond(ep, ta)); ta += 1
            out.append(P._rec(ep, alerts, r12, [], red, comp, fc))
        C.update(g.C); gaps += g.gaps
    return out, C, gaps


def run_a1_variant(sd, candidate_only=False):
    cand = P.A1C
    pol = _mheg_policy(P.MH)
    out, C, gaps = [], Counter(), []
    for persona, trip, episodes in sd.trips():
        g = _mk(InstrumentedGuard, cand["theta_off"], cand["min_realert_gap"] * UNIT_FRAMES,
                cand["max_alerts_per_trip"], cand["escalation_delta"],
                cand["end_persistence"] * UNIT_FRAMES)
        ta = 0
        for ep in episodes:
            C["episodes"] += 1; C["episode_starts_with_active_event"] += int(g._active)
            mheg = MHEGScheduler(pol)
            p_last, h_last = 0.0, 1.0; win = deque(maxlen=SAR_WINDOW)
            alerts, r12, red, comp = [], [], 0, 0
            for t in range(WINDOW_T):
                g.wall += 1
                d = mheg.decide(float(ep.dx[t]), h_last)
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last); comp += 1
                win.append(0 if d.compute else 1)
                if candidate_only and p_last < cand["theta_on"]:
                    continue
                alert, rd = g.step(p_last, cand["theta_on"], _LEVELS.level_of(p_last))
                if alert:
                    alerts.append(t); red += rd; r12.append(sum(win) / len(win))
                    g.observe_response(int(_respond(ep, ta))); ta += 1
            out.append(P._rec(ep, alerts, r12, [], red, comp, 0))
        C.update(g.C); gaps += g.gaps
    return out, C, gaps


def job(seed):
    sd = P.generate(seed)
    res = {"seed": seed, "counters": {}, "gaps": {}}
    for name, fn in (("A1", lambda: run_a1_variant(sd)),
                     ("A1C", lambda: run_a1_variant(sd, candidate_only=True)),
                     ("A2", lambda: run_a2_variant(sd)),
                     ("A2T", lambda: run_a2_variant(sd, tick=True))):
        recs, C, gaps = fn()
        res[name] = P.clusters(recs, seed)
        res["counters"][name] = dict(C)
        res["gaps"][name] = gaps
    return res


def main():
    seeds = P.SPEC["holdout_seeds"]
    with Pool(2) as pool:
        raw = pool.map(job, seeds)
    arms = ("A1", "A1C", "A2", "A2T")
    cl = {a: [c for r in raw for c in r[a]] for a in arms}
    for a in arms:
        for c in cl[a]:
            c.pop("_rho12", None)
    C = {a: Counter() for a in arms}; G = {a: [] for a in arms}
    for r in raw:
        for a in arms:
            C[a].update(r["counters"][a]); G[a] += r["gaps"][a]
    import statistics as st
    summ = {"means": {a: {f: P_mean(cl[a], f) for f in
                          ("recall", "tta_ef", "far_per_trip", "alerts_per_trip", "sar",
                           "redundant_per_trip", "exec_per_trip")} for a in arms},
            "counters": {a: dict(C[a]) for a in arms},
            "inter_alert_wall_frames": {a: {"n": len(G[a]), "median": st.median(G[a]),
                                            "p10": sorted(G[a])[len(G[a]) // 10],
                                            "share_lt_20": sum(g < 20 for g in G[a]) / len(G[a])}
                                        for a in arms if G[a]},
            "paired": {}}
    for x, y in (("A2T", "A2"), ("A2T", "A1"), ("A2", "A1"), ("A1C", "A1"), ("A2T", "A1C"), ("A2", "A1C")):
        summ["paired"][f"{x}-{y}"] = {f: metrics.paired_diff(cl[x], cl[y], f) for f in
                                      ("recall", "tta_ef", "far_per_trip", "alerts_per_trip",
                                       "sar", "redundant_per_trip")}
    summ["gates_vs_A1"] = {a: {k: v for k, v in metrics.gates_h2(cl[a], cl["A1"], 2.5, 0.02, 1.0).items()
                               if k != "diagnostics"} for a in ("A2", "A2T", "A1C")}
    (P.OUT / "egs_audit.json").write_text(json.dumps(summ, indent=1, default=str))
    (P.OUT / "raw_egs_audit.json").write_text(json.dumps({a: cl[a] for a in arms}))
    print(json.dumps({k: summ[k] for k in ("means", "counters", "inter_alert_wall_frames")}, indent=1))
    for k, d in summ["paired"].items():
        print(k, {f: f"{v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for f, v in d.items()})
    for a, g in summ["gates_vs_A1"].items():
        print("gates", a, g["admissible"], [k for k, v in g.items() if v is False])


def P_mean(cl, f):
    v = [c[f] for c in cl if c.get(f) is not None]
    return sum(v) / len(v) if v else None


if __name__ == "__main__":
    main()
