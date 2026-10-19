"""Phase 3 — non-invasive instrumentation of A2 (frozen config) on the 20
holdout seeds: decision counters for the freshness path, plus per-frame
traces of representative cases A-E (seed 3001), A2 vs A3 vs A4 on the
same realizations. Frozen code is not modified (EAML.step is wrapped)."""
import csv, json, sys
from collections import Counter, deque
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e11_camera_ready import posthoc as P
from edge_runtime import EAML, MHEGScheduler
from edge_runtime.ossl import OSSLayer
import edge_runtime.eaml as E
from experiments.e10_eaml.arms import CONFIDENCE, SAR_WINDOW, _respond, build_a2_policy, h_bin
from experiments.e10_eaml.scenarios import WINDOW_T


def run_variant(sd, variant, keep=None):
    """variant in {A2, A4, A3}. Returns counters and optional traces for
    episodes in `keep` (set of (persona, trip, slot))."""
    pol = build_a2_policy(P.A2C, P.MH)
    C = Counter(); traces = {}
    for persona, trip, episodes in sd.trips():
        guard = P._guard(pol); ta = 0
        for ep in episodes:
            key = (persona, trip, ep.slot)
            mheg = MHEGScheduler(pol); ossl = OSSLayer(pol)
            eaml = EAML(pol, mheg=mheg if variant == "A2" else None); eaml.guard = guard
            p_last, h_last = 0.0, 1.0; win = deque(maxlen=SAR_WINDOW)
            pending = False; last_force_served = -99; tr = []
            for t in range(WINDOW_T):
                d = mheg.decide(float(ep.dx[t]), h_last)
                if d.trigger_reason == "force_compute":
                    C["force_served"] += 1; pending = False; last_force_served = t
                if d.compute:
                    p_last = float(ep.p_obs[t]); h_last = h_bin(p_last)
                    ev = ossl.materialize(t, d, p_t=p_last, entropy_t=h_last, confidence_t=CONFIDENCE)
                else:
                    ev = ossl.materialize(t, d)
                if variant == "A3":   # suppression invisible: carried p as fresh state
                    ev = E.ZState(t=t, p=p_last, level=pol.level_of(p_last), entropy=h_last,
                                  delta_x=float(ep.dx[t]), confidence=CONFIDENCE)
                win.append(0 if d.compute else 1)
                dec = eaml.step(ev)
                rho12 = sum(win) / len(win)
                C["frames"] += 1; C["exec"] += int(d.compute)
                if dec.risk_reasons:
                    C["candidates"] += 1
                if dec.action == "force_compute_before_alert":
                    C["force_requested"] += 1; pending = True
                elif dec.action == "hold":
                    for b in dec.guard_blocks:
                        C[f"hold_{b}"] += 1
                elif dec.action == "alert":
                    C["alerts"] += 1; C["alerts_stale12"] += int(rho12 > 0.60)
                    C["alerts_rhoW_gt_060"] += int(dec.reused_ratio > 0.60)
                    if t - last_force_served <= 1:
                        C["alerts_within_1f_after_refresh"] += 1
                    eaml.observe_response(_respond(ep, ta)); ta += 1
                if keep and key in keep:
                    tr.append(dict(variant=variant, persona=persona, trip=trip, slot=ep.slot,
                                   recipe=ep.recipe, onset=ep.onset, t=t, mheg=d.trigger_reason,
                                   evidence="z" if d.compute else "OSS", p_used=round(p_last, 3),
                                   age=dec.last_compute_age, rho_W=round(dec.reused_ratio, 3),
                                   rho12=round(rho12, 3), R_E=round(dec.episode_risk, 3),
                                   risk="+".join(dec.risk_reasons), action=dec.action,
                                   blocks="+".join(dec.guard_blocks)))
            if pending:
                C["force_lost_at_episode_end"] += 1
            if keep and key in keep:
                traces[key] = tr
    return C, traces


def job(seed):
    sd = P.generate(seed)
    return {v: dict(run_variant(sd, v)[0]) for v in ("A2", "A4", "A3")}


def pick_cases(seed=3001):
    sd = P.generate(seed)
    _, tA2 = run_variant(sd, "A2", keep=None)
    # full traces once (cheap enough on one seed): keep all, then choose
    allkeys = {(p, tr, ep.slot) for p, tr, eps in sd.trips() for ep in eps}
    _, T2 = run_variant(sd, "A2", keep=allkeys)
    _, T3 = run_variant(sd, "A3", keep=allkeys)
    _, T4 = run_variant(sd, "A4", keep=allkeys)
    cases = {}
    def first(pred):
        for k in sorted(T2):
            if pred(k):
                return k
    cases["A_fresh_alert"] = first(lambda k: any(r["action"] == "alert" and r["age"] == 0 for r in T2[k]))
    cases["B_reuse_dominated_candidate"] = first(lambda k: any(r["risk"] and r["rho_W"] > 0.6 for r in T2[k]))
    def served_then_alert(k):
        rs = T2[k]
        for i, r in enumerate(rs[:-1]):
            if r["action"] == "force_compute_before_alert" and rs[i + 1]["mheg"] == "force_compute" \
               and any(x["action"] == "alert" for x in rs[i + 1:i + 3]):
                return True
    cases["C_refresh_served_then_alert"] = first(served_then_alert)
    cases["D_candidate_held"] = first(lambda k: any(r["action"] == "hold" and "evidence_quality" in r["blocks"] for r in T2[k])
                                      and not any(r["action"] == "alert" for r in T2[k]))
    cases["E_A3_stale_alert"] = first(lambda k: any(r["action"] == "alert" and r["rho12"] > 0.6 for r in T3[k]))
    rows = []
    for name, k in cases.items():
        if k is None:
            continue
        for T in (T2, T3, T4):
            for r in T[k]:
                rows.append(dict(case=name, **r))
    with open(P.OUT / "sar_trace_cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return {n: list(k) if k else None for n, k in cases.items()}


if __name__ == "__main__":
    cases = pick_cases()
    with Pool(2) as pool:
        per = pool.map(job, P.SPEC["holdout_seeds"])
    tot = {v: Counter() for v in ("A2", "A4", "A3")}
    for r in per:
        for v, c in r.items():
            tot[v].update(c)
    out = {"cases": cases, "counters_holdout": {v: dict(c) for v, c in tot.items()},
           "n_trips": 20 * 12 * 20}
    (P.OUT / "sar_trace_counters.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
