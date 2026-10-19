"""Operational invariants of MHEG / OSSL / EAML / EGS as implemented."""
import dataclasses
import random
import pytest
from edge_runtime import (EAML, EdgePolicy, EdgeRuntime, EventAwareGovernance,
                          MHEGScheduler, OSSLayer, ZState)
from edge_runtime.ossl import OSSummary

POL = dataclasses.replace(EdgePolicy(), tau_delta=0.005, window_w=8, theta_conf=0.30,
                          theta_on=0.65, theta_off=0.62, min_realert_gap=20,
                          max_alerts_per_trip=4, end_persistence=10)


# ------------------------------------------------------------------ MHEG
def test_max_skip_bounds_consecutive_reuse():
    m = MHEGScheduler(POL)
    seq = [m.decide(0.0, 0.0).compute for _ in range(30)]
    runs, r = [], 0
    for c in seq:
        if c:
            runs.append(r); r = 0
        else:
            r += 1
    assert max(runs + [r]) <= POL.max_skip


def test_force_request_is_served_on_next_decide_only_once():
    m = MHEGScheduler(POL)
    m.decide(1.0, 1.0)
    m.request_force_compute()
    d = m.decide(0.0, 0.0)
    assert d.compute and d.trigger_reason == "force_compute"
    assert not m.decide(0.0, 0.0).compute          # consumed


def test_runtime_frame0_computes():
    calls = []
    rt = EdgeRuntime(POL, lambda t: (calls.append(t) or (0.1, 0.5, 0.95)))
    rt.process_frame(delta_x=0.0)
    assert calls == [0]


# ------------------------------------------------------------------ OSSL
def test_skipped_frame_never_accepts_new_probability():
    o = OSSLayer(POL); m = MHEGScheduler(POL)
    o.materialize(0, m.decide(1.0, 1.0), p_t=0.4, entropy_t=0.9)
    d = m.decide(0.0, 0.0)
    with pytest.raises(ValueError):
        o.materialize(1, d, p_t=0.5, entropy_t=0.9)
    oss = o.materialize(1, d)
    assert isinstance(oss, OSSummary) and oss.carried_p == 0.4 and oss.infer_status == "reuse"


def test_first_frame_cannot_be_suppressed():
    o = OSSLayer(POL); m = MHEGScheduler(POL)
    m._suppression_count = 0
    d = dataclasses.replace(m.decide(0.0, 0.0), compute=False)
    with pytest.raises(ValueError):
        o.materialize(0, d)


def test_one_evidence_object_per_frame_and_unsafe_unreachable():
    """Through the real MHEG, a suppressed frame has dx < tau_delta and
    elapsed <= max_skip, so quality 'unsafe' can never be produced."""
    rng = random.Random(7)
    m, o = MHEGScheduler(POL), OSSLayer(POL)
    p, h = 0.0, 1.0
    for t in range(5000):
        dx = rng.choice([0.0, 0.001, 0.0045, 0.01])
        d = m.decide(dx, h)
        if d.compute:
            p = rng.random(); h = 0.3
            ev = o.materialize(t, d, p_t=p, entropy_t=h)
            assert isinstance(ev, ZState)
        else:
            ev = o.materialize(t, d)
            assert ev.suppression_quality != "unsafe"
            assert ev.elapsed_since_compute <= POL.max_skip


# ------------------------------------------------------------------ EAML
def _z(t, p):
    return ZState(t=t, p=p, level=POL.level_of(p), entropy=0.5, delta_x=0.1, confidence=0.95)


def _oss(t, p, frm, el):
    return OSSummary(t=t, infer_status="reuse", reused_from=frm, elapsed_since_compute=el,
                     delta_x=0.0, max_delta_since_last_compute=0.0, entropy_carried=0.5,
                     carried_p=p, confidence_decay=0.9, cheap_risk_proxy=p,
                     trigger_margin_delta=-0.005, trigger_margin_entropy=-0.1,
                     suppression_count=el, suppression_quality="safe")


def test_reuse_dominated_young_window_is_held():
    e = EAML(POL)
    for t in range(3):
        e.step(_z(t, 0.9))
    out = [e.step(_oss(t, 0.9, 2, t - 2)).action for t in range(3, 8)]
    # window 8: 3 z + 5 OSS -> rho=0.625 > 0.60, age 5 > 4 -> force
    assert out[-1] in ("force_compute_before_alert", "hold")


def test_force_issued_when_reuse_dominated_and_old_and_no_alert_that_frame():
    m = MHEGScheduler(POL); e = EAML(POL, mheg=m)
    e.step(_z(0, 0.95))
    acts = [e.step(_oss(t, 0.95, 0, t)) for t in range(1, 8)]
    last = acts[-1]
    assert last.reused_ratio > 0.60 and last.last_compute_age > 4
    assert last.action == "force_compute_before_alert" and m._force_requested


def test_no_alert_possible_with_rho_W_above_060():
    """SAR measured on the EAML window is zero by construction."""
    rng = random.Random(3)
    for _ in range(200):
        e = EAML(POL)
        for t in range(40):
            ev = _z(t, rng.random()) if rng.random() < 0.3 else _oss(t, rng.random(), 0, 3)
            d = e.step(ev)
            if d.action == "alert":
                assert d.reused_ratio <= 0.60


def test_guard_is_stepped_only_on_candidate_frames():
    class G(EventAwareGovernance):
        n = 0
        def step(self, *a, **k):
            G.n += 1
            return super().step(*a, **k)
    e = EAML(POL); e.guard = G(theta_off=0.62, min_realert_gap=20, max_alerts_per_trip=4)
    for t in range(20):
        e.step(_z(t, 0.05))           # no RISK
    assert G.n == 0


def test_e2_single_high_frame_is_not_blocked_by_code():
    """Characterisation: E2 (single high frame cannot trigger) is NOT enforced
    by the implementation; a lone p=1.0 among low values can still be a
    candidate via the peak clause (documented divergence D6)."""
    e = EAML(POL)
    for t in range(7):
        e.step(_z(t, 0.25))
    d = e.step(_z(7, 1.0))
    assert "peak" in d.risk_reasons
