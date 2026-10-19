"""EGS time base: A1 steps the guard every frame, A2 only on EAML candidate
frames. These tests pin the executed semantics and the exploratory
TickingGuard used by E11 (A2T), which must never deliver on a tick."""
from edge_runtime import EventAwareGovernance
from experiments.e11_camera_ready.egs_audit import TickingGuard


def _g(cls=EventAwareGovernance):
    return cls(theta_off=0.62, min_realert_gap=20, max_alerts_per_trip=4,
               escalation_delta=0.10, end_persistence=10)


def test_event_stays_open_while_guard_not_stepped():
    g = _g()
    alert, _ = g.step(0.9, 0.65, "Critical")
    assert alert == 1 and g._active
    # no further steps (A2 outside candidate frames): nothing can close it
    assert g._active and g._below_count == 0


def test_event_closes_after_end_persistence_low_steps():
    g = _g()
    g.step(0.9, 0.65, "Critical")
    for _ in range(10):
        g.step(0.1, 0.65, "Normal")
    assert not g._active


def test_ticks_close_event_and_advance_gap_but_never_alert():
    g = _g(TickingGuard)
    g.step(0.9, 0.65, "Critical")
    n = g._alerts
    for _ in range(25):
        g.tick(0.1)
    assert not g._active and g._since_last >= 25 and g._alerts == n


def test_reopen_requires_gap_in_guard_steps():
    g = _g()
    g.step(0.9, 0.65, "Critical")
    for _ in range(10):
        g.step(0.1, 0.65, "Normal")          # closes, since_last = 10
    a, _ = g.step(0.9, 0.65, "Critical")
    assert a == 0                             # gap 20 not yet reached
