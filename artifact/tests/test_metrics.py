"""Metric definitions exactly as executed (E10 metrics.py)."""
import math
import pytest
from experiments.e10_eaml import metrics
from experiments.e10_eaml.scenarios import DT_S, WINDOW_T


def rec(alerts, onset=48, cat="Critico", prog=False, persona="P01"):
    return {"persona": persona, "trip": 0, "slot": 0, "recipe": "X", "categoria": cat,
            "progressive": prog, "onset": onset, "alerts": alerts, "n_alerts": len(alerts),
            "stale_alerts": 0, "redundant_alerts": 0, "computed": 96, "reused": 0,
            "force_compute": 0}


def test_detected_delay_is_first_alert_at_or_after_onset():
    o = metrics.episode_outcomes(rec([30, 60, 70], onset=48))
    assert o["detected"] and math.isclose(o["tta_ef"], 12 * DT_S)
    assert o["false_alerts"] == 1                       # alert at 30 < onset


def test_alert_exactly_at_onset_counts_with_zero_delay():
    o = metrics.episode_outcomes(rec([48], onset=48))
    assert o["detected"] and o["tta_ef"] == 0.0


def test_miss_penalty_is_censored_delay_plus_10s_not_10s():
    """Documents the executed metric (Alt A): (T - onset)*dt + T_max."""
    o = metrics.episode_outcomes(rec([], onset=48))
    assert not o["detected"]
    assert math.isclose(o["tta_ef"], (WINDOW_T - 48) * DT_S + 10.0)
    assert math.isclose(o["tta_ef"], 15.0)
    late = metrics.episode_outcomes(rec([], onset=95))
    assert 10.0 < late["tta_ef"] <= 20.0


def test_alerts_only_before_onset_are_a_miss():
    o = metrics.episode_outcomes(rec([10, 20], onset=48))
    assert not o["detected"] and o["false_alerts"] == 2


def test_non_critical_alerts_are_false_alerts():
    o = metrics.episode_outcomes(rec([5, 50], onset=-1, cat="Alerta"))
    assert not o["critical"] and o["false_alerts"] == 2


def test_cluster_sar_and_zero_alert_convention():
    r = rec([50, 60], onset=48); r["stale_alerts"] = 1
    c = metrics.cluster_metrics([r], "P01", 1, n_trips=1)
    assert c["sar"] == 0.5
    r0 = rec([], onset=48)
    assert metrics.cluster_metrics([r0], "P01", 1, 1)["sar"] == 0.0


def test_paired_diff_normal_approximation():
    a = [{"persona": "P", "seed": s, "x": v} for s, v in enumerate([1.0, 2.0, 3.0, 4.0])]
    b = [{"persona": "P", "seed": s, "x": 0.0} for s in range(4)]
    d = metrics.paired_diff(a, b, "x")
    sd = math.sqrt(sum((v - 2.5) ** 2 for v in [1, 2, 3, 4]) / 3)
    se = sd / 2
    assert math.isclose(d["mean"], 2.5) and math.isclose(d["se"], se)
    assert math.isclose(d["u90"], 2.5 + 1.645 * se)
    assert math.isclose(d["ci95"][0], 2.5 - 1.96 * se)
