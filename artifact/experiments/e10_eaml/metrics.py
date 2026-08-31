"""E10 · metrics — métricas (§5), agregação por cluster e portões (§7).

Unidade de atribuição: cluster persona×seed. Toda inferência usa diferenças
pareadas candidato − referência agregadas por cluster (nunca episódio solto).
U90 segue a convenção de E9: limite superior unilateral com z=1,645.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .scenarios import WINDOW_T, DT_S, TRIP_SLOTS

T_MAX_S = 10.0
Z_ONE_SIDED = 1.645
Z_TWO_SIDED = 1.96


# ------------------------------------------------------------ por episódio
def _first_alert_from_onset(rec: Dict):
    return next((t for t in rec["alerts"] if t >= rec["onset"]), None)


def episode_outcomes(rec: Dict) -> Dict:
    """Deriva desfechos de um episódio (crítico: detecção/TTA; senão: FAs)."""
    out = {"critical": rec["categoria"] == "Critico" and rec["onset"] >= 0}
    if out["critical"]:
        ta = _first_alert_from_onset(rec)
        detected = ta is not None
        tta = ((ta - rec["onset"]) * DT_S if detected
               else (WINDOW_T - rec["onset"]) * DT_S)
        out.update(detected=detected,
                   tta_ef=tta + (0.0 if detected else T_MAX_S),
                   false_alerts=sum(1 for t in rec["alerts"] if t < rec["onset"]))
    else:
        out.update(detected=None, tta_ef=None,
                   false_alerts=rec["n_alerts"])
    return out


# ------------------------------------------------------------ por cluster
def cluster_metrics(records: List[Dict], persona: str, seed: int,
                    n_trips: int) -> Dict:
    recs = [r for r in records if r["persona"] == persona]
    crit = {"all": [], "abrupt": [], "progressive": []}
    fa = 0
    n_alerts = stale = redundant = force = computed = 0
    for r in recs:
        o = episode_outcomes(r)
        n_alerts += r["n_alerts"]; stale += r["stale_alerts"]
        redundant += r["redundant_alerts"]; force += r["force_compute"]
        computed += r["computed"]; fa += o["false_alerts"]
        if o["critical"]:
            regime = "progressive" if r["progressive"] else "abrupt"
            crit["all"].append(o); crit[regime].append(o)

    def _recall(lst):
        return (sum(1 for o in lst if o["detected"]) / len(lst)) if lst else None

    def _tta(lst):
        return (sum(o["tta_ef"] for o in lst) / len(lst)) if lst else None

    total_frames = len(recs) * WINDOW_T
    return {
        "persona": persona, "seed": seed, "n_episodes": len(recs),
        "recall": _recall(crit["all"]),
        "recall_abrupt": _recall(crit["abrupt"]),
        "recall_progressive": _recall(crit["progressive"]),
        "tta_ef": _tta(crit["all"]),
        "tta_ef_abrupt": _tta(crit["abrupt"]),
        "tta_ef_progressive": _tta(crit["progressive"]),
        "far_per_trip": fa / n_trips,
        "alerts_per_trip": n_alerts / n_trips,
        "sar": (stale / n_alerts) if n_alerts else 0.0,
        "redundant_per_trip": redundant / n_trips,
        "force_per_trip": force / n_trips,
        "skip_pct": 1.0 - computed / total_frames if total_frames else 0.0,
    }


def collect_clusters(records: List[Dict], seed: int, personas: List[str],
                     n_trips: int) -> List[Dict]:
    return [cluster_metrics(records, p, seed, n_trips) for p in personas]


# ------------------------------------------------------------ pareado
def paired_diff(cand: List[Dict], ref: List[Dict], field: str) -> Dict:
    """Δ = candidato − referência por cluster (persona, seed) pareado."""
    ref_by = {(c["persona"], c["seed"]): c for c in ref}
    diffs = []
    for c in cand:
        r = ref_by[(c["persona"], c["seed"])]
        if c[field] is None or r[field] is None:
            continue
        diffs.append(c[field] - r[field])
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "u90": None, "ci95": None}
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return {"n": n, "mean": mean, "se": se,
            "u90": mean + Z_ONE_SIDED * se,
            "ci95": (mean - Z_TWO_SIDED * se, mean + Z_TWO_SIDED * se)}


def _mean_of(clusters: List[Dict], field: str):
    vals = [c[field] for c in clusters if c[field] is not None]
    return sum(vals) / len(vals) if vals else None


# ------------------------------------------------------------ portões (§7)
def gates(cand: List[Dict], ref: List[Dict], budget: float = 2.5,
          recall_margin: float = 0.02) -> Dict:
    """G1–G4 vs referência (A0), no agregado e por regime (G1/G2)."""
    rep: Dict = {"diagnostics": {}}

    def g1(field):
        d = paired_diff(cand, ref, field)
        rep["diagnostics"][f"d_{field}"] = d
        return d["u90"] is not None and d["u90"] < 0.0

    def g2(field):
        d = paired_diff(cand, ref, field)
        rep["diagnostics"][f"d_{field}"] = d
        return d["mean"] is not None and d["mean"] >= -recall_margin

    rep["G1_tta"] = g1("tta_ef")
    rep["G1_tta_abrupt"] = g1("tta_ef_abrupt")
    rep["G1_tta_progressive"] = g1("tta_ef_progressive")
    rep["G2_recall"] = g2("recall")
    rep["G2_recall_abrupt"] = g2("recall_abrupt")
    rep["G2_recall_progressive"] = g2("recall_progressive")

    alerts = _mean_of(cand, "alerts_per_trip")
    rep["alerts_per_trip"] = alerts
    rep["G3_budget"] = alerts is not None and alerts <= budget

    d_far = paired_diff(cand, ref, "far_per_trip")
    rep["diagnostics"]["d_far"] = d_far
    rep["G4_far"] = d_far["u90"] is not None and d_far["u90"] <= 0.0

    rep["admissible"] = all(rep[k] for k in (
        "G1_tta", "G1_tta_abrupt", "G1_tta_progressive",
        "G2_recall", "G2_recall_abrupt", "G2_recall_progressive",
        "G3_budget", "G4_far"))
    return rep


def gates_h2(cand: List[Dict], ref: List[Dict], budget: float = 2.5,
             recall_margin: float = 0.02, tta_margin_s: float = 1.0) -> Dict:
    """Implementação FIEL da H2 pré-declarada (§1) para candidato vs. EGA:
    NÃO-INFERIORIDADE em TTA_ef (U90 < +tta_margin_s) e recall, orçamento e
    FAR não pior, E SUPERIORIDADE em ≥1 de {SAR, redundantes, FAR}
    (IC95 excluindo zero). Emenda v3 — protocolo §13."""
    rep: Dict = {"diagnostics": {}}

    def noninf_tta(field):
        d = paired_diff(cand, ref, field)
        rep["diagnostics"][f"d_{field}"] = d
        return d["u90"] is not None and d["u90"] < tta_margin_s

    def noninf_recall(field):
        d = paired_diff(cand, ref, field)
        rep["diagnostics"][f"d_{field}"] = d
        return d["mean"] is not None and d["mean"] >= -recall_margin

    rep["G1n_tta"] = noninf_tta("tta_ef")
    rep["G1n_tta_abrupt"] = noninf_tta("tta_ef_abrupt")
    rep["G1n_tta_progressive"] = noninf_tta("tta_ef_progressive")
    rep["G2_recall"] = noninf_recall("recall")
    rep["G2_recall_abrupt"] = noninf_recall("recall_abrupt")
    rep["G2_recall_progressive"] = noninf_recall("recall_progressive")

    alerts = _mean_of(cand, "alerts_per_trip")
    rep["alerts_per_trip"] = alerts
    rep["G3_budget"] = alerts is not None and alerts <= budget

    d_far = paired_diff(cand, ref, "far_per_trip")
    rep["diagnostics"]["d_far"] = d_far
    rep["G4_far"] = d_far["u90"] is not None and d_far["u90"] <= 0.0

    # G5 — superioridade em pelo menos uma dimensão de qualidade de alerta
    sup = {}
    for field in ("sar", "redundant_per_trip", "far_per_trip"):
        d = paired_diff(cand, ref, field)
        rep["diagnostics"][f"sup_{field}"] = d
        sup[field] = (d["mean"] is not None and d["mean"] < 0.0
                      and d["ci95"][1] < 0.0)
    rep["G5_superiority"] = any(sup.values())
    rep["G5_dimensions"] = [k for k, v in sup.items() if v]

    rep["admissible"] = all(rep[k] for k in (
        "G1n_tta", "G1n_tta_abrupt", "G1n_tta_progressive",
        "G2_recall", "G2_recall_abrupt", "G2_recall_progressive",
        "G3_budget", "G4_far", "G5_superiority"))
    return rep


def gate_g4b_sar(a2: List[Dict], a1: List[Dict]) -> Dict:
    """G4b (H2): SAR(A2) < SAR(A1) com IC pareado excluindo zero."""
    d = paired_diff(a2, a1, "sar")
    ok = (d["mean"] is not None and d["mean"] < 0.0
          and d["ci95"][1] < 0.0)
    return {"pass": ok, "diff": d}


def summary_row(arm: str, clusters: List[Dict]) -> Dict:
    fields = ("recall", "recall_abrupt", "recall_progressive",
              "tta_ef", "tta_ef_abrupt", "tta_ef_progressive",
              "far_per_trip", "alerts_per_trip", "sar",
              "redundant_per_trip", "force_per_trip", "skip_pct")
    row = {"arm": arm}
    for f in fields:
        m = _mean_of(clusters, f)
        row[f] = round(m, 4) if m is not None else None
    return row
