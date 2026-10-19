"""K1 · análise pareada original (v7) vs corrigido (v7.1). Somente leitura das
saídas `raw_k1_*.json`; nenhuma execução de braço.

Produz `outputs/summary_k1.json` com:
  * verificação de reprodução: v7 desta execução == E11 raw_*.json == E10 CSV;
  * médias por braço e gerador; diferença pareada v7.1 − v7 por braço;
  * contrastes do E10 (H1', G4b, H3; A2 vs A0) recalculados em cada gerador
    com as MESMAS funções de `experiments.e10_eaml.metrics`;
  * decomposição por receita (alertas falsos; recall e atraso por crítica);
  * A-OSC, S3 (vencedor congelado) e sensibilidade (v7 = E11 congelado).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.e10_eaml import metrics as M

K1 = Path(__file__).resolve().parent / "outputs"
E10 = Path(__file__).resolve().parents[1] / "e10_eaml" / "outputs"
E11 = Path(__file__).resolve().parents[1] / "e11_camera_ready" / "outputs"
SPEC = json.loads((E10 / "spec_e10.json").read_text(encoding="utf-8"))
FIELDS = ["recall", "recall_abrupt", "recall_progressive", "tta_ef", "tta_ef_abrupt",
          "tta_ef_progressive", "delay_detected", "tta_flat10", "far_per_trip",
          "alerts_per_trip", "redundant_per_trip", "sar", "sar12_0.60", "sar_int",
          "force_per_trip", "exec_per_trip", "skip_pct"]
PAIR_FIELDS = ["recall", "tta_ef", "delay_detected", "far_per_trip", "alerts_per_trip",
               "redundant_per_trip", "sar", "sar12_0.60", "exec_per_trip"]


def load(which):
    p = K1 / f"raw_k1_{which}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def gather(raw, gen):
    arms = {}
    for r in raw["runs"]:
        for a, v in r[gen]["arms"].items():
            arms.setdefault(a, {"clusters": [], "by_recipe": {}})
            arms[a]["clusters"].extend(v["clusters"])
            for rec, d in v["by_recipe"].items():
                acc = arms[a]["by_recipe"].setdefault(rec, {k: 0 for k in d})
                for k, x in d.items():
                    acc[k] += x
    return arms


def mean(cl, f):
    v = [c[f] for c in cl if c.get(f) is not None]
    return sum(v) / len(v) if v else None


def means(cl):
    return {f: mean(cl, f) for f in FIELDS}


def pdiff(a, b, f):
    d = M.paired_diff(a, b, f)
    return {k: d[k] for k in ("n", "mean", "ci95", "u90")}


def recipe_table(arms, n_trips_total):
    out = {}
    for a, v in arms.items():
        rows = {}
        for rec, d in sorted(v["by_recipe"].items()):
            row = {"n": d["n"], "alerts": d["alerts"], "false_alerts": d["false_alerts"],
                   "false_alerts_per_trip": d["false_alerts"] / n_trips_total}
            if d["n_crit"]:
                row["recall"] = d["detected"] / d["n_crit"]
                row["delay_detected"] = d["delay_sum"] / d["detected"] if d["detected"] else None
                row["tta_ef"] = d["tta_ef_sum"] / d["n_crit"]
            rows[rec] = row
        out[a] = rows
    return out


def e10_contrasts(arms, a2="A2_frozen"):
    b, rm = SPEC["budget_alerts_per_trip"], SPEC["recall_margin"]
    tm = SPEC.get("tta_noninferiority_margin_s", 1.0)
    A0, A1, A2, A3 = (arms[k]["clusters"] for k in ("A0", "A1", a2, "A3"))
    h1 = M.gates_h2(A2, A1, b, rm, tta_margin_s=tm)
    g4b = M.gate_g4b_sar(A2, A1)
    h3s = M.paired_diff(A3, A2, "sar")
    h3g = M.gates_h2(A3, A1, b, rm, tta_margin_s=tm)
    hyp1 = h1["admissible"]
    hyp2 = hyp1 and g4b["pass"]
    hyp3 = hyp2 and ((not h3g["admissible"]) or (h3s["mean"] is not None and h3s["mean"] > 0
                                                  and h3s["ci95"][0] > 0))
    gate_keys = [k for k in h1 if k.startswith("G")]
    return {"H1p_gates": {k: h1[k] for k in gate_keys},
            "H1p_diag": {k: {kk: v[kk] for kk in ("mean", "ci95", "u90")}
                         for k, v in h1["diagnostics"].items()},
            "G4b": {"pass": g4b["pass"], "mean": g4b["diff"]["mean"], "ci95": g4b["diff"]["ci95"]},
            "H3_sar_A3_minus_A2": {k: h3s[k] for k in ("mean", "ci95")},
            "H3_A3_vs_A1_admissible": h3g["admissible"],
            "A2_vs_A0": {f: pdiff(A2, A0, f) for f in PAIR_FIELDS},
            "A2_vs_A1": {f: pdiff(A2, A1, f) for f in PAIR_FIELDS},
            "A3_vs_A2": {f: pdiff(A3, A2, f) for f in PAIR_FIELDS},
            "A1_vs_A0": {f: pdiff(A1, A0, f) for f in PAIR_FIELDS},
            "hypotheses": {"H1_prime": bool(hyp1), "H2_prime": bool(hyp2), "H3": bool(hyp3)}}


def reproduction(raw, e11_name, e10_csv=None, arm_map=None):
    """v7 clusters from this run vs frozen E11 raw (and E10 CSV)."""
    rep = {"e11_mismatches": 0, "e11_compared": 0, "e10_mismatches": 0, "e10_compared": 0}
    e11 = {r["seed"]: r for r in json.loads((E11 / e11_name).read_text(encoding="utf-8"))}
    for r in raw["runs"]:
        ref = e11.get(r["seed"])
        if ref is None:
            continue
        for a, v in r["v7"]["arms"].items():
            if a not in ref:
                continue
            for c, cr in zip(v["clusters"], ref[a]):
                for k, x in c.items():
                    if k in cr and not k.startswith("_"):
                        rep["e11_compared"] += 1
                        y = cr[k]
                        if isinstance(x, str) or isinstance(y, str):
                            rep["e11_mismatches"] += int(x != y); continue
                        if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-12):
                            rep["e11_mismatches"] += 1
        if "hash" in ref:
            rep.setdefault("hash_equal", True)
            rep["hash_equal"] &= (ref["hash"] == r["v7"]["hash"])
    if e10_csv:
        rows = list(csv.DictReader(open(E10 / e10_csv, encoding="utf-8")))
        idx = {(x["arm"], x["persona"], int(x["seed"])): x for x in rows}
        for r in raw["runs"]:
            for mine, theirs in arm_map.items():
                for c in r["v7"]["arms"][mine]["clusters"]:
                    x = idx[(theirs, c["persona"], c["seed"])]
                    for k in ("recall", "tta_ef", "far_per_trip", "alerts_per_trip",
                              "sar", "redundant_per_trip"):
                        rep["e10_compared"] += 1
                        y = x[k]
                        y = None if y in ("", "None") else float(y)
                        if (c[k] is None) != (y is None) or (y is not None and abs(c[k] - y) > 1e-9):
                            rep["e10_mismatches"] += 1
    return rep


def block(raw, n_trips_total, contrasts=True):
    g0, g1 = gather(raw, "v7"), gather(raw, "v7_1")
    out = {"means": {g: {a: means(v["clusters"]) for a, v in arms.items()}
                     for g, arms in (("v7", g0), ("v7_1", g1))},
           "k1_minus_v7": {a: {f: pdiff(g1[a]["clusters"], g0[a]["clusters"], f)
                               for f in PAIR_FIELDS} for a in g0},
           "by_recipe": {"v7": recipe_table(g0, n_trips_total),
                         "v7_1": recipe_table(g1, n_trips_total)}}
    if contrasts:
        out["e10_contrasts"] = {"v7": e10_contrasts(g0), "v7_1": e10_contrasts(g1)}
        out["e10_contrasts_uniformSAR"] = {"v7": e10_contrasts(g0, "A2"),
                                           "v7_1": e10_contrasts(g1, "A2")}
    return out


def main():
    res = {}
    n_tr = len(SPEC["personas"]) * SPEC["n_trips"]
    h = load("holdout")
    if h:
        res["holdout"] = block(h, n_tr * len(h["runs"]))
        res["holdout"]["reproduction"] = reproduction(
            h, "raw_holdout.json", "results_holdout.csv",
            {"A0": "A0", "A1": "A1", "A2_frozen": "A2", "A3": "A3"})
        res["holdout"]["hash_v7_1_differs"] = all(r["v7"]["hash"] != r["v7_1"]["hash"]
                                                  for r in h["runs"])
    a = load("aosc")
    if a:
        res["aosc"] = block(a, n_tr * len(a["runs"]))
        res["aosc"]["reproduction"] = reproduction(a, "raw_aosc.json")
    s = load("s3")
    if s:
        g0, g1 = gather(s, "v7"), gather(s, "v7_1")
        res["s3"] = {"winner": s["meta"]["s3_winner_frozen"],
                     "means": {"v7": {k: means(v["clusters"]) for k, v in g0.items()},
                               "v7_1": {k: means(v["clusters"]) for k, v in g1.items()}},
                     "A2_vs_A1R": {g: {f: pdiff(x["A2"]["clusters"], x["A1R"]["clusters"], f)
                                       for f in PAIR_FIELDS} for g, x in (("v7", g0), ("v7_1", g1))},
                     "by_recipe": {"v7": recipe_table(g0, n_tr * len(s["runs"])),
                                   "v7_1": recipe_table(g1, n_tr * len(s["runs"]))}}
        sa = json.loads((E10 / "strongaccept" / "strongaccept_result.json").read_text())
        res["s3"]["frozen_e10_fresh_means"] = sa["S3_fair_egs"]["fresh_means"]
    sv = load("sens")
    if sv:
        orig = {r["seed"]: r for r in json.loads((E11 / "raw_sens.json").read_text())}
        keys = [k for k in sv["runs"][0]["v7_1"]]
        tab = {}
        for k in keys:
            c1 = [c for r in sv["runs"] for c in r["v7_1"][k]["clusters"]]
            c0 = [c for r in sv["runs"] for c in orig[r["seed"]][k]]
            tab[k] = {"v7": {f: mean(c0, f) for f in ("recall", "tta_ef", "far_per_trip",
                                                      "alerts_per_trip", "sar12_0.60")},
                      "v7_1": {f: mean(c1, f) for f in ("recall", "tta_ef", "far_per_trip",
                                                        "alerts_per_trip", "sar12_0.60")}}
        # A2 − A1 FAR per sensitivity point and generator
        pts = sorted({k.split("|")[0] for k in keys})
        contrasts = {}
        for p in pts:
            if f"{p}|A1" in keys and f"{p}|A2" in keys:
                row = {}
                for g, src in (("v7", "orig"), ("v7_1", "k1")):
                    if src == "orig":
                        a1 = [c for r in sv["runs"] for c in orig[r["seed"]][f"{p}|A1"]]
                        a2 = [c for r in sv["runs"] for c in orig[r["seed"]][f"{p}|A2"]]
                    else:
                        a1 = [c for r in sv["runs"] for c in r["v7_1"][f"{p}|A1"]["clusters"]]
                        a2 = [c for r in sv["runs"] for c in r["v7_1"][f"{p}|A2"]["clusters"]]
                    row[g] = {f: pdiff(a2, a1, f) for f in ("far_per_trip", "recall", "tta_ef",
                                                           "sar12_0.60")}
                contrasts[p] = row
        res["sens"] = {"points": tab, "A2_minus_A1": contrasts}
    (K1 / "summary_k1.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("written", list(res))


if __name__ == "__main__":
    main()
