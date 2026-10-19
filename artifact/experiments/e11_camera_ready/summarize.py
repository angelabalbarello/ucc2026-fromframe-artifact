"""Summaries of E11 raw outputs -> outputs/summary_e11.json (+ console)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e10_eaml import metrics

OUT = Path(__file__).resolve().parent / "outputs"
F = ("recall", "recall_abrupt", "recall_progressive", "tta_ef", "delay_detected",
     "tta_flat10", "far_per_trip", "alerts_per_trip", "sar", "sar12_0.50",
     "sar12_0.60", "sar12_0.70", "sar_int", "redundant_per_trip", "force_per_trip",
     "exec_per_trip", "skip_pct")


def mean(cl, f):
    v = [c[f] for c in cl if c.get(f) is not None]
    return sum(v) / len(v) if v else None


def pooled_sar(cl, thr="0.60"):
    rho = [x for c in cl for x in c.get("_rho12", [])]
    return (sum(x > float(thr) for x in rho) / len(rho), len(rho)) if rho else (None, 0)


def collect(raw):
    arms = [k for k in raw[0] if k not in ("seed", "hash")]
    return {a: [c for r in raw for c in r[a]] for a in arms}


def table(cl_by):
    return {a: {f: (round(mean(cl, f), 4) if mean(cl, f) is not None else None) for f in F}
            for a, cl in cl_by.items()}


def diffs(cl_by, a, b, fields=("recall", "tta_ef", "far_per_trip", "alerts_per_trip",
                                  "sar", "redundant_per_trip", "exec_per_trip")):
    out = {}
    for f in fields:
        d = metrics.paired_diff(cl_by[a], cl_by[b], f)
        out[f] = {"mean": d["mean"], "ci95": d["ci95"], "u90": d["u90"]}
    return out


def fmt(d):
    return {k: f"{v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for k, v in d.items() if v["mean"] is not None}


def main():
    res = {}
    for which in ("holdout", "aosc", "sens"):
        p = OUT / f"raw_{which}.json"
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        cl = collect(raw)
        if which != "sens":
            res[which] = {"means": table(cl),
                          "pooled_sar12": {a: pooled_sar(c) for a, c in cl.items()},
                          "pooled_sar12_050": {a: pooled_sar(c, "0.50") for a, c in cl.items()},
                          "pooled_sar12_070": {a: pooled_sar(c, "0.70") for a, c in cl.items()}}
            pairs = [("A2", "A1"), ("A3", "A2"), ("A4", "A2"), ("A5", "A2"),
                     ("A1F", "A1"), ("A2", "A1F"), ("A3", "A1"), ("A4", "A1"),
                     ("A1_maxskip2", "A1"), ("A2", "A1_maxskip2")]
            res[which]["paired"] = {f"{a}-{b}": diffs(cl, a, b) for a, b in pairs
                                    if a in cl and b in cl}
            res[which]["gates_vs_A1"] = {}
            if which == "holdout":
                for a in ("A2", "A5", "A3", "A4", "A1F", "A1_maxskip2"):
                    g = metrics.gates_h2(cl[a], cl["A1"], 2.5, 0.02, 1.0)
                    res[which]["gates_vs_A1"][a] = {k: v for k, v in g.items() if k != "diagnostics"}
        else:
            variants = sorted({k.split("|")[0] for k in cl})
            sens = {}
            for v in variants:
                a2 = cl[f"{v}|A2"]
                a1 = cl.get(f"{v}|A1", cl["base|A1"])
                g = metrics.gates_h2(a2, a1, 2.5, 0.02, 1.0)
                sens[v] = {"A2": {f: mean(a2, f) for f in ("recall", "tta_ef", "far_per_trip", "alerts_per_trip", "sar", "force_per_trip", "exec_per_trip")},
                           "A1": {f: mean(a1, f) for f in ("recall", "tta_ef", "far_per_trip", "alerts_per_trip", "sar", "exec_per_trip")},
                           "d": diffs({"x": a2, "y": a1}, "x", "y", ("recall", "tta_ef", "far_per_trip", "sar")),
                           "admissible": g["admissible"], "G3": g["G3_budget"],
                           "failed": [k for k in ("G1n_tta", "G1n_tta_abrupt", "G1n_tta_progressive", "G2_recall", "G2_recall_abrupt", "G2_recall_progressive", "G3_budget", "G4_far", "G5_superiority") if not g[k]]}
            res["sens"] = sens
    (OUT / "summary_e11.json").write_text(json.dumps(res, indent=1, default=str))
    for w in ("holdout", "aosc"):
        if w in res:
            print(f"==== {w}")
            for a, row in res[w]["means"].items():
                print(f"{a:13s}", " ".join(f"{k}={v}" for k, v in row.items() if v is not None))
            print("pooled sar12 .5/.6/.7", {a: (res[w]['pooled_sar12_050'][a][0], res[w]['pooled_sar12'][a], res[w]['pooled_sar12_070'][a][0]) for a in res[w]['pooled_sar12']})
            for k, d in res[w]["paired"].items():
                print(k, fmt(d))
            for a, g in res[w]["gates_vs_A1"].items():
                print("gates", a, g["admissible"], [k for k, v in g.items() if v is False])
    if "sens" in res:
        print("==== sens")
        for v, s in res["sens"].items():
            print(f"{v:18s} adm={s['admissible']} fail={s['failed']} A2:" + " ".join(f"{k}={x:.3f}" for k, x in s["A2"].items()) +
                  " | A1: " + " ".join(f"{k}={x:.3f}" for k, x in s["A1"].items()))
            print("   d:", fmt(s["d"]))


if __name__ == "__main__":
    main()
