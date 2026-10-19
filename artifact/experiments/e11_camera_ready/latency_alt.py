"""Phase 1 — latency metric: implemented (Alt A) vs. published semantics
(Alt B: missed episode = exactly 10 s). Uses E11 raw cluster outputs."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e10_eaml import metrics
OUT = Path(__file__).resolve().parent / "outputs"
res = {}
for w in ("holdout", "aosc"):
    raw = json.loads((OUT / f"raw_{w}.json").read_text())
    cl = {a: [c for r in raw for c in r[a]] for a in raw[0] if a not in ("seed", "hash")}
    m = lambda a, f: sum(c[f] for c in cl[a] if c[f] is not None) / sum(1 for c in cl[a] if c[f] is not None)
    res[w] = {"means": {a: {"altA_impl": m(a, "tta_ef"), "altB_flat10": m(a, "tta_flat10"),
                            "delay_detected_only": m(a, "delay_detected"), "recall": m(a, "recall")}
                        for a in ("A0", "A1", "A2", "A3", "A4", "A5", "A1F") if a in cl}}
    res[w]["paired_vs_A1"] = {}
    for a in [x for x in ("A2", "A3", "A4", "A5", "A1F") if x in cl]:
        res[w]["paired_vs_A1"][a] = {f: metrics.paired_diff(cl[a], cl["A1"], f) for f in ("tta_ef", "tta_flat10", "delay_detected")}
(OUT / "latency_alternatives.json").write_text(json.dumps(res, indent=1))
for w, r in res.items():
    print("==", w)
    for a, v in r["means"].items():
        print(a, {k: round(x, 3) for k, x in v.items()})
    for a, d in r["paired_vs_A1"].items():
        print(a, "-A1", {f: (round(x["mean"], 3), [round(y, 3) for y in x["ci95"]], round(x["u90"], 3)) for f, x in d.items()})
