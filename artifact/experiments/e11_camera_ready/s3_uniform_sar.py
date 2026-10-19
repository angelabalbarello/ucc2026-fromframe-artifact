"""Re-score E10 S3 (re-optimized EGS vs A2, fresh seeds 6001-6010) with the
uniform 12-frame SAR window. A1R uses the frozen S3 winner; nothing is
re-selected. Reproduces the published non-SAR S3 numbers as a check."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e10_eaml import metrics
from experiments.e11_camera_ready import posthoc as P

E10S3 = json.loads((P.E10 / "strongaccept" / "strongaccept_result.json").read_text())["S3_fair_egs"]
WIN = E10S3["winner"]

def job(seed):
    sd = P.generate(seed)
    return {"A1R": P.clusters(P.run_a1x(sd, WIN, P.MH), seed),
            "A2": P.clusters(P.run_a2x(sd, P.A2C, P.MH), seed)}

if __name__ == "__main__":
    with Pool(2) as pool:
        raw = pool.map(job, E10S3["fresh_seeds"])
    cl = {a: [c for r in raw for c in r[a]] for a in ("A1R", "A2")}
    m = lambda a, f: sum(c[f] for c in cl[a]) / len(cl[a])
    out = {"winner": WIN,
           "means": {a: {f: m(a, f) for f in ("recall", "tta_ef", "far_per_trip", "alerts_per_trip", "sar")} for a in cl},
           "published_means": E10S3["fresh_means"],
           "A2_vs_A1R": {f: metrics.paired_diff(cl["A2"], cl["A1R"], f) for f in ("recall", "far_per_trip", "alerts_per_trip", "sar")}}
    (P.OUT / "s3_uniform_sar.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
