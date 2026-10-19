"""R7 per-frame decision trace + empirical check of invariant E2
(no alert supported by a single high-risk frame) on holdout seeds."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e11_camera_ready import posthoc as P
from edge_runtime import EAML
import edge_runtime.eaml as E

OUT = P.OUT
# --- E2 check: wrap EAML.step to inspect the window at each delivered alert
counts = {"alerts": 0, "single_high_frame": 0, "reason_peak_only": 0}
_orig = EAML.step
def step(self, ev):
    dec = _orig(self, ev)
    if dec.action == "alert":
        ps = [E._p_of(e) for e in self.buffer.items]
        hi = sum(p >= 0.55 for p in ps)
        counts["alerts"] += 1
        counts["single_high_frame"] += int(hi <= 1)
        counts["reason_peak_only"] += int(dec.risk_reasons == ("peak",))
    return dec
EAML.step = step

seeds = [int(s) for s in sys.argv[1:]] or [3001, 3002, 3003]
trace_done = False
for seed in seeds:
    sd = P.generate(seed)
    if not trace_done:
        # pick the first progressive-critical episode in which A2 issued a
        # forced refresh followed by an alert
        recs = P.run_a2x(sd, P.A2C, P.MH)
        pick = next(r for r in recs if r["categoria"] == "Critico" and r["force_compute"] > 0 and r["n_alerts"] > 0 and r["progressive"])
        key = (pick["persona"], pick["trip"], pick["slot"])
        _, tr = P.run_a2x(sd, P.A2C, P.MH, trace_key=key)
        with open(OUT / "trace_episode.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tr[0])); w.writeheader(); w.writerows(tr)
        print("trace", seed, key, pick["recipe"], "onset", pick["onset"], "alerts", pick["alerts"])
        trace_done = True
    else:
        P.run_a2x(sd, P.A2C, P.MH)
print(counts)
(OUT / "e2_check.json").write_text(json.dumps({"seeds": seeds, **counts}, indent=1))
