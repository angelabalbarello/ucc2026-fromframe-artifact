"""What happens after a served forced refresh in A2 (holdout)?"""
import json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e11_camera_ready import posthoc as P

def job(seed):
    sd = P.generate(seed)
    allkeys = {(p, tr, ep.slot) for p, tr, eps in sd.trips() for ep in eps}
    from experiments.e11_camera_ready.sar_trace import run_variant
    _, T = run_variant(sd, "A2", keep=allkeys)
    C = Counter(); example = None
    for k, rs in T.items():
        for i, r in enumerate(rs):
            if r["action"] != "force_compute_before_alert":
                continue
            nxt = rs[i + 1] if i + 1 < len(rs) else None
            C["requests"] += 1
            if nxt is None:
                C["lost_end"] += 1; continue
            C[f"t+1_{nxt['action']}|{nxt['blocks']}"] += 1
            nxt_alert = next((j - i for j in range(i + 1, len(rs)) if rs[j]["action"] == "alert"), None)
            if nxt_alert is not None:
                C["alert_later_same_episode"] += 1
                C[f"delay_bin_{min(nxt_alert, 20)//5*5}"] += 1
                if example is None and nxt_alert <= 10:
                    example = list(k)
            else:
                C["no_alert_after_in_episode"] += 1
    return dict(C), example

if __name__ == "__main__":
    with Pool(2) as pool:
        res = pool.map(job, P.SPEC["holdout_seeds"])
    tot = Counter(); ex = None
    for c, e in res:
        tot.update(c); ex = ex or (e and [res.index((c, e))] + e)
    out = {"counters": dict(tot), "example_seed_index_and_key": ex}
    (P.OUT / "refresh_followup.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
