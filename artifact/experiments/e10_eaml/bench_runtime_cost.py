"""Microbenchmark of the evidence-layer overhead (Paper C, UCC 2026).

Measures the per-frame cost of the three edge levels in isolation from the
detector: MHEG gate decision, OSSL suppression-summary materialization
(reuse path), and EAML window update + alert decision, plus the full
per-frame pipeline with a stub detector. Also reports the serialized size
of one OSS_t and of the EAML window.

The detector is replaced by a trivial stub so that the measured cost is the
architecture's own overhead, the quantity needed to judge whether the
evidence contract is affordable relative to a real detector (typically
milliseconds per frame). Platform is reported honestly (single x86 thread,
desktop-class CPU); embedded measurement remains future work.
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
from pathlib import Path
import time
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge_runtime.policy_state import EdgePolicy
from edge_runtime.mheg import MHEGScheduler, MHEGDecision
from edge_runtime.ossl import OSSLayer
from edge_runtime.eaml import EAML
from edge_runtime.runtime import EdgeRuntime

N_WARM = 20_000
N = 200_000
SEED = 42


def _pcts(samples_ns):
    s = sorted(samples_ns)
    return {
        "p50_us": s[len(s) // 2] / 1000.0,
        "p95_us": s[int(len(s) * 0.95)] / 1000.0,
        "mean_us": statistics.fmean(s) / 1000.0,
    }


def synth_stream(n, seed=SEED):
    import random
    rng = random.Random(seed)
    out = []
    p = 0.3
    for _ in range(n):
        dx = abs(rng.gauss(0.0, 0.25))
        p = min(0.99, max(0.01, p + rng.gauss(0.0, 0.05)))
        out.append((dx, p))
    return out


def bench(fn, args_iter):
    samples = []
    for args in args_iter:
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append(time.perf_counter_ns() - t0)
    return samples


def main():
    pol = EdgePolicy()
    stream = synth_stream(N_WARM + N)

    # --- MHEG.decide --------------------------------------------------------
    mheg = MHEGScheduler(pol)
    args = [(dx, 0.5) for dx, _ in stream]
    bench(mheg.decide, args[:N_WARM])
    mheg_ns = bench(mheg.decide, args[N_WARM:])

    # --- OSSL.materialize, reuse path (builds one OSS_t) ---------------------
    ossl = OSSLayer(pol)
    seed_dec = MHEGDecision(compute=True, trigger_reason="delta",
                            delta_x=0.6, entropy_prev=0.5, suppression_count=0)
    ossl.materialize(0, seed_dec, p_t=0.4, entropy_t=0.6, confidence_t=1.0)
    reuse_args = []
    for i, (dx, _) in enumerate(stream):
        d = MHEGDecision(compute=False, trigger_reason="reuse",
                         delta_x=min(dx, 0.49), entropy_prev=0.5,
                         suppression_count=i % 8)
        reuse_args.append((i + 1, d))
    bench(ossl.materialize, reuse_args[:N_WARM])
    oss_ns = bench(ossl.materialize, reuse_args[N_WARM:])

    # --- EAML.step (window push + full RISK/GUARD decision) ------------------
    eaml = EAML(pol)
    ossl2 = OSSLayer(pol)
    evs = []
    for i, (dx, p) in enumerate(stream):
        if i % 3 == 0:
            d = MHEGDecision(compute=True, trigger_reason="delta",
                             delta_x=dx, entropy_prev=0.5,
                             suppression_count=0)
            ev = ossl2.materialize(i, d, p_t=p, entropy_t=0.5,
                                   confidence_t=1.0)
        else:
            d = MHEGDecision(compute=False, trigger_reason="reuse",
                             delta_x=min(dx, 0.49), entropy_prev=0.5,
                             suppression_count=i % 3)
            ev = ossl2.materialize(i, d)
        evs.append((ev,))
    bench(eaml.step, evs[:N_WARM])
    eaml_ns = bench(eaml.step, evs[N_WARM:])

    # --- Full per-frame pipeline with stub detector --------------------------
    rt = EdgeRuntime(pol, infer_fn=lambda t: (0.4, 0.6, 1.0))
    args = [(dx,) for dx, _ in stream]
    bench(rt.process_frame, args[:N_WARM])
    full_ns = bench(rt.process_frame, args[N_WARM:])

    # --- Sizes ----------------------------------------------------------------
    d = MHEGDecision(compute=False, trigger_reason="reuse", delta_x=0.2,
                     entropy_prev=0.5, suppression_count=3)
    oss_obj = ossl.materialize(len(stream) + 10, d)
    oss_bytes = len(json.dumps(asdict(oss_obj)).encode())
    window_bytes = oss_bytes * pol.window_w

    result = {
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "n_frames_measured": N,
        "mheg_decide": _pcts(mheg_ns),
        "ossl_materialize_oss": _pcts(oss_ns),
        "eaml_step": _pcts(eaml_ns),
        "full_pipeline_stub_detector": _pcts(full_ns),
        "oss_serialized_bytes": oss_bytes,
        "eaml_window_serialized_bytes_ub": window_bytes,
        "window_w": pol.window_w,
    }
    out = str(Path(__file__).parent / "outputs" / "bench_runtime_cost.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
