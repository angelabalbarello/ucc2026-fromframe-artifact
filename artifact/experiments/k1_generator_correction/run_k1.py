"""K1 · re-execução pareada do E10 com o gerador corrigido (v7.1).

Executa os MESMOS braços e controles do E10/E11, com as configurações
congeladas em `experiments/e10_eaml/outputs/spec_e10.json` e as MESMAS
sementes, uma vez com o gerador original (v7) e uma vez com o corrigido
(v7.1). Nenhuma busca, re-seleção ou regra do E13. Saídas apenas em
`experiments/k1_generator_correction/outputs/` (E10/E11 intocados).

Modos:
  holdout  20 sementes de holdout; A0, A1, A2 (E10, A2_frozen), A2 com SAR
           uniforme, A3, A4, A5, A1F, A1_maxskip{6,4,2}; v7 e v7.1
  aosc     10 sementes A-OSC (5001-5010); A0, A1, A2, A3, A4, A1F; v7 e v7.1
  sens     sensibilidade one-at-a-time do E11 (7001-7010); só v7.1
           (a referência v7 é o `raw_sens.json` congelado do E11)
  s3       controle S3: EGS re-otimizado CONGELADO (vencedor de dev do E10,
           sem nova otimização) vs A2 nas sementes 6001-6010; v7 e v7.1

Uso (a partir de artifact/):  python -m experiments.k1_generator_correction.run_k1 holdout
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from experiments.e10_eaml import arms, metrics
from experiments.e10_eaml import scenarios as scen
from experiments.e11_camera_ready import posthoc as PH
from experiments.k1_generator_correction import genswitch as GS

OUT = Path(__file__).resolve().parent / "outputs"
E10_OUT = Path(__file__).resolve().parents[1] / "e10_eaml" / "outputs"
SPEC, A1C, A2C, MH = PH.SPEC, PH.A1C, PH.A2C, PH.MH
S3_WINNER = json.loads((E10_OUT / "strongaccept" / "strongaccept_result.json")
                       .read_text(encoding="utf-8"))["S3_fair_egs"]["winner"]
S3_SEEDS = list(range(6001, 6011))


def by_recipe(recs):
    out = {}
    for r in recs:
        o = metrics.episode_outcomes(r)
        d = out.setdefault(r["recipe"], {"n": 0, "alerts": 0, "false_alerts": 0,
                                          "n_crit": 0, "detected": 0,
                                          "delay_sum": 0.0, "tta_ef_sum": 0.0,
                                          "computed": 0, "force": 0})
        d["n"] += 1; d["alerts"] += r["n_alerts"]; d["false_alerts"] += o["false_alerts"]
        d["computed"] += r["computed"]; d["force"] += r["force_compute"]
        if o["critical"]:
            d["n_crit"] += 1; d["tta_ef_sum"] += o["tta_ef"]
            if o["detected"]:
                d["detected"] += 1
                d["delay_sum"] += (metrics._first_alert_from_onset(r) - r["onset"]) * PH.DT_S
    return out


def driver_response(recs, sd):
    """Derived (no arm behaviour change): response probability q and realised
    response at each delivered alert, replaying the arms' own rule
    q = response_probability(persona, alerts_so_far_in_trip), response =
    draw < q; also at the first post-onset alert of each critical episode."""
    draw = {(e.persona, e.trip, e.slot): e.response_draw for e in sd.episodes}
    count, out = {}, {}
    for r in recs:
        key = (r["persona"], r["trip"])
        k = count.get(key, 0)
        d = out.setdefault(r["persona"], {"alerts": 0, "q_sum": 0.0, "resp": 0,
                                          "crit_alerted": 0, "crit_q_sum": 0.0,
                                          "crit_resp": 0, "crit_prior_alerts_sum": 0})
        ta = metrics._first_alert_from_onset(r) if metrics.episode_outcomes(r)["critical"] else None
        dr = draw[(r["persona"], r["trip"], r["slot"])]
        for t in r["alerts"]:
            q = scen.response_probability(r["persona"], k)
            d["alerts"] += 1; d["q_sum"] += q; d["resp"] += int(dr < q)
            if ta is not None and t == ta:
                d["crit_alerted"] += 1; d["crit_q_sum"] += q
                d["crit_resp"] += int(dr < q); d["crit_prior_alerts_sum"] += k
            k += 1
        count[key] = k
    return out


def pack(recs, seed, sd=None):
    cl = PH.clusters(recs, seed)
    for c in cl:
        c.pop("_rho12", None)
    out = {"clusters": cl, "by_recipe": by_recipe(recs)}
    if sd is not None:
        out["driver_response"] = driver_response(recs, sd)
    return out


def _holdout_arms(sd, seed):
    res = {"A0": pack(arms.run_a0(sd, SPEC["a0_config"], MH), seed, sd),
           "A1": pack(PH.run_a1x(sd, A1C, MH), seed, sd),
           "A2_frozen": pack(arms.run_a2(sd, A2C, MH), seed, sd),
           "A2": pack(PH.run_a2x(sd, A2C, MH), seed, sd),
           "A5": pack(PH.run_a2x(sd, A2C, MH, minimal_oss=True), seed, sd),
           "A3": pack(arms.run_a3(sd, A2C, MH), seed, sd),
           "A4": pack(PH.run_a2x(sd, A2C, MH, force=False), seed, sd),
           "A1F": pack(PH.run_a1x(sd, A1C, MH, fresh=True), seed, sd)}
    for ms in (6, 4, 2):
        res[f"A1_maxskip{ms}"] = pack(PH.run_a1x(sd, A1C, dict(MH, max_skip=ms)), seed, sd)
    return res


def _aosc_arms(sd, seed):
    return {"A0": pack(arms.run_a0(sd, SPEC["a0_config"], MH), seed, sd),
            "A1": pack(PH.run_a1x(sd, A1C, MH), seed, sd),
            "A2_frozen": pack(arms.run_a2(sd, A2C, MH), seed, sd),
            "A2": pack(PH.run_a2x(sd, A2C, MH), seed, sd),
            "A3": pack(arms.run_a3(sd, A2C, MH), seed, sd),
            "A4": pack(PH.run_a2x(sd, A2C, MH, force=False), seed, sd),
            "A1F": pack(PH.run_a1x(sd, A1C, MH, fresh=True), seed, sd)}


def _resp_arms(sd, seed):
    """Supplementary instrumentation for the holdout run (same frozen arms,
    same seeds): records driver-response statistics only."""
    keep = {}
    for name, recs in (("A0", arms.run_a0(sd, SPEC["a0_config"], MH)),
                       ("A1", PH.run_a1x(sd, A1C, MH)),
                       ("A2_frozen", arms.run_a2(sd, A2C, MH)),
                       ("A3", arms.run_a3(sd, A2C, MH)),
                       ("A4", PH.run_a2x(sd, A2C, MH, force=False))):
        keep[name] = {"driver_response": driver_response(recs, sd),
                      "clusters": [{k: c[k] for k in ("persona", "seed", "recall", "far_per_trip",
                                                     "alerts_per_trip")}
                                   for c in metrics.collect_clusters(recs, seed, PH.PERS, PH.NTR)]}
    return keep


def _s3_arms(sd, seed):
    return {"A1R": pack(arms.run_a1(sd, S3_WINNER, MH), seed, sd),
            "A2": pack(arms.run_a2(sd, A2C, MH), seed, sd)}


def job(args):
    which, seed = args
    out = {"seed": seed}
    gens = ("v7_1",) if which == "sens" else ("v7", "v7_1")
    for g in gens:
        GS.use(g)
        if which == "sens":
            r = PH.job_sens(seed)
            out[g] = {k: {"clusters": [{kk: vv for kk, vv in c.items() if kk != "_rho12"}
                                       for c in v]}
                      for k, v in r.items() if k != "seed"}
            continue
        sd = PH.generate(seed, osc=(which == "aosc"))
        fn = {"holdout": _holdout_arms, "aosc": _aosc_arms, "s3": _s3_arms,
              "resp": _resp_arms}[which]
        out[g] = {"hash": sd.exogenous_hash, "arms": fn(sd, seed)}
    GS.use("v7")
    return out


def main(which):
    OUT.mkdir(exist_ok=True)
    seeds = {"holdout": SPEC["holdout_seeds"], "aosc": PH.AOSC, "sens": PH.FRESH,
             "s3": S3_SEEDS, "resp": SPEC["holdout_seeds"]}[which]
    t0 = time.time()
    with Pool(2) as pool:
        res = []
        for r in pool.imap_unordered(job, [(which, s) for s in seeds]):
            res.append(r)
            print(f"[k1:{which}] seed {r['seed']} {time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda r: r["seed"])
    meta = {"generator_v7_sha16": GS.sha16(GS.PATHS["v7"]),
            "generator_v7_1_sha16": GS.sha16(GS.PATHS["v7_1"]),
            "spec": "experiments/e10_eaml/outputs/spec_e10.json (frozen)",
            "s3_winner_frozen": S3_WINNER if which == "s3" else None,
            "seeds": seeds, "elapsed_s": time.time() - t0}
    (OUT / f"raw_k1_{which}.json").write_text(json.dumps({"meta": meta, "runs": res}),
                                             encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
