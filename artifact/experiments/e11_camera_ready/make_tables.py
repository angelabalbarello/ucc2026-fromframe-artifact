"""Generate CSV + LaTeX tables for the camera-ready docs from E11 outputs."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e10_eaml import metrics
OUT = Path(__file__).resolve().parent / "outputs"
FPT = 12 * 96  # frames per trip
S = json.loads((OUT / "summary_e11.json").read_text())

# ---- Phase 4: cost by arm
rows = []
for w in ("holdout", "aosc"):
    for a, m in S[w]["means"].items():
        if a == "A2_frozen":
            continue
        rows.append(dict(regime=w, arm=a, frames_per_trip=FPT,
                         inferences_per_trip=round(m["exec_per_trip"], 2),
                         suppressed_pct=round(100 * m["skip_pct"], 2),
                         C_rel=round(m["exec_per_trip"] / FPT, 4),
                         forced_refresh_requests_per_trip=round(m["force_per_trip"], 3),
                         recall=m["recall"], tta_ef=m["tta_ef"], far=m["far_per_trip"],
                         alerts=m["alerts_per_trip"], sar_uniform12=m["sar"],
                         redundant=m["redundant_per_trip"]))
with open(OUT / "cost_by_arm.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

lab = {"A0": "A0 Legacy", "A1": "A1 EGS", "A2": "A2 EAML", "A5": "A5 EAML, age+reuse only",
       "A3": "A3 EAML$-$OSSL", "A4": "A4 no refresh", "A1F": "A1+F (EGS+age/reuse+refresh)",
       "A1_maxskip6": "A1, max\\_skip=6", "A1_maxskip4": "A1, max\\_skip=4", "A1_maxskip2": "A1, max\\_skip=2"}
tex = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
       r"Arm & Inf./trip & $C_{\mathrm{rel}}$ & Force/trip & Recall & FAR/tr. & Alerts/tr. & SAR$_{12}$\\", r"\midrule"]
for r in rows:
    if r["regime"] != "holdout":
        continue
    tex.append(f"{lab[r['arm']]} & {r['inferences_per_trip']:.1f} & {r['C_rel']:.3f} & "
               f"{r['forced_refresh_requests_per_trip']:.2f} & {r['recall']:.3f} & {r['far']:.3f} & "
               f"{r['alerts']:.3f} & {r['sar_uniform12']:.4f}\\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_cost.tex").write_text("\n".join(tex))

# ---- Phase 7: sensitivity
order = ["base", "W=6", "W=10", "W=12", "m=2", "m=4", "k=2", "k=4", "theta_ep=0.65", "theta_ep=0.75",
         "theta_on=0.6", "theta_on=0.7", "rho_max=0.5", "rho_max=0.7", "age_max=2", "age_max=6",
         "tau_delta=0.002486", "tau_delta=0.009946", "tau_H=0.5", "tau_H=0.7", "max_skip=4", "max_skip=12",
         "noise=0.05", "noise=0.2", "noise_AR1=0.8", "noise_AR1=0.95"]
srows = []
for v in order:
    s = S["sens"][v]; d = s["d"]
    srows.append(dict(variant=v, A1_recall=s["A1"]["recall"], A2_recall=s["A2"]["recall"],
                      d_recall=d["recall"]["mean"], d_recall_lo=d["recall"]["ci95"][0], d_recall_hi=d["recall"]["ci95"][1],
                      d_tta=d["tta_ef"]["mean"], d_tta_u90=d["tta_ef"]["u90"],
                      d_far=d["far_per_trip"]["mean"], d_far_lo=d["far_per_trip"]["ci95"][0], d_far_hi=d["far_per_trip"]["ci95"][1],
                      A1_sar=s["A1"]["sar"], A2_sar=s["A2"]["sar"], d_sar=d["sar"]["mean"],
                      A2_alerts=s["A2"]["alerts_per_trip"], A2_force=s["A2"]["force_per_trip"],
                      A1_inf=s["A1"]["exec_per_trip"], A2_inf=s["A2"]["exec_per_trip"],
                      admissible=s["admissible"], failed=";".join(s["failed"])))
with open(OUT / "sensitivity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(srows[0])); w.writeheader(); w.writerows(srows)
pretty = lambda v: (v.replace("theta_ep", r"$\theta_{\mathrm{ep}}$").replace("theta_on", r"$\theta_{\mathrm{on}}$")
                    .replace("rho_max", r"$\rho_{\max}$").replace("age_max", r"$a_{\max}$")
                    .replace("tau_delta=0.002486", r"$\tau_\Delta\times0.5$").replace("tau_delta=0.009946", r"$\tau_\Delta\times2$")
                    .replace("tau_H", r"$\tau_H$").replace("max_skip", r"max\_skip")
                    .replace("noise_AR1=", "AR(1) $\\phi$=").replace("noise=", "noise "))
tex = [r"\begin{tabular}{@{}lrrrrrc@{}}", r"\toprule",
       r"Variant & $\Delta$Recall & U90$\Delta$TTA & $\Delta$FAR & SAR A1/A2 & Alerts & Adm.\\", r"\midrule"]
for r in srows:
    mark = "\\checkmark" if r["admissible"] else "--"
    tex.append(f"{pretty(r['variant'])} & {r['d_recall']:+.3f} & {r['d_tta_u90']:+.2f} & {r['d_far']:+.2f} & "
               f"{r['A1_sar']:.3f}/{r['A2_sar']:.3f} & {r['A2_alerts']:.2f} & {mark}" + "\\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_sensitivity.tex").write_text("\n".join(tex))
print(open(OUT / "table_cost.tex").read()); print(open(OUT / "table_sensitivity.tex").read())
