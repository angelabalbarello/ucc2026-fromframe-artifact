# Results matrix (camera-ready, v1.0.0)

Every number below is reproduced by the code in this repository. Classes:

* **H — historical (E10, frozen):** reproduced exactly; keep as published.
* **C-acc — accounting correction:** same frozen runs, re-scored with the definition used for the other arms (uniform 12-frame SAR window). The original value is kept in `experiments/e10_eaml/outputs` and reported as "E10 accounting".
* **C-def — definition correction:** the number is unchanged; the paper's text or equation was wrong.
* **X — post-acceptance exploratory:** new analysis, labelled as such and never used to re-select anything.

## 1. Holdout (20 seeds, 240 persona×seed clusters)

| Arm | Recall | TTAef (s) [H, Alt A] | TTAef, miss = 10 s [X] | Delay if detected [X] | FAR/trip | Alerts/trip | SAR₁₂ | Red./trip | Inferences/trip [X] |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 1.000 | 1.49 | 1.49 | 1.49 | 1.250 | 3.486 | 0.034 | — | 368.0 |
| A1 | 0.825 | 4.14 | 2.98 | 1.47 | 0.874 | 2.898 | 0.030 | 1.33 | 368.0 |
| **A2** | 0.846 | 4.21 | 3.18 | 1.94 | 0.335 | 2.481 | **0.002** [C-acc; E10: 0.000] | 1.16 | 368.3 |
| A3 | 0.830 | 4.42 | 3.30 | 1.93 | 0.321 | 2.539 | 0.007 | 1.22 | 368.0 |
| A4 [H, post-hoc] | 0.846 | 4.20 | 3.18 | 1.94 | 0.336 | 2.478 | 0.003 | 1.16 | 368.0 |
| A5 = A2 with age + reuse only [X] | identical to A2 in every cluster | | | | | | | | |
| A1+F: EGS + reuse/age + refresh [X] | 0.837 | 3.96 | 2.87 | 1.47 | 0.883 | 2.893 | 0.021 | 1.36 | 368.2 |
| A1, max_skip 2 [X] | 0.824 | 4.15 | — | — | 0.877 | 2.899 | 0.0001 | 1.34 | 553.1 |
| A2T: A2 + wall-clock guard [X] | 0.982 | 1.95 | — | — | 0.528 | 3.310 | 0.0035 | 1.60 | 368.3 |
| A1C: A1 + candidate-only guard [X] | 0.750 | 5.65 | — | — | 0.450 | 2.240 | 0.020 | 0.36 | 368.0 |

## 2. Gates and paired contrasts vs. A1 (95% CI; U90 where it is the gate statistic)

| Item | Published (E10) | Camera-ready value | Class | Verdict change |
|---|---|---|---|---|
| G1 ΔTTAef A2−A1 | +0.066, U90 0.254 | unchanged; Alt B +0.207 [0.083, 0.330], U90 0.310; delay if detected +0.467 [0.447, 0.486] | H + X | pass under both |
| G2 ΔRecall | +0.021 [0.006, 0.037]; abrupt +0.018, progressive +0.024 | unchanged | H | — |
| G3 alerts | 2.481 | unchanged | H | — |
| G4 ΔFAR | −0.539, U90 −0.514 | unchanged | H | — |
| G5 ΔSAR A2−A1 | −0.030 [−0.033, −0.027] | **−0.028 [−0.031, −0.025]** | C-acc | still pass (3/3) |
| G5 Δredundant / ΔFAR | −0.18; −0.54 | unchanged | H | — |
| A2 admissible | yes | yes | — | — |
| A3 vs. A1 | inadmissible (G3) | unchanged | H | — |
| A4 vs. A1 | not evaluated | admissible | X | new |
| A5 / A1+F / A1 max_skip 2 / A2T / A1C | — | yes / no (G3, G4) / no (G3, G4) / no (G3) / no (G1, G2) | X | new |
| "Effective-latency parity" | text | "within the 1.0 s non-inferiority margin; +0.47 s when detected" | C-def | wording |

## 3. Mechanism attribution (holdout)

| Contrast | Published | Camera-ready | Class |
|---|---|---|---|
| ΔSAR A3−A2 | +0.0066 [0.005, 0.008] | **+0.0046 [0.0032, 0.0059]** | C-acc |
| ΔTTA A3−A2 | +0.22 [0.17, 0.27] | unchanged | H |
| ΔSAR A4−A2 (force channel) | +0.0033 [0.0023, 0.0044] ("0 → 0.0033") | **+0.0013 [0.0006, 0.0020]** | C-acc |
| A4 unserved refresh requests | 0.71/trip | unchanged | H |
| Refresh outcome at t+1 | not reported | never an alert; 89% still held, 11% RISK gone; 26% followed by an alert later in the episode | X |
| A5 − A2 | — | 0 on every metric (identical) | X |
| A2 − A1+F | — | FAR −0.549 [−0.578, −0.519]; SAR −0.019; recall +0.009 [−0.006, 0.024] | X |
| A2T − A1 (wall-clock time base on both) | — | recall +0.157, TTA −2.19 s, FAR −0.346, alerts +0.412 (over budget), SAR −0.026 | X |
| A2 − A1C (candidate-only time base on both) | — | recall +0.096, TTA −1.44 s, FAR −0.116, alerts +0.240, SAR −0.018 | X |
| Episodes starting with an open EGS event | — | A1 1.8% vs. A2 66.6% | X |

## 4. SAR accounting-threshold check (S2)

| | Published | Camera-ready (uniform 12-frame window) | Class |
|---|---|---|---|
| A2 share of alerts with ρ > 0.50 / 0.60 / 0.70 | 0 / 0 / 0 (11,907 alerts) | **2.2% / 0.21% / 0** | C-acc |
| A1 share of alerts with ρ > 0.50 / 0.60 / 0.70 | 11.7% / 3.0% / — | 11.7% / 3.0% / 0.3% | H |
| Interpretation | "not an artifact of the threshold" | zero on the EAML's own window holds by construction; on the uniform window A2 stays about 14× below A1 at 0.60 | C-acc |

## 5. Re-optimized EGS on fresh seeds 6001–6010 (S3)

| | Published | Camera-ready | Class |
|---|---|---|---|
| A1R alerts / FAR / SAR | 3.17 / 0.66 / 0.018 | unchanged | H |
| A2 alerts / FAR / SAR | 2.49 / 0.36 / **0** | 2.49 / 0.36 / **0.0029** | C-acc |
| A1R recall / TTA | **not reported** | **1.000 / 1.61 s** (A2: 0.841 / 4.29 s; A2−A1R recall −0.159 [−0.171, −0.147]) | H, reported in the camera-ready |
| Verdict | A1R inadmissible (budget) | unchanged; A1R reaches full recall at +27% alert load | — |

## 6. A-OSC stress (10 seeds)

| Item | Published | Camera-ready | Class |
|---|---|---|---|
| A0 alerts / FAR | 13.1 / 9.9 | unchanged | H |
| A1 recall / TTAef | 0.296 / 11.7 s | unchanged; decomposition: 70% misses × (5 s censored + 10 s) + 1.22 s when detected; Alt B 7.40 s | H + C-def |
| A2 − A1 recall / TTAef / FAR | +0.252 / −3.5 s / −0.51 | unchanged; Alt B −1.99 s | H |
| A2 SAR | 0 | **0.0012** | C-acc |
| ΔSAR A3−A2 | [0.097, 0.120] | +0.107 [0.096, 0.119] | C-acc (negligible) |
| A4 pattern "same as holdout" | stated | ΔSAR A4−A2 +0.0005 [−0.0003, 0.0013], **not detected** | C-acc |
| Refresh requests/trip | 5.9 | unchanged; inferences +2.1/trip (+0.4%) | H + X |
| Budget | no arm meets it | unchanged | H |

## 7. Sensitivity (fresh seeds 7001–7010) [X]

FAR and SAR gains hold in all 25 one-at-a-time variants. Admissibility fails for W ≥ 10, θon = 0.60, ρmax = 0.70, τΔ×0.5, τΔ×2, noise 0.05 / 0.20, and AR(1) φ = 0.8 / 0.95. Under low or AR(1)-correlated noise the recall advantage reverses (−0.02 to −0.03) and TTA is +0.7 to +0.85 s (`artifact/experiments/e11_camera_ready/outputs/sensitivity.csv`).

## 8. Items not re-run

The runtime microbenchmark (Table VII) was not re-run: it is hardware-dependent, so the published values are kept and marked as such.
