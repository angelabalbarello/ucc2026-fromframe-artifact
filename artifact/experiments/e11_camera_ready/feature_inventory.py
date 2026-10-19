"""Inventory of the 31 generated attributes as they reach the E10 arms.
In E10 the attributes influence the arms ONLY through dx_t (MHEG scheduling);
the risk seen by the alerting layers is y_frame_clean * risk_mod + noise.
Empirical statistics on one development seed (201)."""
import csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.e10_eaml import scenarios as S

NAMES = (["beh_normal", "beh_phone", "beh_radio", "beh_drinking", "beh_reach_object",
          "beh_grooming", "beh_talking"] + ["seatbelt_on", "seatbelt_off"] +
         ["conf_behavior", "conf_seatbelt", "speed_norm", "kin_accel", "kin_brake",
          "kin_gyro", "fatigue_flag", "phone_x_speed", "ctx_highway", "ctx_urban",
          "ctx_rain", "ctx_night", "traffic_free", "traffic_moderate", "traffic_congested",
          "audio_siren", "audio_horn", "audio_child_cry", "audio_phone_ring", "audio_loud_music",
          "eye_gaze_off_road", "lane_offset_norm"])
GROUP = (["behaviour one-hot (state machine)"] * 7 + ["seatbelt one-hot"] * 2 +
         ["sensor confidence (Gaussian, precursor drop)"] * 2 + ["speed / 150 km/h (noisy)"] +
         ["kinematic channel driven by risk curve + road noise (3 regimes)"] * 3 +
         ["state flag"] + ["phone x speed interaction"] + ["context flag"] * 4 +
         ["traffic one-hot"] * 3 + ["audio flag, active only at SENSOR_LEVEL>=4"] * 5 +
         ["eye-gaze series (v7)", "lane-offset series (v7)"])
assert len(NAMES) == len(GROUP) == S._gen.FEATURE_SIZE == 31


def main(seed=201):
    import random as _r
    root = np.random.SeedSequence(seed)
    ss_recipe, ss_gen, _, _ = root.spawn(4)
    rng_recipe = np.random.default_rng(ss_recipe)
    gen_seeds = np.random.default_rng(ss_gen).integers(0, 2**31 - 1, size=10_000)
    Xs, idx = [], 0
    for persona in S.ALL_PERSONAS[:4]:
        for trip in range(5):
            order = list(S.TRIP_SLOTS); rng_recipe.shuffle(order)
            for cat in order:
                rec = S.SLOT_POOLS[cat][int(rng_recipe.integers(0, len(S.SLOT_POOLS[cat])))]
                c = int(gen_seeds[idx]); idx += 1
                np.random.seed(c); _r.seed(c)
                X, *_ = S._gen.generate_episode(rec, S.RECIPES_ALL, window=S.WINDOW_T,
                                                SENSOR_LEVEL=S.SENSOR_LEVEL)
                Xs.append(X)
    X = np.concatenate(Xs).astype(np.float64)
    contrib = np.concatenate([np.abs(np.diff(x, axis=0)) / (1 + np.abs(x[:-1])) for x in Xs]).mean(0) / 31
    rows = []
    for j in range(31):
        col = X[:, j]
        rows.append(dict(index=j, name=NAMES[j], group=GROUP[j], min=round(col.min(), 4),
                         max=round(col.max(), 4), mean=round(col.mean(), 4),
                         share_nonzero=round(float((np.abs(col) > 1e-9).mean()), 4),
                         constant=bool(col.max() - col.min() < 1e-12),
                         mean_dx_contribution=round(float(contrib[j]), 6),
                         dx_share=round(float(contrib[j] / contrib.sum()), 4)))
    out = Path(__file__).resolve().parent / "outputs" / "feature_inventory.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
