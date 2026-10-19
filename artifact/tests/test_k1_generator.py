"""K1 (generator v7.1) changes only recipe L02 and its dependent channels.

Isolation proof required before any K1 run:
  1. the original generator is untouched (hash) and v7.1 differs from it only
     in the header comment and in the peak_then_drop floor line;
  2. every recipe other than L02 yields bit-identical X, y_clean and metadata;
  3. L02 keeps its label, onset, segments and non-risk-coupled channels, and
     its latent peak now follows its declared target (0.35 + 0.05);
  4. a full E10 seed (normal and A-OSC mix) is bit-identical outside L02:
     p_obs, dx, onset, label, recipe and response draw;
  5. the training recipes (N/C), which built the detector checkpoints of
     other studies, are identical, so no checkpoint depends on K1;
  6. switching back restores the original generator exactly.
"""
import random

import numpy as np
import pytest

from experiments.e10_eaml import scenarios as SC
from experiments.k1_generator_correction import genswitch as GS

RISK_COUPLED = [9, 10, 12, 13, 14, 29, 30]
OTHER = [j for j in range(31) if j not in RISK_COUPLED]
# descriptive metadata computed from the latent curve (never a label)
RISK_DERIVED_META = {
    "detectable_start_035", "detectable_start_045", "detectable_start_055",
    "onset_gap_035", "onset_gap_045", "onset_gap_055",
    "max_y_frame_clean", "max_y_frame_obs", "mean_y_frame_clean",
    "mean_y_frame_obs", "last6_mean_clean", "last6_mean_obs",
    "risk_episode", "risk_episode_severity", "risk_episode_operational",
    "mean_eye_gaze", "mean_lane_offset"}


@pytest.fixture(autouse=True)
def _restore():
    GS.use("v7")
    yield
    GS.use("v7")


def _gen(mod, recipe, seed):
    np.random.seed(seed); random.seed(seed)
    recipes = {**mod.RECIPES_TRAIN, **mod.RECIPES_BORDERLINE}
    X, yo, y, _, meta = mod.generate_episode(recipe, recipes, window=96, SENSOR_LEVEL=2)
    return X, yo, y, meta


def test_original_untouched_and_textual_diff_is_minimal():
    assert GS.sha16(GS.PATHS["v7"]) == "efede62fe3aceed8"
    a = GS.PATHS["v7"].read_bytes().splitlines()
    b = GS.PATHS["v7_1"].read_bytes().splitlines()
    code_b = [l for l in b if not l.strip().startswith(b"#")]
    code_a = [l for l in a if not l.strip().startswith(b"#")]
    removed = [l for l in code_a if l not in code_b]
    added = [l for l in code_b if l not in code_a]
    assert [l.strip() for l in removed] == [b"peak = max(0.75, target + 0.05)"]
    assert [l.strip() for l in added] == [b"peak = target + 0.05"]


def test_recipe_tables_identical():
    m0, m1 = GS._load("v7"), GS._load("v7_1")
    assert m0.RECIPES_TRAIN == m1.RECIPES_TRAIN
    assert m0.RECIPES_BORDERLINE == m1.RECIPES_BORDERLINE
    users = sorted(k for k, v in {**m1.RECIPES_TRAIN, **m1.RECIPES_BORDERLINE}.items()
                   if v.get("curve") == "peak_then_drop" and not v.get("progressive"))
    assert users == ["C01", "C04", "L02"]


def test_all_recipes_except_l02_bit_identical():
    m0, m1 = GS._load("v7"), GS._load("v7_1")
    recipes = list({**m0.RECIPES_TRAIN, **m0.RECIPES_BORDERLINE})
    for r in recipes:
        if r == "L02":
            continue
        for s in range(15):
            a, b = _gen(m0, r, s), _gen(m1, r, s)
            assert np.array_equal(a[0], b[0]), r
            assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]), r
            assert a[3] == b[3], r


def test_l02_changes_only_risk_and_coupled_channels():
    m0, m1 = GS._load("v7"), GS._load("v7_1")
    peaks0, peaks1 = [], []
    for s in range(30):
        a, b = _gen(m0, "L02", s), _gen(m1, "L02", s)
        assert np.array_equal(a[0][:, OTHER], b[0][:, OTHER])
        ma, mb = a[3], b[3]
        for k in ("categoria", "class", "class_name", "recipe_id", "event_start",
                  "event_end", "event_duration", "regime", "persistent_tail", "window"):
            assert ma[k] == mb[k], k
        changed = {k for k in ma if ma[k] != mb[k]}
        assert changed <= RISK_DERIVED_META, changed - RISK_DERIVED_META
        peaks0.append(a[2].max()); peaks1.append(b[2].max())
    assert min(peaks0) >= 0.70                       # v7: floor 0.75 (smoothed)
    assert max(peaks1) <= 0.40 + 1e-6                # v7.1: target + 0.05
    assert min(peaks1) >= 0.30


@pytest.mark.parametrize("osc", [False, True])
def test_full_seed_identical_outside_l02(osc):
    GS.use("v7"); a = SC.generate_seed(11001, None, 3, osc_mix=osc)
    GS.use("v7_1"); b = SC.generate_seed(11001, None, 3, osc_mix=osc)
    n_l02 = 0
    for x, y in zip(a.episodes, b.episodes):
        assert (x.persona, x.trip, x.slot, x.recipe, x.categoria, x.progressive,
                x.onset, x.response_draw) == \
               (y.persona, y.trip, y.slot, y.recipe, y.categoria, y.progressive,
                y.onset, y.response_draw)
        if x.recipe == "L02":
            n_l02 += 1
            assert not np.array_equal(x.p_obs, y.p_obs)
        else:
            assert np.array_equal(x.p_obs, y.p_obs) and np.array_equal(x.dx, y.dx)
    assert n_l02 > 0
    assert a.exogenous_hash != b.exogenous_hash


def test_switch_back_restores_original():
    h0 = SC.generate_seed(11002, ["P01"], 2).exogenous_hash
    GS.use("v7_1"); GS.use("v7")
    assert GS.current() == "v7"
    assert SC.generate_seed(11002, ["P01"], 2).exogenous_hash == h0
