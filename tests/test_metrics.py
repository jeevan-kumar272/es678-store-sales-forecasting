import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import metrics as M

def test_hand_computed_values():
    y = np.array([100.0, 200.0]); p = np.array([110.0, 180.0])
    assert np.isclose(M.mae(y, p), 15.0)
    assert np.isclose(M.wape(y, p), 30 / 300)
    assert np.isclose(M.rmse(y, p), np.sqrt(250))
    assert np.isclose(M.rmspe(y, p), 0.1)
    assert np.isclose(M.r2(y, p), 0.9)
    df = pd.DataFrame({"sales": y, "pred": p, "baseline_seasonal_naive": [90.0, 220.0], "store_key": [1, 1]})
    s = M.summarize(df, "pred", scale=10.0)
    assert np.isclose(s["mase"], 1.5) and np.isclose(s["skill_vs_naive"], 1 - 15 / 15)

def test_within_store_r2_ignores_store_level():
    y = np.array([100.0, 110.0, 1000.0, 1010.0]); store = [1, 1, 2, 2]
    perfect_levels_only = np.array([105.0, 105.0, 1005.0, 1005.0])
    assert M.r2(y, perfect_levels_only) > 0.99            # looks great overall
    assert M.r2_within_store(y, perfect_levels_only, store) < 0.01 + 0.0   # but has no skill inside a store

def make_frame(n_stores=60, per=40, seed=0):
    rng = np.random.default_rng(seed)
    store = np.repeat(np.arange(n_stores), per); y = rng.gamma(9, 700, n_stores * per)
    return pd.DataFrame({"store_key": store, "sales": y, "good": y * rng.normal(1, 0.05, len(y)),
                         "bad": y * rng.normal(1, 0.25, len(y))})

def test_bootstrap_interval_contains_point_estimate_and_is_reproducible():
    df = make_frame(); sums = M.store_sums(df, ["good", "bad"])
    a = M.bootstrap_ci(sums, ["good", "bad"], "bad", B=400, seed=3)
    b = M.bootstrap_ci(sums, ["good", "bad"], "bad", B=400, seed=3)
    pd.testing.assert_frame_equal(a, b)
    for r in a.itertuples():
        assert r.mae_lo <= r.mae <= r.mae_hi
    assert np.isclose(a.loc[a.model == "good", "mae"].iloc[0], M.mae(df.sales.values, df.good.values))

def test_paired_difference_detects_a_clearly_better_model():
    df = make_frame(); sums = M.store_sums(df, ["good", "bad"])
    r = M.bootstrap_ci(sums, ["good", "bad"], "bad", B=400, seed=3).set_index("model").loc["good"]
    assert r["mae_diff_vs_bad"] < 0 and r["diff_hi"] < 0       # whole interval below zero

def test_resampling_whole_stores_gives_wider_intervals_than_resampling_rows():
    rng = np.random.default_rng(5); stores = np.repeat(np.arange(40), 50)
    store_effect = np.repeat(rng.normal(0, 400, 40), 50)        # errors shared within a store
    y = np.full(len(stores), 5000.0); p = y + store_effect + rng.normal(0, 100, len(y))
    df = pd.DataFrame({"store_key": stores, "sales": y, "m": p})
    ci = M.bootstrap_ci(M.store_sums(df, ["m"]), ["m"], "m", B=600, seed=1).iloc[0]
    store_width = ci.mae_hi - ci.mae_lo
    e = np.abs(y - p); row_means = [rng.choice(e, len(e)).mean() for _ in range(600)]
    row_width = np.percentile(row_means, 97.5) - np.percentile(row_means, 2.5)
    assert store_width > 2 * row_width                          # row resampling would be far too optimistic
