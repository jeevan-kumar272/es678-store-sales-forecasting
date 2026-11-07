import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, pytest
import cv, final, features, intervals as I
from utils import load_config

CFG = load_config()
PARAMS = {"ridge": {"alpha": 5.0}, "lightgbm": {"n_estimators": 60}}

@pytest.fixture(scope="module")
def stage(fake_features, tmp_path_factory):
    d = tmp_path_factory.mktemp("iv")
    feats = features.feature_columns(fake_features)
    oof = cv.run_cv(fake_features, CFG, ["ridge"], feats, seed=1, params=PARAMS, verbose=False)
    test = final.fit_and_score(fake_features, CFG, PARAMS, "ridge", models=("linear", "ridge"), out_dir=d / "a",
                               processed_dir=d / "p", lock_path=d / "a" / "lock.json", B=50)["preds"]
    res = I.run_intervals(fake_features, oof, test, "ridge", PARAMS, CFG, out_dir=d / "out")
    return res, oof, test, d

def test_offsets_give_nominal_coverage_on_their_own_residuals():
    rng = np.random.default_rng(0); res = rng.normal(0, 0.1, 5000); g = rng.integers(0, 3, 5000)
    off = I.quantile_offsets(res, g, [0.1, 0.9])
    lo, hi = I.apply_offsets(np.full(5000, 1000.0), g, off, 0), I.apply_offsets(np.full(5000, 1000.0), g, off, 1)
    y = 1000.0 * np.exp(res)       # y lies at pred * exp(residual)
    assert abs(np.mean((y >= lo) & (y <= hi)) - 0.8) < 0.02 and (lo < hi).all()

def test_small_groups_fall_back_to_the_pooled_offsets():
    res = np.r_[np.random.default_rng(1).normal(0, .1, 1000), [5.0] * 10]; g = np.r_[np.zeros(1000), np.ones(10)]
    off = I.quantile_offsets(res, g, [0.1, 0.9], min_n=200)
    assert 0 in off and 1 not in off

def test_test_period_coverage_is_close_to_nominal_on_clean_fake_data(stage):
    res = stage[0]; s = res["summary"]
    assert 0.72 <= s["conformal"]["coverage"] <= 0.88
    assert 0.55 <= s["quantile_regression"]["coverage"] <= 0.95
    assert set(res["coverage"].slice) == {"store_type", "promo", "holiday_code", "month"}
    assert "gate4" in s and 0 <= s["gate4"]["coverage_on_F1_to_F3"] <= 1

def test_quantile_stocking_beats_point_forecast_when_shortage_is_costly(stage):
    nv = stage[0]["newsvendor"].set_index("cost_ratio_short_to_surplus")
    assert nv.loc[10, "cost_quantile_forecast"] < nv.loc[10, "cost_point_forecast"]
    assert nv.loc[5, "saving_pct"] > 0
    assert nv.loc[10, "share_of_days_short"] < nv.loc[1, "share_of_days_short"]      # higher quantile means fewer shortages

def test_calibration_uses_only_development_folds_and_files_are_written(stage):
    res, oof, test, d = stage
    assert set(oof.fold) == {"F1", "F2", "F3", "F4", "F5"}
    for f in ["intervals_summary.json", "intervals_coverage_by_group.csv", "newsvendor.csv", "conformal_offsets.json"]:
        assert (d / "out" / f).exists(), f
