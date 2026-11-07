import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, pytest
import cv, final, features, ablation as A, failure_audit as FA
import etl
from make_fake_data import make
from utils import load_config

CFG = load_config()
PARAMS = {"ridge": {"alpha": 5.0}}

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("an")
    raw = make(d / "raw", n_stores=40); conn = etl.connect(d / "w.db")
    etl.build(conn, raw, replay_days=3, expected_stores=40)
    df = features.build_features(conn, CFG, 7)
    oof = cv.run_cv(df, CFG, ["ridge"], features.feature_columns(df), seed=1, params=PARAMS, verbose=False)
    tp = final.fit_and_score(df, CFG, PARAMS, "ridge", models=("linear", "ridge"), out_dir=d / "a", processed_dir=d / "p",
                             lock_path=d / "a" / "lock.json", B=50)["preds"]
    return conn, df, oof, tp

def test_every_feature_is_in_exactly_one_ablation_group(world):
    feats = features.feature_columns(world[1]); g = A.group_columns(feats)
    allc = [c for v in g.values() for c in v]
    assert sorted(allc) == sorted(feats) and len(allc) == len(set(allc))
    assert set(g) == {"calendar", "promo_planned", "store_competition", "history", "store_average"}

def test_ablation_table_is_paired_and_complete(world):
    t = A.ablation(world[1], CFG, "ridge", PARAMS["ridge"], B=100, store_fraction=1.0)
    assert len(t) == 6 and t.group_removed.iloc[0].startswith("none")
    assert (t.increase_lo <= t.mae_increase + 1e-9).all() and (t.mae_increase <= t.increase_hi + 1e-9).all()
    assert t.loc[t.group_removed == "history", "mae_increase"].iloc[0] != 0

def test_horizon_sweep_drops_lags_that_are_too_fresh(world):
    t = A.horizon_sweep(world[0], CFG, "ridge", PARAMS["ridge"], [1, 14])
    assert "1" in t.lags_used.iloc[0].split(",") and "7" in t.lags_used.iloc[0].split(",")
    assert t.lags_used.iloc[1].split(",")[0] == "14" and "7" not in t.lags_used.iloc[1].split(",")
    assert np.isfinite(t.mae).all()

def test_leak_demo_shows_same_day_customers_inflates_the_score(world):
    r = A.leak_demo(world[0], world[1], CFG, "ridge", PARAMS["ridge"], store_fraction=1.0)
    assert r["with_same_day_customers_INVALID"]["mae"] < 0.5 * r["normal"]["mae"] and r["mae_reduction_pct"] > 50

def test_audit_slices_cover_exactly_the_scored_rows(world):
    _, df, _, tp = world; tabs = FA.audit_tables(tp, df, "ridge")
    scored = int(tp.baseline_seasonal_naive.notna().sum())
    for k in ["store_type", "assortment", "promo", "holiday_code", "dow", "month", "thin_history"]:
        assert int(tabs[k].n.sum()) == scored, k
    assert len(tabs["worst_stores"]) == 20 and tabs["worst_stores"].mae.is_monotonic_decreasing
    assert tabs["largest_errors"].error.abs().is_monotonic_decreasing

def test_cold_start_holds_stores_out_and_blanks_their_history(world):
    _, df, _, tp = world
    c = FA.cold_start(df, CFG, "ridge", PARAMS["ridge"], tp, seed=42, fraction=0.25)
    assert c["held_out_stores"] == 10 and c["rows"] > 0
    t = c["table"]; assert np.isfinite(t[["mae", "wape"]].to_numpy()).all()
    assert t.loc["frozen_model_with_history", "mae"] <= t.loc["cold_start_model", "mae"] * 1.05   # history should help

def test_smearing_reduces_the_bias_of_a_log_model(world):
    _, _, oof, tp = world; s = FA.smearing(oof, tp, "ridge"); t = s["table"]
    assert s["smearing_factor"] > 1.0
    assert abs(t.loc["with_smearing", "mean_error_euros"]) < abs(t.loc["as_fitted", "mean_error_euros"])
