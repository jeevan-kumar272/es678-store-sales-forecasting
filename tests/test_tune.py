import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import features, tune
from utils import load_config

CFG = load_config()

def feats_of(df): return features.feature_columns(df)

def test_best_trial_is_the_minimum_and_inside_the_search_space(fake_features):
    r = tune.tune_model(fake_features, CFG, "ridge", feats_of(fake_features), seed=1, n_trials=4)
    assert len(r["trial_values"]) == 4 and np.isclose(r["best_value"], min(r["trial_values"]))
    assert 1e-2 <= r["best_params"]["alpha"] <= 1e3

def test_tuning_is_reproducible_for_the_same_seed(fake_features):
    a = tune.tune_model(fake_features, CFG, "ridge", feats_of(fake_features), seed=2, n_trials=4)
    b = tune.tune_model(fake_features, CFG, "ridge", feats_of(fake_features), seed=2, n_trials=4)
    assert a["best_params"] == b["best_params"] and a["trial_values"] == b["trial_values"]

def test_test_period_rows_cannot_influence_tuning(fake_features):
    tampered = fake_features.copy()
    tampered.loc[tampered.split == "test", "sales"] *= 1000        # destroy the test targets
    tampered.loc[tampered.split == "test", "baseline_seasonal_naive"] *= 1000
    a = tune.tune_model(fake_features, CFG, "ridge", feats_of(fake_features), seed=1, n_trials=3)
    b = tune.tune_model(tampered, CFG, "ridge", feats_of(fake_features), seed=1, n_trials=3)
    assert a["trial_values"] == b["trial_values"]

def test_store_sample_keeps_whole_stores_and_all_dates(fake_features):
    sub = tune.sample_stores(fake_features, 0.5, seed=42)
    assert 0.4 * fake_features.store_key.nunique() <= sub.store_key.nunique() <= 0.6 * fake_features.store_key.nunique()
    full_counts = fake_features.groupby("store_key").size()
    assert (sub.groupby("store_key").size() == full_counts.loc[sub.store_key.unique()]).all()   # no store loses any date
    pd.testing.assert_frame_equal(sub, tune.sample_stores(fake_features, 0.5, seed=42))         # same sample every time

def test_lightgbm_search_runs_and_stays_in_range(fake_features):
    r = tune.tune_model(fake_features, CFG, "lightgbm", feats_of(fake_features), seed=1, n_trials=2, store_fraction=0.5)
    p = r["best_params"]
    assert 200 <= p["n_estimators"] <= 800 and 15 <= p["num_leaves"] <= 127 and 0.02 <= p["learning_rate"] <= 0.15

def test_summary_picks_the_best_seed_and_reports_spread():
    res = {"ridge": {"1": {"best_value": 800.0, "best_params": {"alpha": 5}}, "2": {"best_value": 790.0, "best_params": {"alpha": 7}}}}
    best, summ = tune.summarize(res)
    assert best["ridge"] == {"alpha": 7} and summ["ridge"]["spread"] == 10.0


def test_lasso_search_range_avoids_the_very_slow_tiny_penalties(fake_features):
    r = tune.tune_model(fake_features, CFG, "lasso", feats_of(fake_features), seed=1, n_trials=3, store_fraction=0.5)
    assert 1e-4 <= r["best_params"]["alpha"] <= 1e-1
