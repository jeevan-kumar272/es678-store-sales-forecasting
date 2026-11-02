import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import features, cv, metrics as M
from utils import load_config

CFG = load_config()

def test_five_folds_never_train_on_or_after_the_validation_start(fake_features):
    df = fake_features; seen = 0
    for name, tr, va in cv.fold_indices(df, CFG):
        seen += 1
        f = next(x for x in CFG["split"]["folds"] if x["name"] == name)
        assert len(tr) > 0 and len(va) > 0
        assert df.iloc[tr]["date"].max() < f["val_start"]
        assert df.iloc[va]["date"].min() >= f["val_start"] and df.iloc[va]["date"].max() <= f["val_end"]
        assert not set(tr) & set(va)
        assert df.iloc[tr]["split"].eq("dev").all() and df.iloc[va]["split"].eq("dev").all()
    assert seen == 5

def test_test_period_is_never_used_in_cv(fake_features):
    df = fake_features
    used = set()
    for _, tr, va in cv.fold_indices(df, CFG):
        used |= set(tr) | set(va)
    assert not df.iloc[sorted(used)]["split"].eq("test").any()
    assert df.iloc[sorted(used)]["date"].max() <= CFG["split"]["dev_end"]

def test_models_run_and_are_scored_on_identical_rows(fake_features):
    df = fake_features; feats = features.feature_columns(df)
    oof = cv.run_cv(df, CFG, ["linear", "ridge", "lightgbm"], feats, seed=1,
                    params={"lightgbm": {"n_estimators": 60}}, verbose=False)
    assert oof[["linear", "ridge", "lightgbm"]].notna().all().all()
    assert (oof[["linear", "ridge", "lightgbm"]] > 0).all().all()
    common = oof[oof.baseline_seasonal_naive.notna()]
    scale = M.naive_scale(df[df.split == "dev"])
    naive = M.summarize(common, "baseline_seasonal_naive", scale)["mae"]
    ridge = M.summarize(common, "ridge", scale)["mae"]
    assert ridge < naive            # on the fake data a model with history features must beat last week's value

def test_shuffled_target_check_passes_on_a_clean_pipeline(fake_features):
    r = cv.shuffled_target_check(fake_features, CFG, features.feature_columns(fake_features), seed=1)
    assert r["passed"], r
    assert r["shuffled_model_mae"] > 1.5 * r["real_model_mae"]

class Cheater:
    """A fake model that looks the true answer up through a side channel, like a pipeline bug would."""
    def __init__(self, truth): self.truth = truth
    def fit(self, X, y): return self
    def predict(self, X): return self.truth.loc[X.index].to_numpy(float)

def test_shuffled_target_check_fails_when_the_pipeline_leaks_the_target(fake_features, monkeypatch):
    """Self-check: if label information reaches the model through a side channel, the check must FAIL."""
    df = fake_features
    monkeypatch.setattr(cv, "make_model", lambda name, feats, seed, params=None: (Cheater(df["sales"]), list(feats)))
    r = cv.shuffled_target_check(df, CFG, features.feature_columns(df), seed=1)
    assert not r["passed"]
    assert r["shuffled_model_mae"] < 0.05 * r["constant_mae"]
