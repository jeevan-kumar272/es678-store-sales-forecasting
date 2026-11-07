import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import select_model as S

def table(**cols): return pd.DataFrame(cols, index=["F1", "F2", "F3", "F4", "F5"])

def test_simpler_model_wins_when_within_one_standard_error():
    t = table(ridge=[700, 720, 690, 1000, 640], lightgbm=[690, 700, 680, 980, 630])
    chosen, d = S.choose_model(t)
    assert d["best_model"] == "lightgbm" and chosen == "ridge"

def test_best_model_wins_when_the_gap_is_larger_than_one_standard_error():
    t = table(ridge=[780, 770, 760, 1170, 600], lightgbm=[630, 640, 540, 950, 500])
    chosen, d = S.choose_model(t)
    assert chosen == "lightgbm" and d["folds_won_by_best"]["ridge"] == 5

def test_threshold_is_best_mean_plus_one_standard_error():
    t = table(ridge=[800] * 5, lightgbm=[600, 620, 580, 640, 560])
    _, d = S.choose_model(t)
    assert np.isclose(d["threshold"], d["best_mean_mae"] + d["standard_error"])
    assert np.isclose(d["standard_error"], t["lightgbm"].std(ddof=1) / np.sqrt(5))

def test_fold_table_averages_over_seeds_and_uses_the_common_rows():
    rng = np.random.default_rng(0); n = 200
    oof = pd.DataFrame({"fold": np.repeat(["F1", "F2"], n // 2), "sales": 1000.0, "baseline_seasonal_naive": 900.0,
                        "lightgbm": 1100.0, "lightgbm__seed2": 1300.0, "ridge": 1500.0})
    oof.loc[0, "baseline_seasonal_naive"] = np.nan                  # row without a naive forecast is left out
    tab = S.fold_mae_table(oof, ["ridge", "lightgbm"])
    assert np.allclose(tab["lightgbm"], 200.0) and np.allclose(tab["ridge"], 500.0)   # (100 + 300) / 2 seeds
