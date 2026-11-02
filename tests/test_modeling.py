import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd, pytest
from modeling import make_model

def synth(n_stores=400, days=60, seed=0):
    """Stores of very different size; sales and the sales features scale together, as in the real data."""
    rng = np.random.default_rng(seed)
    scale = np.exp(rng.normal(8.6, 0.55, n_stores)); store = np.repeat(np.arange(n_stores), days)
    s = scale[store]; n = len(store); nz = lambda sd: rng.lognormal(0, sd, n); promo = rng.integers(0, 2, n)
    return pd.DataFrame({"store": store, "scale": s, "promo": promo, "dow": rng.integers(1, 7, n),
                         "sales": s * (1 + .25 * promo) * nz(.12), "lag_7": s * nz(.15), "roll_mean_28": s * nz(.04),
                         "store_mean_past": s * nz(.02), "competition_distance": rng.lognormal(7.5, 1.3, n),
                         "slope_28": rng.normal(0, 30, n)})

FEATS = ["promo", "dow", "lag_7", "roll_mean_28", "store_mean_past", "competition_distance", "slope_28"]

def fit_predict(name, log_features):
    df = synth(); rng = np.random.default_rng(1)
    hold = df.store.isin(rng.choice(400, 80, replace=False)); tr, te = df[~hold], df[hold]
    est, cols = make_model(name, FEATS, 1, {"log_features": log_features})
    est.fit(tr[cols], tr.sales.to_numpy(float))
    return te, est.predict(te[cols])

@pytest.mark.parametrize("name", ["ridge", "lasso"])
def test_log_scaling_fixes_overshoot_on_the_biggest_stores(name):
    te_raw, p_raw = fit_predict(name, False); te_log, p_log = fit_predict(name, True)
    top = (te_log.scale >= te_log.scale.quantile(0.95)).to_numpy()
    mae = lambda te, p, m: np.abs(te.sales.to_numpy()[m] - p[m]).mean()
    assert mae(te_log, p_log, top) < 0.8 * mae(te_raw, p_raw, top)
    assert (p_log / te_log.sales.to_numpy()).max() < 0.85 * (p_raw / te_raw.sales.to_numpy()).max()

def test_small_linear_baseline_also_improves():
    te_raw, p_raw = fit_predict("linear", False); te_log, p_log = fit_predict("linear", True)
    assert np.abs(te_log.sales.to_numpy() - p_log).mean() < 0.9 * np.abs(te_raw.sales.to_numpy() - p_raw).mean()

def test_predictions_are_positive_euros_and_missing_history_is_handled():
    df = synth(); df.loc[df.index[::7], ["lag_7", "roll_mean_28", "slope_28"]] = np.nan
    est, cols = make_model("ridge", FEATS, 1)
    est.fit(df[cols], df.sales.to_numpy(float))
    p = est.predict(df[cols])
    assert np.isfinite(p).all() and (p > 0).all()
