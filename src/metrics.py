"""Error metrics (in euros) and the store-cluster bootstrap. Days within one store are correlated,
so confidence intervals resample whole stores, never rows."""
import numpy as np, pandas as pd

def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def wape(y, p): return float(np.sum(np.abs(y - p)) / np.sum(y))
def rmspe(y, p):
    m = y > 0
    return float(np.sqrt(np.mean(((y[m] - p[m]) / y[m]) ** 2)))
def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
def r2_within_store(y, p, store):
    """R squared after removing each store's own mean: shows skill beyond 'big stores sell more'."""
    ybar = pd.Series(y).groupby(np.asarray(store)).transform("mean").to_numpy()
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - ybar) ** 2))

def naive_scale(train):
    """MASE denominator: in-sample MAE of the seasonal naive forecast on training rows."""
    ok = train["baseline_seasonal_naive"].notna()
    return float(np.mean(np.abs(train.loc[ok, "sales"] - train.loc[ok, "baseline_seasonal_naive"])))

def summarize(df, pred_col, scale, naive_col="baseline_seasonal_naive"):
    y = df["sales"].to_numpy(float); p = df[pred_col].to_numpy(float)
    m = mae(y, p)
    return {"n": len(df), "mae": m, "wape": wape(y, p), "mase": m / scale, "rmse": rmse(y, p), "rmspe": rmspe(y, p),
            "r2": r2(y, p), "r2_within_store": r2_within_store(y, p, df["store_key"]),
            "skill_vs_naive": 1 - m / mae(y, df[naive_col].to_numpy(float))}

def store_sums(df, pred_cols):
    """Per-store sufficient statistics, so thousands of bootstrap draws are cheap."""
    d = pd.DataFrame({"store_key": df["store_key"].to_numpy(), "sales": df["sales"].to_numpy(float)})
    for c in pred_cols:
        e = df["sales"].to_numpy(float) - df[c].to_numpy(float)
        d["ae_" + c] = np.abs(e); d["se_" + c] = e ** 2
    sums = d.groupby("store_key").sum()
    sums["n"] = d.groupby("store_key").size()
    return sums

def bootstrap_ci(sums, pred_cols, base, B=1000, seed=42, level=0.95):
    """Point estimates and percentile intervals for MAE, WAPE, RMSE, plus the paired MAE difference to `base`.
    Each draw resamples whole stores with replacement; every model sees the same draws, so differences are paired."""
    rng = np.random.default_rng(seed)
    S = sums.to_numpy(float); k = len(S); cols = {c: i for i, c in enumerate(sums.columns)}
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=B)
    boot, full = W @ S, S.sum(axis=0)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    def parts(t, c):
        n, sy = t[..., cols["n"]], t[..., cols["sales"]]
        return t[..., cols["ae_" + c]] / n, t[..., cols["ae_" + c]] / sy, np.sqrt(t[..., cols["se_" + c]] / n)
    rows = []
    for c in pred_cols:
        pm, pw, pr = parts(full, c); bm, bw, br = parts(boot, c)
        row = {"model": c}
        for name, pt, bt in (("mae", pm, bm), ("wape", pw, bw), ("rmse", pr, br)):
            row[name] = pt; row[name + "_lo"], row[name + "_hi"] = np.percentile(bt, [lo, hi])
        if c != base:
            bb = parts(boot, base)[0]; d = bm - bb
            row["mae_diff_vs_" + base] = pm - parts(full, base)[0]
            row["diff_lo"], row["diff_hi"] = np.percentile(d, [lo, hi])
        rows.append(row)
    return pd.DataFrame(rows)
