"""Look inside the cross-validation errors: are a few rows or stores responsible for most of the damage, and where?
Usage: python src/diagnose_cv.py [--oof data/processed/cv_oof.pkl] [--features data/processed/features_h7.pkl]
Reads results that already exist; trains nothing."""
import argparse
import numpy as np, pandas as pd
from utils import load_config, ROOT

IDS = {"store_key", "date", "sales", "baseline_seasonal_naive", "baseline_wd_mean4", "fold"}

def tail_table(df, preds):
    rows = {}
    for p in preds:
        e = (df.sales - df[p]).to_numpy(float); se = e ** 2
        k = max(1, int(0.01 * len(se)))
        rows[p] = {"mae": np.abs(e).mean(), "rmse": np.sqrt(se.mean()), "rmse_over_mae": np.sqrt(se.mean()) / np.abs(e).mean(),
                   "share_of_sq_error_in_worst_1pct_rows": np.sort(se)[-k:].sum() / se.sum()}
    return pd.DataFrame(rows).T

def worst_stores(df, p, n=10):
    g = df.assign(ae=(df.sales - df[p]).abs()).groupby("store_key").agg(rows=("ae", "size"), mae=("ae", "mean"),
                                                                       mean_sales=("sales", "mean"), mean_pred=(p, "mean"))
    g["pct_of_total_abs_error"] = 100 * g.mae * g.rows / (g.mae * g.rows).sum()
    return g.sort_values("mae", ascending=False).head(n)

def slices(df, preds):
    d = df.copy()
    d["history_missing"] = d.roll_mean_28.isna().map({True: "yes", False: "no"})
    d["competition_distance_band"] = pd.qcut(d.competition_distance, 5, duplicates="drop").astype(str)
    d["month"] = d.date.str[:7]
    out = {}
    for col in ["store_type", "promo", "holiday_code", "history_missing", "competition_distance_band", "month"]:
        out[col] = d.groupby(col).apply(lambda g: pd.Series({"rows": len(g), **{p: np.abs(g.sales - g[p]).mean() for p in preds}}),
                                        include_groups=False)
    return out

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(ROOT / "data" / "processed" / "cv_oof.pkl"))
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / ("features_h%d.pkl" % cfg["horizon_days"])))
    a = ap.parse_args()
    oof = pd.read_pickle(a.oof); feats = pd.read_pickle(a.features)
    ctx = ["store_key", "date", "store_type", "promo", "holiday_code", "competition_distance", "lag_7", "roll_mean_28", "days_to_christmas"]
    df = oof.merge(feats[ctx], on=["store_key", "date"], how="left")
    df = df[df.baseline_seasonal_naive.notna()]
    models = [c for c in oof.columns if c not in IDS and "__seed" not in c]
    preds = ["baseline_seasonal_naive", "baseline_wd_mean4"] + models
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    print("1. How heavy is the error tail? (rmse_over_mae near 1.3 is typical; much higher means a few huge errors)")
    t = tail_table(df, preds); print(t.round(3).to_string())
    suspect = t.loc[models, "rmse_over_mae"].idxmax()
    print("\n2. Ten worst stores for '%s' (the model with the heaviest tail)" % suspect)
    print(worst_stores(df, suspect).round(1).to_string())
    best = min(models, key=lambda m: np.abs(df.sales - df[m]).mean())
    print("\n   Ten worst stores for '%s' (the best model)" % best)
    print(worst_stores(df, best).round(1).to_string())
    print("\n3. Ten largest single errors for '%s'" % suspect)
    big = df.assign(error=df.sales - df[suspect]).reindex(columns=["store_key", "date", "sales", suspect, "error", "lag_7", "roll_mean_28", "competition_distance", "store_type"])
    print(big.loc[big.error.abs().sort_values(ascending=False).head(10).index].round(1).to_string(index=False))
    print("\n4. Mean absolute error by slice (euros)")
    show = ["baseline_seasonal_naive", "baseline_wd_mean4", suspect] + ([best] if best != suspect else [])
    for name, tab in slices(df, show).items():
        print("\n  by", name); print(tab.round(1).to_string())

if __name__ == "__main__":
    main()
