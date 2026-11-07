"""Graduate analysis G3 and G4 on the frozen test predictions. Nothing here refits or changes the chosen model, except
the cold-start test, which fits a separate copy without the held-out stores.
  - Error by store type, assortment, promotion, holiday, weekday, month, thin history; worst stores; largest single errors.
  - Cold start: hold out 10 percent of stores completely and forecast them with no history, from attributes and calendar only.
  - Retransformation bias: a log-scale model underestimates the mean; a smearing factor from development residuals corrects it.
Usage: python src/failure_audit.py"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from utils import load_config, ROOT
from features import feature_columns
from modeling import make_model
from metrics import summarize, naive_scale

HISTORY_PREFIXES = ("lag_", "roll_mean_", "slope_", "promo_mean", "nonpromo_mean", "store_mean_past", "roll_ratio")
COUNT_PREFIX = "roll_n_"

def _stats(g, p):
    e = (g.sales - g[p]).abs()
    return pd.Series({"n": len(g), "mae": e.mean(), "wape": e.sum() / g.sales.sum(), "mean_sales": g.sales.mean(),
                      "bias": (g[p] - g.sales).mean(), "naive_mae": (g.sales - g.baseline_seasonal_naive).abs().mean()})

def audit_tables(test_preds, features_df, chosen, top=20):
    ctx = features_df[["store_key", "date", "store_type", "assortment", "promo", "holiday_code", "dow", "roll_n_56"]]
    d = test_preds.merge(ctx, on=["store_key", "date"], how="left")
    d = d[d.baseline_seasonal_naive.notna()].copy()
    d["month"] = d.date.str[:7]; d["thin_history"] = np.where(d.roll_n_56 < 20, "yes (fewer than 20 open days)", "no")
    out = {}
    for col in ["store_type", "assortment", "promo", "holiday_code", "dow", "month", "thin_history"]:
        out[col] = d.groupby(col).apply(_stats, p=chosen, include_groups=False).reset_index()
    stores = d.groupby("store_key").apply(_stats, p=chosen, include_groups=False)
    out["worst_stores"] = stores.sort_values("mae", ascending=False).head(top).reset_index()
    d["error"] = d[chosen] - d.sales; d["abs_error"] = d.error.abs()
    out["largest_errors"] = d.sort_values("abs_error", ascending=False).head(top)[
        ["store_key", "date", "sales", chosen, "error", "baseline_seasonal_naive", "promo", "holiday_code", "store_type"]]
    ape = (d.abs_error / d.sales)
    out["ape_quantiles"] = pd.DataFrame({"quantile": [0.5, 0.75, 0.9, 0.95, 0.99], "absolute_pct_error": ape.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).values})
    out["store_level_wape"] = pd.DataFrame({"quantile": [0.1, 0.5, 0.9, 1.0],
                                            "wape": (stores.mae / stores.mean_sales).quantile([0.1, 0.5, 0.9, 1.0]).values})
    return out

def cold_start(df, cfg, chosen, params, test_preds, seed=42, fraction=None):
    """Forecast held-out stores with no history. The model never saw these stores in training, and their history features are blanked."""
    fraction = cfg["eval"]["cold_start_store_fraction"] if fraction is None else fraction
    feats = feature_columns(df); dev, test = df[df.split == "dev"], df[df.split == "test"]
    stores = np.sort(test.store_key.unique())
    held = np.random.default_rng(seed).choice(stores, max(1, int(round(fraction * len(stores)))), replace=False)
    est, cols = make_model(chosen, feats, seed, params)
    train = dev[~dev.store_key.isin(held)]
    est.fit(train[cols], train["sales"].to_numpy(float))
    t = test[test.store_key.isin(held)].copy()
    for c in feats:
        if c.startswith(HISTORY_PREFIXES): t[c] = np.nan
        elif c.startswith(COUNT_PREFIX): t[c] = 0.0
    t["cold_start_model"] = est.predict(t[cols])
    base = train.groupby(["store_type", "dow"])["sales"].median().rename("type_weekday_median").reset_index()
    t = t.merge(base, on=["store_type", "dow"], how="left")
    t = t.merge(test_preds[["store_key", "date", chosen]].rename(columns={chosen: "frozen_model_with_history"}), on=["store_key", "date"])
    rows = {}
    for name in ["frozen_model_with_history", "cold_start_model", "type_weekday_median"]:
        e = (t.sales - t[name]).abs(); rows[name] = {"mae": e.mean(), "wape": e.sum() / t.sales.sum()}
    res = pd.DataFrame(rows).T; res.insert(0, "rows", len(t))
    return {"table": res, "held_out_stores": int(len(held)), "rows": int(len(t))}

def smearing(oof, test_preds, chosen):
    """A model fitted on log sales predicts something close to the median. The mean of the sales is higher: for log residuals e,
    multiply (prediction + 1) by the average of exp(e). The factor comes from development residuals only."""
    e = np.log1p(oof.sales.to_numpy(float)) - np.log1p(oof[chosen].to_numpy(float))
    factor = float(np.mean(np.exp(e)))
    t = test_preds[test_preds.baseline_seasonal_naive.notna()]
    rows = {}
    for name, p in (("as_fitted", t[chosen].to_numpy(float)), ("with_smearing", (t[chosen].to_numpy(float) + 1) * factor - 1)):
        y = t.sales.to_numpy(float)
        rows[name] = {"mean_error_euros": float(np.mean(p - y)), "mean_error_pct_of_sales": float(100 * np.mean(p - y) / np.mean(y)),
                      "mae": float(np.mean(np.abs(p - y))), "wape": float(np.sum(np.abs(p - y)) / np.sum(y))}
    return {"smearing_factor": factor, "table": pd.DataFrame(rows).T}

def main():
    cfg = load_config(); P = ROOT / "data" / "processed"
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(P / ("features_h%d.pkl" % cfg["horizon_days"])))
    ap.add_argument("--test-preds", default=str(P / "test_preds.pkl")); ap.add_argument("--oof", default=str(P / "cv_oof.pkl"))
    ap.add_argument("--params", default=str(ROOT / "artifacts" / "best_params.json"))
    ap.add_argument("--choice", default=str(ROOT / "artifacts" / "model_choice.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    a = ap.parse_args()
    chosen = json.load(open(a.choice))["chosen"]; params = json.load(open(a.params)).get(chosen, {})
    df = pd.read_pickle(a.features); tp = pd.read_pickle(a.test_preds); oof = pd.read_pickle(a.oof)
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("Failure audit of the frozen '%s' model on the test period (euros)\n" % chosen)
    tabs = audit_tables(tp, df, chosen)
    for k, t in tabs.items():
        print("--", k); print(t.round(3).to_string(index=False)); print(); t.to_csv(out / ("audit_%s.csv" % k), index=False)
    print("-- Cold start: %d percent of stores held out and forecast with no history" % int(100 * cfg["eval"]["cold_start_store_fraction"]))
    c = cold_start(df, cfg, chosen, params, tp, seed=cfg["seed"]); print(c["table"].round(3).to_string()); c["table"].to_csv(out / "audit_cold_start.csv")
    print("   (%d stores, %d rows)\n" % (c["held_out_stores"], c["rows"]))
    s = smearing(oof, tp, chosen)
    print("-- Retransformation bias, smearing factor %.4f (from development residuals)" % s["smearing_factor"])
    print(s["table"].round(3).to_string()); s["table"].to_csv(out / "audit_smearing.csv")

if __name__ == "__main__":
    main()
