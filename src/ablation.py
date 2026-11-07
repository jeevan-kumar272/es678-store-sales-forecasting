"""Graduate analysis G2: which signals carry the model, and how robust is the result?
All three analyses use the DEVELOPMENT period only (the five rolling folds); the test period is not touched.
  1. Feature-group ablation: drop one group at a time, refit with the frozen settings, report the change in MAE with a paired
     store-cluster bootstrap interval.
  2. Horizon sweep: forecast 1, 7, 14 and 28 days ahead (lags older than the horizon only).
  3. Leakage demonstration: add same-day Customers on purpose and show how much it inflates the score. This run is INVALID by design.
Usage: python src/ablation.py [--groups] [--horizon-sweep] [--leak-demo] [--store-fraction 0.5]"""
import argparse, json, pathlib, sqlite3
import numpy as np, pandas as pd
from utils import load_config, ROOT
import features as F
from cv import run_cv
from metrics import store_sums, bootstrap_ci, summarize, naive_scale
from tune import sample_stores

GROUPS = {
    "calendar": ["dow", "dom", "week", "month", "is_month_end3", "days_to_christmas", "days_from_easter",
                 "days_since_holiday", "holiday_code", "school"],
    "promo_planned": ["promo", "promo_prev7", "promo2_active"],
    "store_competition": ["store_type", "assortment", "competition_distance", "comp_dist_missing", "comp_open_months",
                          "comp_open_missing", "promo2_flag"],
    "history": None,                                    # lags, rolling statistics, trend, promo and non-promo averages
    "store_average": ["store_mean_past"],
}
HISTORY_PREFIXES = ("lag_", "roll_", "slope_", "promo_mean", "nonpromo_mean")

def group_columns(feats):
    """Every feature belongs to exactly one group."""
    out = {g: [c for c in cols if c in feats] for g, cols in GROUPS.items() if cols is not None}
    taken = {c for v in out.values() for c in v}
    out["history"] = [c for c in feats if c.startswith(HISTORY_PREFIXES) and c not in taken]
    return out

def ablation(df, cfg, model, params, seed=1, B=1000, store_fraction=0.5, verbose=False):
    sub = sample_stores(df, store_fraction, cfg["seed"]); feats = F.feature_columns(sub); groups = group_columns(feats)
    runs = {"full": feats}
    for g, cols in groups.items():
        runs["without_" + g] = [c for c in feats if c not in cols]
    base = None; preds = {}
    for name, fs in runs.items():
        oof = run_cv(sub, cfg, [model], fs, seed, params={model: params}, verbose=False)
        if base is None: base = oof[["store_key", "date", "sales", "baseline_seasonal_naive"]].copy()
        else: assert (oof[["store_key", "date"]].values == base[["store_key", "date"]].values).all()
        preds[name] = oof[model].to_numpy()
        if verbose: print("  finished", name, flush=True)
    d = base.copy()
    for k, v in preds.items(): d[k] = v
    d = d[d.baseline_seasonal_naive.notna()]
    ci = bootstrap_ci(store_sums(d, list(preds)), list(preds), "full", B=B, seed=cfg["seed"]).set_index("model")
    rows = [{"group_removed": "none (full model)", "features_removed": 0, "mae": ci.loc["full", "mae"], "mae_lo": ci.loc["full", "mae_lo"],
             "mae_hi": ci.loc["full", "mae_hi"], "mae_increase": 0.0, "increase_lo": 0.0, "increase_hi": 0.0, "increase_pct": 0.0}]
    for g in groups:
        r = ci.loc["without_" + g]
        rows.append({"group_removed": g, "features_removed": len(groups[g]), "mae": r.mae, "mae_lo": r.mae_lo, "mae_hi": r.mae_hi,
                     "mae_increase": r["mae_diff_vs_full"], "increase_lo": r.diff_lo, "increase_hi": r.diff_hi,
                     "increase_pct": 100 * r["mae_diff_vs_full"] / ci.loc["full", "mae"]})
    return pd.DataFrame(rows)

def horizon_sweep(conn, cfg, model, params, horizons, seed=1, verbose=False):
    rows = []
    for h in horizons:
        d = F.build_features(conn, cfg, h); feats = F.feature_columns(d)
        oof = run_cv(d, cfg, [model], feats, seed, params={model: params}, verbose=False)
        c = oof[oof.baseline_seasonal_naive.notna()]
        s = summarize(c, model, naive_scale(d[d.split == "dev"]))
        rows.append({"horizon_days": h, "rows_scored": s["n"], "mae": s["mae"], "wape": s["wape"], "mase": s["mase"],
                     "naive_mae": float(np.abs(c.sales - c.baseline_seasonal_naive).mean()), "skill_vs_naive": s["skill_vs_naive"],
                     "lags_used": ",".join(x[4:] for x in feats if x.startswith("lag_"))})
        if verbose: print("  horizon", h, "done", flush=True)
    return pd.DataFrame(rows)

def leak_demo(conn, df, cfg, model, params, seed=1, store_fraction=0.5):
    """Adds Customers on the target day. Not knowable at forecast time, so the result is not a valid score."""
    sub = sample_stores(df, store_fraction, cfg["seed"]); feats = F.feature_columns(sub)
    cu = pd.read_sql("SELECT store_key, date_key, customers AS customers_same_day FROM fact_daily_sales", conn)
    sub = sub.assign(date_key=sub.date.str.replace("-", "").astype(int)).merge(cu, on=["store_key", "date_key"], how="left").drop(columns="date_key")
    res = {}
    for name, fs in (("normal", feats), ("with_same_day_customers_INVALID", feats + ["customers_same_day"])):
        oof = run_cv(sub, cfg, [model], fs, seed, params={model: params}, verbose=False)
        c = oof[oof.baseline_seasonal_naive.notna()]
        s = summarize(c, model, naive_scale(sub[sub.split == "dev"]))
        res[name] = {"mae": s["mae"], "wape": s["wape"], "r2": s["r2"], "rmspe": s["rmspe"]}
    res["mae_reduction_pct"] = 100 * (1 - res["with_same_day_customers_INVALID"]["mae"] / res["normal"]["mae"])
    return res

def main():
    cfg = load_config(); P = ROOT / "data" / "processed"
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(P / ("features_h%d.pkl" % cfg["horizon_days"])))
    ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"]))
    ap.add_argument("--params", default=str(ROOT / "artifacts" / "best_params.json"))
    ap.add_argument("--choice", default=str(ROOT / "artifacts" / "model_choice.json"))
    ap.add_argument("--groups", action="store_true"); ap.add_argument("--horizon-sweep", action="store_true")
    ap.add_argument("--leak-demo", action="store_true"); ap.add_argument("--store-fraction", type=float, default=0.5)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    a = ap.parse_args()
    if not (a.groups or a.horizon_sweep or a.leak_demo): a.groups = a.horizon_sweep = a.leak_demo = True
    chosen = json.load(open(a.choice))["chosen"]; params = json.load(open(a.params)).get(chosen, {})
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(a.features); conn = sqlite3.connect(a.db)
    print("Model: %s (frozen settings). Development period only; the test period is not used here." % chosen)
    if a.groups:
        print("\n1. Feature-group ablation (%.0f%% of stores, same sample for every run; intervals from a paired store bootstrap)" % (100 * a.store_fraction), flush=True)
        t = ablation(df, cfg, chosen, params, store_fraction=a.store_fraction, B=cfg["eval"]["bootstrap_B"], verbose=True)
        t.to_csv(out / "ablation.csv", index=False); print(t.round(2).to_string(index=False))
    if a.horizon_sweep:
        print("\n2. Horizon sweep", flush=True)
        t = horizon_sweep(conn, cfg, chosen, params, cfg["eval"]["horizon_sweep"], verbose=True)
        t.to_csv(out / "horizon_sweep.csv", index=False); print(t.round(3).to_string(index=False))
    if a.leak_demo:
        print("\n3. Leakage demonstration: adding same-day Customers (INVALID, shown only to justify excluding it)", flush=True)
        r = leak_demo(conn, df, cfg, chosen, params, store_fraction=a.store_fraction)
        (out / "leak_demo.json").write_text(json.dumps(r, indent=2)); print(json.dumps(r, indent=2))

if __name__ == "__main__":
    main()
