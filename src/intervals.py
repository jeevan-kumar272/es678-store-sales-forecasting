"""Prediction intervals for the chosen model, built two ways, and an honest check of their real coverage.
  1. Conformal-style: quantiles of out-of-fold log residuals from folds F4 and F5 (Christmas and early 2015), per store type.
  2. LightGBM quantile regression for the 10th and 90th percentile, fitted on development rows only.
Gate 4 (before looking at the test): the offsets from F4 and F5 should cover 75 to 85 percent of the earlier folds F1 to F3.
Coverage on the test period is reported as it is, even if it falls short. Nothing here changes the frozen model.
Usage: python src/intervals.py"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from utils import load_config, ROOT
from features import feature_columns
from modeling import make_model

CAL_FOLDS, CHECK_FOLDS = ("F4", "F5"), ("F1", "F2", "F3")

def log_resid(y, p):
    return np.log1p(np.asarray(y, float)) - np.log1p(np.asarray(p, float))

def quantile_offsets(res, group, taus, min_n=200):
    """Per-group quantiles of the log residuals. Groups with too few residuals use the pooled value."""
    res, group = np.asarray(res), np.asarray(group)
    out = {"all": tuple(float(q) for q in np.quantile(res, taus))}
    for g in np.unique(group[~pd.isna(group)]):
        r = res[group == g]
        if len(r) >= min_n:
            out[int(g)] = tuple(float(q) for q in np.quantile(r, taus))
    return out

def apply_offsets(pred, group, offsets, k):
    off = pd.Series(group).map(lambda g: offsets.get(int(g), offsets["all"])[k] if not pd.isna(g) else offsets["all"][k]).to_numpy(float)
    return np.expm1(np.log1p(np.asarray(pred, float)) + off)

def coverage_table(d, lo, hi, by):
    d = d.assign(inside=(d.sales >= lo) & (d.sales <= hi), below=d.sales < lo, above=d.sales > hi, width=(hi - lo) / d.pred)
    g = d.groupby(by)
    t = pd.DataFrame({"n": g.size(), "coverage": g.inside.mean(), "below_share": g.below.mean(), "above_share": g.above.mean(),
                      "median_width_over_forecast": g.width.median()})
    return t

def fit_quantile_models(dev, features, params, seed, level):
    a = (1 - level) / 2
    models = []
    for alpha in (a, 1 - a):
        p = dict(params or {}); p.update({"objective": "quantile", "alpha": alpha})
        est, cols = make_model("lightgbm", features, seed, p)
        est.fit(dev[cols], dev["sales"].to_numpy(float)); models.append((est, cols))
    return models

def newsvendor(test, pred, offsets, ratios=(1, 3, 5, 10)):
    """Cost of running short is `ratio` times the cost of surplus. Compare stocking at the point forecast with stocking at the
    matching quantile (tau = ratio / (ratio + 1)). Sales stand in for demand, which is a simplification."""
    y = test.sales.to_numpy(float); rows = []
    for r in ratios:
        tau = r / (r + 1.0)
        off = quantile_offsets(offsets["res"], offsets["group"], [tau])
        stock = apply_offsets(pred, test.store_type.to_numpy(), off, 0)
        cost = lambda s: float(np.mean(r * np.maximum(y - s, 0) + np.maximum(s - y, 0)))
        point, quant = cost(np.asarray(pred, float)), cost(stock)
        rows.append({"cost_ratio_short_to_surplus": r, "tau": round(tau, 3), "cost_point_forecast": point,
                     "cost_quantile_forecast": quant, "saving_pct": 100 * (1 - quant / point),
                     "share_of_days_short": float(np.mean(y > stock))})
    return pd.DataFrame(rows)

def run_intervals(features_df, oof, test_preds, chosen, params, cfg, level=0.8, seed=42, out_dir=None):
    out_dir = pathlib.Path(out_dir or ROOT / "artifacts"); out_dir.mkdir(parents=True, exist_ok=True)
    feats = feature_columns(features_df); a = (1 - level) / 2
    ctx = features_df[["store_key", "date", "store_type", "promo", "holiday_code"]]
    o = oof.merge(ctx, on=["store_key", "date"], how="left").rename(columns={chosen: "pred"})
    o["res"] = log_resid(o.sales, o.pred)
    cal, chk = o[o.fold.isin(CAL_FOLDS)], o[o.fold.isin(CHECK_FOLDS)]
    off = quantile_offsets(cal.res, cal.store_type, [a, 1 - a])

    lo, hi = apply_offsets(chk.pred, chk.store_type, off, 0), apply_offsets(chk.pred, chk.store_type, off, 1)
    gate_cov = float(((chk.sales >= lo) & (chk.sales <= hi)).mean())
    gate = {"coverage_on_F1_to_F3": gate_cov, "target": level, "passed": bool(0.75 <= gate_cov <= 0.85)}

    t = test_preds.merge(ctx, on=["store_key", "date"], how="left").rename(columns={chosen: "pred"})
    t["month"] = t.date.str[:7]
    clo, chi = apply_offsets(t.pred, t.store_type, off, 0), apply_offsets(t.pred, t.store_type, off, 1)
    dev = features_df[features_df.split == "dev"]
    (qlo_m, cols), (qhi_m, _) = fit_quantile_models(dev, feats, params.get("lightgbm"), seed, level)
    test_feat = features_df.set_index(["store_key", "date"]).loc[list(zip(t.store_key, t.date))].reset_index()
    q1, q2 = qlo_m.predict(test_feat[cols]), qhi_m.predict(test_feat[cols])
    qlo, qhi = np.minimum(q1, q2), np.maximum(q1, q2)

    tables, summary = [], {"level": level, "gate4": gate, "n_test_rows": int(len(t)), "conformal_offsets_log": {str(k): v for k, v in off.items()}}
    for name, l, h in (("conformal", clo, chi), ("quantile_regression", qlo, qhi)):
        summary[name] = {"coverage": float(((t.sales >= l) & (t.sales <= h)).mean()),
                         "median_width_over_forecast": float(np.median((h - l) / t.pred))}
        for by in ["store_type", "promo", "holiday_code", "month"]:
            ct = coverage_table(t, l, h, by).reset_index().rename(columns={by: "group"})
            ct.insert(0, "slice", by); ct.insert(0, "method", name); tables.append(ct)
    cov = pd.concat(tables, ignore_index=True)
    nv = newsvendor(t, t.pred, {"res": cal.res.to_numpy(), "group": cal.store_type.to_numpy()})
    cov.to_csv(out_dir / "intervals_coverage_by_group.csv", index=False); nv.to_csv(out_dir / "newsvendor.csv", index=False)
    (out_dir / "intervals_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "conformal_offsets.json").write_text(json.dumps({"level": level, "log_offsets": {str(k): v for k, v in off.items()}}, indent=2))
    return {"summary": summary, "coverage": cov, "newsvendor": nv}

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    P = ROOT / "data" / "processed"
    ap.add_argument("--features", default=str(P / ("features_h%d.pkl" % cfg["horizon_days"])))
    ap.add_argument("--oof", default=str(P / "cv_oof.pkl")); ap.add_argument("--test-preds", default=str(P / "test_preds.pkl"))
    ap.add_argument("--params", default=str(ROOT / "artifacts" / "best_params.json"))
    ap.add_argument("--choice", default=str(ROOT / "artifacts" / "model_choice.json"))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    params = json.load(open(a.params)); chosen = json.load(open(a.choice))["chosen"]
    r = run_intervals(pd.read_pickle(a.features), pd.read_pickle(a.oof), pd.read_pickle(a.test_preds), chosen, params, cfg,
                      level=cfg["eval"]["interval_level"], seed=cfg["seed"], out_dir=a.out_dir)
    s = r["summary"]; g = s["gate4"]
    print("Gate 4 (offsets from folds F4 and F5, checked on F1 to F3): coverage %.3f against a target of %.2f -> %s" % (
        g["coverage_on_F1_to_F3"], g["target"], "PASS" if g["passed"] else "OUTSIDE 75 to 85 percent, report it honestly"))
    print("\nTest period, nominal %.0f%% intervals:" % (100 * s["level"]))
    for k in ("conformal", "quantile_regression"):
        print("  %-20s real coverage %.3f   median width / forecast %.3f" % (k, s[k]["coverage"], s[k]["median_width_over_forecast"]))
    print("\nCoverage by slice:"); print(r["coverage"].round(3).to_string(index=False))
    print("\nStocking to a quantile instead of the point forecast (cost of running short = ratio x cost of surplus):")
    print(r["newsvendor"].round(2).to_string(index=False))
    print("\nSaved intervals_summary.json, intervals_coverage_by_group.csv, newsvendor.csv and conformal_offsets.json in artifacts/.")

if __name__ == "__main__":
    main()
