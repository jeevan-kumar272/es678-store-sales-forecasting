"""Freeze the chosen model and score the test period ONCE.

Reads the tuned settings (artifacts/best_params.json) and the model chosen by the pre-declared rule
(artifacts/model_choice.json). Fits every model on the development rows only, predicts the test rows, scores all of them
on the same rows with store-cluster bootstrap intervals, saves the chosen model as a .pkl file, and writes a lock file with
the git commit. A second run is refused. The lock is written only after the scores exist, so a crash before that point
does not use up the test period.
Usage: python src/final.py [--models linear,ridge,lasso,lightgbm]"""
import argparse, datetime, json, pathlib, subprocess
import joblib, numpy as np, pandas as pd
from utils import load_config, ROOT
from features import feature_columns
from modeling import make_model
from metrics import summarize, naive_scale, store_sums, bootstrap_ci

def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, cwd=ROOT).strip()
    except Exception:
        return "unknown (not a git repository)"

def fit_and_score(df, cfg, params, chosen, models=("linear", "ridge", "lasso", "lightgbm"), seed=None,
                  out_dir=None, processed_dir=None, lock_path=None, B=None):
    out_dir = pathlib.Path(out_dir or ROOT / "artifacts"); processed_dir = pathlib.Path(processed_dir or ROOT / "data" / "processed")
    lock_path = pathlib.Path(lock_path or out_dir / "test_lock.json")
    if lock_path.exists():
        raise RuntimeError("The test period was already scored (%s). Only a new commit with a logged reason can unlock it." % lock_path)
    assert chosen in models, "the chosen model must be among the models being fitted"
    seed = cfg["seed"] if seed is None else seed
    B = B or cfg["eval"]["bootstrap_B"]
    feats = feature_columns(df)
    dev, test = df[df.split == "dev"], df[df.split == "test"]
    assert dev["date"].max() <= cfg["split"]["dev_end"] and test["date"].min() >= cfg["split"]["test_start"]

    pred = test[["store_key", "date", "sales", "baseline_seasonal_naive", "baseline_wd_mean4"]].copy()
    fitted = {}
    for m in models:
        est, cols = make_model(m, feats, seed, params.get(m))
        est.fit(dev[cols], dev["sales"].to_numpy(float))           # development rows only
        pred[m] = est.predict(test[cols]); fitted[m] = (est, cols)

    common = pred[pred.baseline_seasonal_naive.notna()]
    scale = naive_scale(dev)
    names = ["baseline_seasonal_naive", "baseline_wd_mean4"] + list(models)
    table = pd.DataFrame({p: summarize(common, p, scale) for p in names}).T
    ci = bootstrap_ci(store_sums(common, names), names, "baseline_seasonal_naive", B=B, seed=cfg["seed"])
    ci_vs_wd = bootstrap_ci(store_sums(common, names), names, "baseline_wd_mean4", B=B, seed=cfg["seed"])

    est, cols = fitted[chosen]
    out_dir.mkdir(parents=True, exist_ok=True); processed_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": est, "columns": cols, "features": feats, "name": chosen, "params": params.get(chosen, {}),
                 "horizon_days": cfg["horizon_days"], "trained_through": cfg["split"]["dev_end"], "seed": seed},
                out_dir / "model.pkl")
    schema = {"store_id": "integer, 1 to 1115", "target_date": "YYYY-MM-DD, at most horizon_days after the last loaded day",
              "promo": "0 or 1, planned promotion on the target day", "state_holiday": "0, a, b or c",
              "school_holiday": "0 or 1", "horizon_days": cfg["horizon_days"],
              "example": {"store_id": 1, "target_date": "2015-07-31", "promo": 1, "state_holiday": "0", "school_holiday": 0}}
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2))
    pred.to_pickle(processed_dir / "test_preds.pkl")
    table.to_csv(out_dir / "test_results.csv"); ci.to_csv(out_dir / "test_bootstrap.csv", index=False)
    ci_vs_wd.to_csv(out_dir / "test_bootstrap_vs_wd_mean4.csv", index=False)
    row = ci[ci.model == chosen].iloc[0]
    metrics = {"chosen_model": chosen, "test_rows_scored": int(len(common)), "test_rows_total": int(len(pred)),
               "mae": float(row.mae), "mae_ci": [float(row.mae_lo), float(row.mae_hi)], "wape": float(row.wape),
               "wape_ci": [float(row.wape_lo), float(row.wape_hi)], "rmse": float(row.rmse),
               "skill_vs_naive": float(table.loc[chosen, "skill_vs_naive"]), "mase": float(table.loc[chosen, "mase"]),
               "mae_diff_vs_naive": float(row["mae_diff_vs_baseline_seasonal_naive"]),
               "mae_diff_vs_naive_ci": [float(row.diff_lo), float(row.diff_hi)], "bootstrap_resamples": int(B),
               "trained_through": cfg["split"]["dev_end"], "test_period": [cfg["split"]["test_start"], cfg["split"]["test_end"]]}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lock_path.write_text(json.dumps({"git_commit": _git_commit(), "time": datetime.datetime.now().isoformat(timespec="seconds"),
                                     "chosen_model": chosen}, indent=2))        # written only after the scores exist
    return {"table": table, "ci": ci, "ci_vs_wd": ci_vs_wd, "metrics": metrics, "preds": pred}

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / ("features_h%d.pkl" % cfg["horizon_days"])))
    ap.add_argument("--params", default=str(ROOT / "artifacts" / "best_params.json"))
    ap.add_argument("--choice", default=str(ROOT / "artifacts" / "model_choice.json"))
    ap.add_argument("--models", default="linear,ridge,lasso,lightgbm")
    ap.add_argument("--out-dir", default=None); ap.add_argument("--processed-dir", default=None)
    a = ap.parse_args()
    params = json.load(open(a.params)); chosen = json.load(open(a.choice))["chosen"]
    df = pd.read_pickle(a.features)
    print("Chosen model (from the pre-declared rule): %s" % chosen)
    print("Fitting on development rows only, then scoring the test period ONCE ...", flush=True)
    r = fit_and_score(df, cfg, params, chosen, tuple(m for m in a.models.split(",") if m), out_dir=a.out_dir, processed_dir=a.processed_dir)
    t, ci = r["table"], r["ci"]
    print("\nTest period %s to %s: scored on %d of %d rows (rows without a seasonal naive forecast are left out)" % (
        cfg["split"]["test_start"], cfg["split"]["test_end"], r["metrics"]["test_rows_scored"], r["metrics"]["test_rows_total"]))
    print("\nTest results (euros):")
    print(t[["mae", "wape", "mase", "rmse", "rmspe", "r2", "r2_within_store", "skill_vs_naive"]].round(4).to_string())
    print("\nStore-cluster bootstrap, 95%% intervals (%d resamples):" % cfg["eval"]["bootstrap_B"])
    print(ci[["model", "mae", "mae_lo", "mae_hi", "wape", "wape_lo", "wape_hi"]].round(3).to_string(index=False))
    print("\nPaired difference in MAE against the seasonal naive (negative = better):")
    print(ci.dropna(subset=["diff_lo"])[["model", "mae_diff_vs_baseline_seasonal_naive", "diff_lo", "diff_hi"]].round(1).to_string(index=False))
    print("\nSaved model.pkl, schema.json, metrics.json, test_results.csv, test_bootstrap.csv and the lock file in artifacts/.")
    print("The test period is now used up. Any further analysis reads these saved predictions; it never changes the model.")

if __name__ == "__main__":
    main()
