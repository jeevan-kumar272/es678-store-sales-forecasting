"""Run baselines and models through the rolling-origin folds and print one honest comparison table.
Usage: python src/run_cv.py [--models linear,ridge,lasso,lightgbm] [--seed 1] [--sanity]
Scoring rule: every model is scored on the SAME rows, those where the seasonal naive forecast exists.
The number of rows left out is printed."""
import argparse, json, pathlib
import numpy as np, pandas as pd
from utils import load_config, ROOT
from features import feature_columns
from cv import run_cv, shuffled_target_check
from metrics import summarize, naive_scale, store_sums, bootstrap_ci

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / ("features_h%d.pkl" % cfg["horizon_days"])))
    ap.add_argument("--models", default="linear,ridge,lasso,lightgbm")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sanity", action="store_true", help="also run the shuffled-target check")
    ap.add_argument("--params", default=None, help="JSON file with tuned parameters per model (from tune.py)")
    ap.add_argument("--extra-seeds", default="", help="more seeds for the random models, e.g. 2,3,4,5")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"), help="where the results tables are saved")
    ap.add_argument("--oof", default=str(ROOT / "data" / "processed" / "cv_oof.pkl"), help="where out-of-fold predictions are saved")
    a = ap.parse_args()
    df = pd.read_pickle(a.features)
    feats = feature_columns(df)
    models = [m for m in a.models.split(",") if m]
    dev = df[df.split == "dev"]
    scale = naive_scale(dev)

    if a.sanity:
        r = shuffled_target_check(df, cfg, feats, a.seed)
        print("Shuffled-target check:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()})
        print("  PASS" if r["passed"] else "  FAIL: investigate before trusting any result", "\n")

    print("Cross-validation on the development period (5 expanding-window folds)")
    params = json.load(open(a.params)) if a.params else {}
    extra = [int(s) for s in a.extra_seeds.split(",") if s]
    oof = run_cv(df, cfg, models, feats, a.seed, params=params, extra_seeds=extra)
    common = oof[oof.baseline_seasonal_naive.notna()].copy()
    print("\nScored on %d of %d validation rows (%d left out because the seasonal naive forecast does not exist)"
          % (len(common), len(oof), len(oof) - len(common)))
    preds = ["baseline_seasonal_naive", "baseline_wd_mean4"] + models
    tab = pd.DataFrame({p: summarize(common, p, scale) for p in preds}).T
    print("\nPooled over all folds (euros):")
    print(tab[["mae", "wape", "mase", "rmse", "rmspe", "r2", "r2_within_store", "skill_vs_naive"]].round(4).to_string())
    seed_cols = {m: [m] + [c for c in oof.columns if c.startswith(m + "__seed")] for m in models}
    if any(len(v) > 1 for v in seed_cols.values()):
        print("\nSeed stability (pooled MAE per seed; only the random models are repeated):")
        rows = {}
        for m, cols in seed_cols.items():
            if len(cols) > 1:
                v = np.array([np.abs(common.sales - common[c]).mean() for c in cols])
                rows[m] = {"seeds": len(cols), "mean_mae": v.mean(), "sd": v.std(ddof=1), "min": v.min(), "max": v.max()}
        print(pd.DataFrame(rows).T.round(2).to_string())
    print("\nMAE by fold:")
    print(common.groupby("fold").apply(lambda g: pd.Series({p: np.mean(np.abs(g.sales - g[p])) for p in preds}), include_groups=False).round(1).to_string())
    ci = bootstrap_ci(store_sums(common, preds), preds, "baseline_seasonal_naive", B=cfg["eval"]["bootstrap_B"], seed=cfg["seed"])
    print("\nStore-cluster bootstrap, 95%% intervals (%d resamples of whole stores):" % cfg["eval"]["bootstrap_B"])
    print(ci[["model", "mae", "mae_lo", "mae_hi", "wape", "wape_lo", "wape_hi"]].round(3).to_string(index=False))
    print("\nPaired difference in MAE against the seasonal naive (negative = better than naive):")
    print(ci.dropna(subset=["diff_lo"])[["model", "mae_diff_vs_baseline_seasonal_naive", "diff_lo", "diff_hi"]].round(1).to_string(index=False))
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "cv_results.csv"); ci.to_csv(out / "cv_bootstrap.csv", index=False)
    oof_path = pathlib.Path(a.oof); oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_pickle(oof_path)
    print("\nSaved %s, %s and %s" % (out / "cv_results.csv", out / "cv_bootstrap.csv", oof_path))

if __name__ == "__main__":
    main()
