"""Apply the pre-declared model choice rule to cross-validation results (development period only).
Rule: take the model with the lowest mean fold MAE; every model whose mean fold MAE is within one standard error of
that best model counts as 'as good'; among those, choose the simplest. Order of simplicity: ridge, lasso, random forest, LightGBM.
Usage: python src/select_model.py [--oof data/processed/cv_oof.pkl] [--candidates ridge,lasso,lightgbm]"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from utils import ROOT

SIMPLICITY = ["ridge", "lasso", "random_forest", "lightgbm"]      # simplest first

def fold_mae_table(oof, models):
    """Per-fold MAE for each model on the rows where the seasonal naive exists. Random models: mean over their seeds."""
    common = oof[oof["baseline_seasonal_naive"].notna()]
    cols = {}
    for m in models:
        runs = [m] + [c for c in oof.columns if c.startswith(m + "__seed")]
        per = [common.groupby("fold").apply(lambda g, c=c: np.abs(g.sales - g[c]).mean(), include_groups=False) for c in runs]
        cols[m] = pd.concat(per, axis=1).mean(axis=1)
    return pd.DataFrame(cols)

def choose_model(fold_mae):
    """fold_mae: rows = folds, columns = candidate models. Returns (chosen, details dict)."""
    mean = fold_mae.mean()
    best = mean.idxmin()
    se = fold_mae[best].std(ddof=1) / np.sqrt(len(fold_mae))
    threshold = mean[best] + se
    within = [m for m in fold_mae.columns if mean[m] <= threshold]
    order = {m: i for i, m in enumerate(SIMPLICITY)}
    chosen = sorted(within, key=lambda m: (order.get(m, 99), mean[m]))[0]
    diffs = fold_mae.sub(fold_mae[best], axis=0)
    return chosen, {"best_mean_mae": float(mean[best]), "best_model": best, "standard_error": float(se), "threshold": float(threshold),
                    "mean_fold_mae": {m: float(v) for m, v in mean.items()}, "within_one_se": within,
                    "folds_won_by_best": {m: int((diffs[m] > 0).sum()) for m in fold_mae.columns if m != best}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(ROOT / "data" / "processed" / "cv_oof.pkl"))
    ap.add_argument("--candidates", default="ridge,lasso,lightgbm")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "model_choice.json"))
    a = ap.parse_args()
    oof = pd.read_pickle(a.oof)
    models = [m for m in a.candidates.split(",") if m]
    tab = fold_mae_table(oof, models)
    chosen, d = choose_model(tab)
    print("Mean absolute error per fold (euros):"); print(tab.round(1).to_string())
    print("\nBest model: %s, mean fold MAE %.1f, standard error across folds %.1f" % (d["best_model"], d["best_mean_mae"], d["standard_error"]))
    print("One-standard-error threshold: %.1f" % d["threshold"])
    print("Models within the threshold:", d["within_one_se"])
    print("Folds in which the best model beat each other model:", d["folds_won_by_best"], "of", len(tab))
    print("\nCHOSEN MODEL (simplest within one standard error): %s" % chosen)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"chosen": chosen, **d}, indent=2))
    print("Saved", out)

if __name__ == "__main__":
    main()
