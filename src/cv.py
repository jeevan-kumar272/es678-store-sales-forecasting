"""Rolling-origin cross-validation on the development period only, plus the shuffled-target sanity check.
Each fold trains on target dates before the validation block and validates on the 42-day block.
The test period is never touched here."""
import numpy as np, pandas as pd
from modeling import make_model

STOCHASTIC = {"lightgbm", "random_forest"}      # models whose result depends on the random seed

def fold_indices(df, cfg):
    dev = df["split"].eq("dev").to_numpy(); d = df["date"].to_numpy()
    for f in cfg["split"]["folds"]:
        tr = dev & (d < f["val_start"])
        va = dev & (d >= f["val_start"]) & (d <= f["val_end"])
        yield f["name"], np.flatnonzero(tr), np.flatnonzero(va)

def run_cv(df, cfg, model_names, features, seed=1, params=None, verbose=True, extra_seeds=()):
    """Returns one row per validation store-day with the fold, the target, both baselines and one column per model."""
    params = params or {}
    parts = []
    for name, tr, va in fold_indices(df, cfg):
        assert df.iloc[tr]["split"].eq("dev").all() and df.iloc[va]["split"].eq("dev").all()
        assert df.iloc[tr]["date"].max() < df.iloc[va]["date"].min()
        out = df.iloc[va][["store_key", "date", "sales", "baseline_seasonal_naive", "baseline_wd_mean4"]].copy()
        out["fold"] = name
        for m in model_names:
            seeds = [seed] + ([s for s in extra_seeds if s != seed] if m in STOCHASTIC else [])
            for k, s in enumerate(seeds):
                est, cols = make_model(m, features, s, params.get(m))
                est.fit(df.iloc[tr][cols], df.iloc[tr]["sales"].to_numpy(float))
                out[m if k == 0 else "%s__seed%d" % (m, s)] = est.predict(df.iloc[va][cols])
            if verbose: print("  fold %s  %-13s trained on %d rows (%d seed%s), scored %d" % (
                name, m, len(tr), len(seeds), "" if len(seeds) == 1 else "s", len(va)), flush=True)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)

def shuffled_target_check(df, cfg, features, seed=1):
    """Train on fold 1 with the training targets shuffled across all rows. The link between features and target is
    destroyed, so the model must be no better than a constant forecast. If it is better, information is reaching the
    model through some channel other than the features, and something in the pipeline is leaking."""
    name, tr, va = next(fold_indices(df, cfg))
    rng = np.random.default_rng(seed)
    y_tr = df.iloc[tr]["sales"].to_numpy(float); y_va = df.iloc[va]["sales"].to_numpy(float)
    constant = np.mean(np.abs(y_va - y_tr.mean()))
    est, cols = make_model("lightgbm", features, seed, {"n_estimators": 200})
    est.fit(df.iloc[tr][cols], rng.permutation(y_tr))
    shuffled = np.mean(np.abs(y_va - est.predict(df.iloc[va][cols])))
    est2, _ = make_model("lightgbm", features, seed, {"n_estimators": 200})
    est2.fit(df.iloc[tr][cols], y_tr)
    real = np.mean(np.abs(y_va - est2.predict(df.iloc[va][cols])))
    return {"constant_mae": constant, "shuffled_model_mae": shuffled, "real_model_mae": real,
            "passed": bool(shuffled >= 0.95 * constant and real < 0.9 * constant)}
