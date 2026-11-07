"""Hyperparameter tuning with Optuna on the rolling-origin folds (development period only).
Objective: mean MAE in euros over the five validation folds, scored on the rows where the seasonal naive exists.
The same trial budget is used for every model. To keep LightGBM and Lasso affordable, tuning runs on a fixed, seeded sample
of STORES (all dates kept, so the time structure is intact); the final comparison then uses all stores.
Usage: python src/tune.py [--models ridge,lasso,lightgbm] [--seeds 1,2,3] [--trials 40] [--store-fraction 0.25] [--resume]
Results are saved after every finished study, so an interrupted run can be resumed."""
import argparse, json, pathlib, time
import numpy as np, pandas as pd, optuna
from utils import load_config, ROOT
from features import feature_columns
from cv import fold_indices
from modeling import make_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

SPACES = {
    "lightgbm": lambda t: {"n_estimators": t.suggest_int("n_estimators", 200, 800),
                           "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                           "num_leaves": t.suggest_int("num_leaves", 15, 127, log=True),
                           "min_child_samples": t.suggest_int("min_child_samples", 20, 400, log=True),
                           "subsample": t.suggest_float("subsample", 0.5, 1.0),
                           "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                           "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30.0, log=True)},
    "ridge": lambda t: {"alpha": t.suggest_float("alpha", 1e-2, 1e3, log=True)},
    "lasso": lambda t: {"alpha": t.suggest_float("alpha", 1e-4, 1e-1, log=True)},   # below 1e-4 it is nearly least squares and very slow
}

def sample_stores(df, fraction, seed):
    """A fixed random sample of whole stores. Every date of a chosen store is kept."""
    if fraction >= 1.0:
        return df
    stores = np.sort(df["store_key"].unique())
    keep = np.random.default_rng(seed).choice(stores, max(1, int(round(fraction * len(stores)))), replace=False)
    return df[df["store_key"].isin(keep)].reset_index(drop=True)

def cv_mae(df, cfg, name, features, params, seed):
    maes = []
    for _, tr, va in fold_indices(df, cfg):
        est, cols = make_model(name, features, seed, params)
        est.fit(df.iloc[tr][cols], df.iloc[tr]["sales"].to_numpy(float))
        v = df.iloc[va]; ok = v["baseline_seasonal_naive"].notna().to_numpy()
        maes.append(float(np.mean(np.abs(v["sales"].to_numpy(float)[ok] - est.predict(v[cols])[ok]))))
    return float(np.mean(maes)), maes

def tune_model(df, cfg, name, features, seed, n_trials, store_fraction=1.0, verbose=False):
    sub = sample_stores(df, store_fraction, cfg["seed"])
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.time()
    def objective(trial):
        params = SPACES[name](trial)
        value, _ = cv_mae(sub, cfg, name, features, params, seed)
        if verbose:
            print("  %s seed %d trial %2d/%d  MAE %.1f  (%.0fs elapsed)" % (name, seed, trial.number + 1, n_trials, value, time.time() - t0), flush=True)
        return value
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_value": float(study.best_value), "best_params": study.best_params, "n_trials": n_trials,
            "trial_values": [float(t.value) for t in study.trials], "stores_used": int(sub["store_key"].nunique()),
            "seconds": round(time.time() - t0)}

def summarize(results):
    """Best params per model across seeds, plus how much the seeds disagree."""
    best, summary = {}, {}
    for m, seeds in results.items():
        vals = {s: r["best_value"] for s, r in seeds.items()}
        s_best = min(vals, key=vals.get)
        best[m] = seeds[s_best]["best_params"]
        summary[m] = {"best_seed": s_best, "best_value_per_seed": vals, "spread": max(vals.values()) - min(vals.values())}
    return best, summary

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / ("features_h%d.pkl" % cfg["horizon_days"])))
    ap.add_argument("--models", default="ridge,lasso,lightgbm")
    ap.add_argument("--seeds", default=",".join(str(s) for s in cfg["models"]["optuna_seeds"]))
    ap.add_argument("--trials", type=int, default=cfg["models"]["optuna_trials"])
    ap.add_argument("--store-fraction", type=float, default=0.25)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "tuning_results.json"
    results = json.loads(path.read_text()) if a.resume and path.exists() else {}
    df = pd.read_pickle(a.features); feats = feature_columns(df)
    for m in [x for x in a.models.split(",") if x]:
        for s in [int(x) for x in a.seeds.split(",") if x]:
            if str(s) in results.get(m, {}):
                print("skipping %s seed %d (already done)" % (m, s)); continue
            frac = a.store_fraction if m in ("lightgbm", "lasso") else 1.0   # ridge is cheap and uses every store
            print("tuning %s, seed %d, %d trials, store fraction %.2f" % (m, s, a.trials, frac), flush=True)
            results.setdefault(m, {})[str(s)] = tune_model(df, cfg, m, feats, s, a.trials, frac, verbose=True)
            path.write_text(json.dumps(results, indent=2))
    best, summary = summarize(results)
    (out / "best_params.json").write_text(json.dumps(best, indent=2))
    print("\nBest settings per model (lowest mean-fold MAE across the tuning seeds):")
    for m in best:
        print(" ", m, best[m]); print("     best MAE per tuning seed:", {k: round(v, 1) for k, v in summary[m]["best_value_per_seed"].items()}, " spread %.1f" % summary[m]["spread"])
    print("\nSaved %s and %s" % (path, out / "best_params.json"))

if __name__ == "__main__":
    main()
