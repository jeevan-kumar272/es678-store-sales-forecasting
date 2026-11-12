import subprocess, sys, pathlib, json
import pandas as pd
import etl, features, ablation as A
from make_fake_data import make
from utils import load_config

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
CFG = load_config()

def run(args):
    r = subprocess.run([sys.executable] + [str(x) for x in args], capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0, "%s failed:\n%s\n%s" % (args[0], r.stdout[-600:], r.stderr[-1500:])
    return r.stdout

def test_every_figure_is_made_from_saved_results(tmp_path):
    raw = make(tmp_path / "raw", n_stores=40); conn = etl.connect(tmp_path / "w.db")
    etl.build(conn, raw, replay_days=3, expected_stores=40)
    df = features.build_features(conn, CFG, 7); feat = tmp_path / "f.pkl"; df.to_pickle(feat)
    art, proc = tmp_path / "art", tmp_path / "proc"; art.mkdir(); proc.mkdir()
    (art / "best_params.json").write_text(json.dumps({"ridge": {"alpha": 5.0}, "lightgbm": {"n_estimators": 60}}))
    run(["run_cv.py", "--features", feat, "--models", "linear,ridge,lightgbm", "--params", art / "best_params.json", "--out-dir", art, "--oof", proc / "oof.pkl"])
    run(["select_model.py", "--oof", proc / "oof.pkl", "--candidates", "ridge,lightgbm", "--out", art / "model_choice.json"])
    run(["final.py", "--features", feat, "--params", art / "best_params.json", "--choice", art / "model_choice.json",
         "--models", "linear,ridge,lightgbm", "--out-dir", art, "--processed-dir", proc])
    run(["intervals.py", "--features", feat, "--oof", proc / "oof.pkl", "--test-preds", proc / "test_preds.pkl",
         "--params", art / "best_params.json", "--choice", art / "model_choice.json", "--out-dir", art])
    run(["failure_audit.py", "--features", feat, "--test-preds", proc / "test_preds.pkl", "--oof", proc / "oof.pkl",
         "--params", art / "best_params.json", "--choice", art / "model_choice.json", "--out-dir", art])
    run(["ablation.py", "--groups", "--features", feat, "--params", art / "best_params.json", "--choice", art / "model_choice.json",
         "--store-fraction", "1.0", "--out-dir", art])
    A.horizon_sweep(conn, CFG, "ridge", {"alpha": 5.0}, [1, 14]).to_csv(art / "horizon_sweep.csv", index=False)
    out = tmp_path / "figs"
    run(["make_figures.py", "--features", feat, "--test-preds", proc / "test_preds.pkl", "--db", tmp_path / "w.db", "--artifacts", art, "--out", out])
    for f in ["fig_data", "fig_results", "fig_intervals", "fig_ablation_horizon", "fig_residuals", "fig_cases"]:
        p = out / (f + ".png"); assert p.exists() and p.stat().st_size > 8000, f
