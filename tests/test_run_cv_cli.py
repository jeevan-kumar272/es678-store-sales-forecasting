import subprocess, sys, pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

def test_command_line_comparison_runs_end_to_end(fake_features, tmp_path):
    feat = tmp_path / "f.pkl"; fake_features.to_pickle(feat)
    r = subprocess.run([sys.executable, str(SRC / "run_cv.py"), "--features", str(feat), "--models", "linear,ridge",
                        "--sanity", "--out-dir", str(tmp_path / "out"), "--oof", str(tmp_path / "oof.pkl")],
                       capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0, r.stderr[-1500:]
    assert "PASS" in r.stdout and "Pooled over all folds" in r.stdout and "Paired difference" in r.stdout
    assert (tmp_path / "out" / "cv_results.csv").exists() and (tmp_path / "oof.pkl").exists()


def test_tuned_params_and_extra_seeds_are_used(fake_features, tmp_path):
    import json
    feat = tmp_path / "f.pkl"; fake_features.to_pickle(feat)
    (tmp_path / "p.json").write_text(json.dumps({"lightgbm": {"n_estimators": 40}, "ridge": {"alpha": 5.0}}))
    r = subprocess.run([sys.executable, str(SRC / "run_cv.py"), "--features", str(feat), "--models", "ridge,lightgbm",
                        "--params", str(tmp_path / "p.json"), "--extra-seeds", "2,3",
                        "--out-dir", str(tmp_path / "out"), "--oof", str(tmp_path / "oof.pkl")],
                       capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0, r.stderr[-1500:]
    assert "Seed stability" in r.stdout
    import pandas as pd
    oof = pd.read_pickle(tmp_path / "oof.pkl")
    assert {"lightgbm__seed2", "lightgbm__seed3"} <= set(oof.columns) and "ridge__seed2" not in oof.columns
