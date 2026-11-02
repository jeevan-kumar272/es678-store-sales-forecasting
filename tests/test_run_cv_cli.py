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
