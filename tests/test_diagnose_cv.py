import subprocess, sys, pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

def test_diagnose_script_runs_on_cv_output(fake_features, tmp_path):
    feat = tmp_path / "f.pkl"; fake_features.to_pickle(feat)
    run = subprocess.run([sys.executable, str(SRC / "run_cv.py"), "--features", str(feat), "--models", "linear,ridge",
                          "--out-dir", str(tmp_path / "out"), "--oof", str(tmp_path / "oof.pkl")],
                         capture_output=True, text=True, cwd=str(SRC))
    assert run.returncode == 0, run.stderr[-1000:]
    r = subprocess.run([sys.executable, str(SRC / "diagnose_cv.py"), "--oof", str(tmp_path / "oof.pkl"), "--features", str(feat)],
                       capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0, r.stderr[-1500:]
    assert "How heavy is the error tail" in r.stdout and "Ten worst stores" in r.stdout and "by store_type" in r.stdout
