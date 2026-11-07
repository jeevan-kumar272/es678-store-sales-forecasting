"""Runs the final-stage scripts from the command line, in order, exactly as you will. Catches wiring mistakes that unit tests miss."""
import subprocess, sys, pathlib, json
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

def run(args, cwd=SRC):
    r = subprocess.run([sys.executable] + [str(a) for a in args], capture_output=True, text=True, cwd=str(cwd))
    assert r.returncode == 0, "%s failed:\n%s\n%s" % (args[0], r.stdout[-800:], r.stderr[-1500:])
    return r.stdout

def test_full_final_stage_chain(fake_features, tmp_path):
    feat = tmp_path / "f.pkl"; fake_features.to_pickle(feat)
    art, proc = tmp_path / "art", tmp_path / "proc"; art.mkdir(); proc.mkdir()
    (art / "best_params.json").write_text(json.dumps({"ridge": {"alpha": 5.0}, "lightgbm": {"n_estimators": 60}}))
    out = run(["run_cv.py", "--features", feat, "--models", "linear,ridge,lightgbm", "--params", art / "best_params.json",
               "--out-dir", art, "--oof", proc / "cv_oof.pkl"])
    assert "Pooled over all folds" in out
    out = run(["select_model.py", "--oof", proc / "cv_oof.pkl", "--candidates", "ridge,lightgbm", "--out", art / "model_choice.json"])
    assert "CHOSEN MODEL" in out
    chosen = json.loads((art / "model_choice.json").read_text())["chosen"]
    out = run(["final.py", "--features", feat, "--params", art / "best_params.json", "--choice", art / "model_choice.json",
               "--models", "linear,ridge,lightgbm", "--out-dir", art, "--processed-dir", proc])
    assert "scoring the test period ONCE" in out and (art / "test_lock.json").exists() and (art / "model.pkl").exists()
    again = subprocess.run([sys.executable, str(SRC / "final.py"), "--features", str(feat), "--params", str(art / "best_params.json"),
                            "--choice", str(art / "model_choice.json"), "--models", "linear,ridge,lightgbm",
                            "--out-dir", str(art), "--processed-dir", str(proc)], capture_output=True, text=True, cwd=str(SRC))
    assert again.returncode != 0 and "already scored" in again.stderr                     # the second run is refused
    out = run(["intervals.py", "--features", feat, "--oof", proc / "cv_oof.pkl", "--test-preds", proc / "test_preds.pkl",
               "--params", art / "best_params.json", "--choice", art / "model_choice.json", "--out-dir", art])
    assert "Gate 4" in out and (art / "intervals_summary.json").exists()
    out = run(["failure_audit.py", "--features", feat, "--test-preds", proc / "test_preds.pkl", "--oof", proc / "cv_oof.pkl",
               "--params", art / "best_params.json", "--choice", art / "model_choice.json", "--out-dir", art])
    assert "Cold start" in out and "smearing factor" in out
    out = run(["ablation.py", "--groups", "--features", feat, "--params", art / "best_params.json",
               "--choice", art / "model_choice.json", "--store-fraction", "1.0", "--out-dir", art])
    assert "none (full model)" in out and "history" in out and (art / "ablation.csv").exists()
