import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import joblib, numpy as np, pandas as pd, pytest
import final
from utils import load_config

CFG = load_config()
MODELS = ("linear", "ridge")
PARAMS = {"ridge": {"alpha": 5.0}}

def run(df, tmp, **kw):
    return final.fit_and_score(df, CFG, PARAMS, "ridge", models=MODELS, out_dir=tmp / "art", processed_dir=tmp / "proc",
                               lock_path=tmp / "art" / "lock.json", B=200, **kw)

def test_scores_once_writes_everything_and_refuses_a_second_run(fake_features, tmp_path):
    r = run(fake_features, tmp_path)
    for f in ["model.pkl", "schema.json", "metrics.json", "test_results.csv", "test_bootstrap.csv", "lock.json"]:
        assert (tmp_path / "art" / f).exists(), f
    assert (tmp_path / "proc" / "test_preds.pkl").exists()
    assert json.loads((tmp_path / "art" / "lock.json").read_text())["chosen_model"] == "ridge"
    with pytest.raises(RuntimeError):
        run(fake_features, tmp_path)
    m = json.loads((tmp_path / "art" / "metrics.json").read_text())
    assert m["mae_ci"][0] <= m["mae"] <= m["mae_ci"][1] and m["test_rows_scored"] <= m["test_rows_total"]

def test_only_development_rows_are_used_for_fitting(fake_features, tmp_path):
    a = run(fake_features, tmp_path / "a")["preds"]
    tampered = fake_features.copy()
    tampered.loc[tampered.split == "test", "sales"] *= 1000        # change the test TARGETS only
    b = run(tampered, tmp_path / "b")["preds"]
    assert np.allclose(a["ridge"], b["ridge"]) and np.allclose(a["linear"], b["linear"])   # the fitted models did not change

def test_saved_model_reproduces_the_saved_predictions(fake_features, tmp_path):
    r = run(fake_features, tmp_path)
    art = joblib.load(tmp_path / "art" / "model.pkl")
    test = fake_features[fake_features.split == "test"].head(50)
    p = art["model"].predict(test[art["columns"]])
    assert np.allclose(p, r["preds"].loc[test.index, "ridge"])
    assert art["trained_through"] == CFG["split"]["dev_end"] and art["horizon_days"] == CFG["horizon_days"]

def test_a_crash_before_scoring_does_not_use_up_the_test_period(fake_features, tmp_path):
    with pytest.raises(Exception):
        final.fit_and_score(fake_features, CFG, {"ridge": {"alpha": "bad"}}, "ridge", models=MODELS, out_dir=tmp_path / "art",
                            processed_dir=tmp_path / "proc", lock_path=tmp_path / "art" / "lock.json", B=50)
    assert not (tmp_path / "art" / "lock.json").exists()
