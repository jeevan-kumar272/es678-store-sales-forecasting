import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import joblib, numpy as np, pandas as pd, pytest
from fastapi.testclient import TestClient
import cv, final, features, intervals, serve
from utils import load_config

CFG = load_config()
PARAMS = {"lightgbm": {"n_estimators": 60}}

@pytest.fixture(scope="module")
def stack(fake_features, tmp_path_factory):
    d = tmp_path_factory.mktemp("srv"); feats = features.feature_columns(fake_features)
    oof = cv.run_cv(fake_features, CFG, ["lightgbm"], feats, seed=1, params=PARAMS, verbose=False)
    res = final.fit_and_score(fake_features, CFG, PARAMS, "lightgbm", models=("linear", "lightgbm"), out_dir=d / "a",
                              processed_dir=d / "p", lock_path=d / "a" / "lock.json", B=50)
    intervals.run_intervals(fake_features, oof, res["preds"], "lightgbm", PARAMS, CFG, out_dir=d / "a")
    art = joblib.load(d / "a" / "model.pkl"); off = json.load(open(d / "a" / "conformal_offsets.json"))
    f = serve.Forecaster(fake_features, art, off, min_date="2015-01-01")
    return f, TestClient(serve.create_app(f)), res["preds"]

def test_api_reproduces_the_frozen_test_predictions(stack):
    f, client, preds = stack
    row = preds.iloc[100]
    r = client.post("/forecast", json={"store_id": int(row.store_key), "target_date": row.date}).json()
    assert np.isclose(r["forecast"], row.lightgbm, atol=0.1) and np.isclose(r["actual_sales"], row.sales)
    assert r["interval_low"] < r["forecast"] < r["interval_high"] and r["mode"] == "replay"
    assert r["forecast_origin"] == (pd.Timestamp(row.date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")

def test_what_if_override_changes_the_forecast_and_is_reported(stack):
    f, client, preds = stack
    row = preds.iloc[200]; base = client.post("/forecast", json={"store_id": int(row.store_key), "target_date": row.date}).json()
    other = 1 - int(f.df[(f.df.store_key == row.store_key) & (f.df.date == row.date)].promo.iloc[0])
    alt = client.post("/forecast", json={"store_id": int(row.store_key), "target_date": row.date, "promo": other}).json()
    assert alt["overrides_applied"] == {"promo": other} and alt["forecast"] != base["forecast"]

def test_unknown_or_unsupported_requests_are_rejected_clearly(stack):
    _, client, _ = stack
    assert client.post("/forecast", json={"store_id": 1, "target_date": "2030-01-01"}).status_code == 404
    assert client.post("/forecast", json={"store_id": 0, "target_date": "2015-05-05"}).status_code == 422
    assert client.post("/forecast", json={"store_id": 1, "target_date": "05/05/2015"}).status_code == 422
    assert client.post("/forecast", json={"store_id": 1, "target_date": "2015-05-05", "state_holiday": "z"}).status_code == 422

def test_health_and_schema_endpoints(stack):
    _, client, _ = stack
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "lightgbm" and h["horizon_days"] == 7
    assert client.get("/schema").status_code == 200

def test_feature_contributions_add_up_to_the_forecast(stack):
    f, _, preds = stack
    row = preds.iloc[300]; ex = f.explain(int(row.store_key), row.date, top=100)
    total = sum(e["log_effect"] for e in ex)
    assert np.isclose(np.expm1(total), f.forecast(int(row.store_key), row.date)["forecast"], rtol=1e-3)

def test_history_for_the_chart_is_ordered_and_contains_the_target(stack):
    f, _, preds = stack
    row = preds.iloc[400]; h = f.history(int(row.store_key), row.date, days=14)
    assert h.date.is_monotonic_increasing and row.date in set(h.date) and (h.lo <= h.hi).all()
