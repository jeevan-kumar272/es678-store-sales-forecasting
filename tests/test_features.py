import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, pytest
from make_fake_data import make
import etl, features
from utils import load_config

N_STORES, H = 30, 7
CFG = load_config()

@pytest.fixture(scope="module")
def wh(tmp_path_factory):
    d = tmp_path_factory.mktemp("feat")
    raw = make(d / "raw", n_stores=N_STORES)
    conn = etl.connect(d / "t.db")
    etl.build(conn, raw, replay_days=3, expected_stores=N_STORES)
    return conn

@pytest.fixture(scope="module")
def feats(wh):
    return features.build_features(wh, CFG, H)

def clone(conn):
    new = sqlite3.connect(":memory:")
    conn.backup(new)
    return new

def aligned(a, b):
    cols = features.feature_columns(a)
    a = a.set_index(["store_key", "date"])[cols]; b = b.set_index(["store_key", "date"])[cols]
    return a, b

def series(conn, store):
    s = pd.read_sql("""SELECT d.full_date AS date, CASE WHEN f.open_flag = 1 THEN f.sales END AS y
                       FROM dim_date d LEFT JOIN fact_daily_sales f ON f.date_key = d.date_key AND f.store_key = ?
                       ORDER BY d.date_key""", conn, params=(store,))
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date")["y"]

def test_history_features_match_an_independent_pandas_calculation(wh, feats):
    rng = np.random.default_rng(1)
    rows = feats[feats.roll_n_28 >= 10].sample(150, random_state=1)
    for r in rows.itertuples():
        s = series(wh, r.store_key); T = pd.Timestamp(r.date); i = T - pd.Timedelta(days=H)
        for k in (7, 14, 21, 28):
            assert np.isclose(s.get(T - pd.Timedelta(days=k), np.nan), getattr(r, "lag_%d" % k), equal_nan=True)
        for W in (7, 28, 56):
            w = s[i - pd.Timedelta(days=W - 1): i]
            assert getattr(r, "roll_n_%d" % W) == w.count()
            if w.count() >= features.MIN_OBS[W]:
                assert np.isclose(w.mean(), getattr(r, "roll_mean_%d" % W), rtol=1e-4)
        w = s[i - pd.Timedelta(days=27): i].dropna()
        x = (w.index - w.index[0]).days.values
        assert np.isclose(np.polyfit(x, w.values, 1)[0], r.slope_28, rtol=1e-3, atol=1e-3)
        past = s[: i].dropna()
        assert np.isclose(past.mean(), r.store_mean_past, rtol=1e-4)

def test_changing_sales_after_the_origin_changes_no_feature(wh, feats):
    cutoff = "2014-06-03"   # a Tuesday, so cutoff + 7 days (also a Tuesday) is a day with real target rows
    cut_key = int(cutoff.replace("-", ""))
    c2 = clone(wh)
    c2.execute("UPDATE fact_daily_sales SET sales = sales * 10 WHERE date_key > ?", (cut_key,)); c2.commit()
    f2 = features.build_features(c2, CFG, H)
    limit = (pd.Timestamp(cutoff) + pd.Timedelta(days=H)).strftime("%Y-%m-%d")
    a, b = aligned(feats[feats.date <= limit], f2[f2.date <= limit])
    pd.testing.assert_frame_equal(a, b)                       # identical: nothing after the origin leaked in
    later_a, later_b = aligned(feats[feats.date > "2014-07-15"], f2[f2.date > "2014-07-15"])
    assert not later_a.equals(later_b)                         # sanity: the change does matter for later rows

def test_customers_never_affects_any_feature(wh, feats):
    c2 = clone(wh)
    c2.execute("UPDATE fact_daily_sales SET customers = customers * 7 + 3"); c2.commit()
    a, b = aligned(feats, features.build_features(c2, CFG, H))
    pd.testing.assert_frame_equal(a, b)

def test_no_forbidden_columns(feats):
    cols = features.feature_columns(feats)
    assert not any("customer" in c for c in cols)
    assert "sales" not in cols and "split" not in cols
    assert not any(c.startswith("baseline_") for c in cols)

def test_warmup_and_split_labels(feats):
    first = pd.Timestamp("2013-01-01") + pd.Timedelta(days=CFG["split"]["warmup_days"] + H)
    assert pd.Timestamp(feats.date.min()) >= first
    assert feats[feats.split == "dev"].date.max() <= CFG["split"]["dev_end"]
    assert feats[feats.split == "test"].date.min() >= CFG["split"]["test_start"]
    assert set(feats.split.unique()) <= {"dev", "test", "after"}

def test_missing_days_stay_empty_not_zero(feats):
    gap_store = feats[(feats.store_key == 1) & (feats.date == "2015-01-05")]   # origin 2014-12-29 is inside store 1's gap
    assert len(gap_store) == 1
    r = gap_store.iloc[0]
    assert np.isnan(r.lag_7) and np.isnan(r.roll_mean_7) and np.isnan(r.roll_mean_28)
    assert r.roll_n_7 == 0 and r.roll_n_28 == 0

def test_only_open_days_with_sales_are_targets(wh, feats):
    n = wh.execute("SELECT COUNT(*) FROM fact_daily_sales WHERE open_flag = 1 AND sales > 0").fetchone()[0]
    assert len(feats) <= n and (feats.sales > 0).all()

def test_baselines_are_same_weekday_sales(feats):
    ok = feats.baseline_seasonal_naive.notna()
    assert np.allclose(feats.loc[ok, "baseline_seasonal_naive"], feats.loc[ok, "lag_7"])


def test_the_leak_detector_catches_a_planted_one(wh, feats, monkeypatch):
    """Self-check: if history windows ended one day too late, the perturbation test must fail."""
    orig = features.history_query
    monkeypatch.setattr(features, "history_query", lambda h, l, w, wu: orig(h - 1, l, w, wu))
    leaky = features.build_features(wh, CFG, H)
    c2 = clone(wh)
    c2.execute("UPDATE fact_daily_sales SET sales = sales * 10 WHERE date_key > 20140603"); c2.commit()
    leaky2 = features.build_features(c2, CFG, H)
    a, b = aligned(leaky[leaky.date <= "2014-06-10"], leaky2[leaky2.date <= "2014-06-10"])
    assert not a.equals(b)
