import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
from make_fake_data import make
import etl

N_STORES = 30

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("wh")
    raw = make(d / "raw", n_stores=N_STORES)
    conn = etl.connect(d / "test.db")
    etl.build(conn, raw, replay_days=5, expected_stores=N_STORES)
    return conn, raw

def count(conn, sql):
    return conn.execute(sql).fetchone()[0]

def test_every_reconciliation_check_passes(built):
    conn, _ = built
    assert count(conn, "SELECT COUNT(*) FROM recon_results") > 0
    assert count(conn, "SELECT COUNT(*) FROM recon_results WHERE passed = 0") == 0
    assert count(conn, "SELECT COUNT(*) FROM etl_batch_log WHERE status != 'ok'") == 0

def test_daily_replay_makes_one_batch_per_day_then_one_bulk(built):
    conn, _ = built
    assert count(conn, "SELECT COUNT(*) FROM etl_batch_log") == 5 + 1

def test_open_days_with_zero_sales_are_rejected_not_loaded(built):
    conn, _ = built
    n_src = count(conn, "SELECT COUNT(*) FROM stg_train WHERE Open = 1 AND Sales = 0")
    assert n_src > 0
    assert count(conn, "SELECT COUNT(*) FROM reject_rows WHERE reason = 'open_zero_sales'") == n_src
    assert count(conn, "SELECT COUNT(*) FROM fact_daily_sales WHERE open_flag = 1 AND sales = 0") == 0

def test_loaded_plus_rejected_equals_source(built):
    conn, _ = built
    assert count(conn, "SELECT COUNT(*) FROM fact_daily_sales") + count(conn, "SELECT COUNT(*) FROM reject_rows") == \
           count(conn, "SELECT COUNT(*) FROM stg_train")

def test_store_dimension_keeps_missing_values_as_null(built):
    conn, _ = built
    assert count(conn, "SELECT COUNT(*) FROM dim_store WHERE competition_distance IS NULL") == 3
    assert count(conn, "SELECT COUNT(*) FROM dim_store WHERE competition_open_since IS NULL") >= 3
    assert count(conn, "SELECT COUNT(*) FROM dim_store WHERE promo2_flag = 0 AND promo_interval != 'none'") == 0

def test_rerun_is_idempotent(built):
    conn, raw = built
    before = count(conn, "SELECT COUNT(*) FROM fact_daily_sales")
    etl.build(conn, raw, replay_days=5, expected_stores=N_STORES)
    assert count(conn, "SELECT COUNT(*) FROM fact_daily_sales") == before
    assert count(conn, "SELECT COUNT(*) FROM recon_results WHERE passed = 0") == 0

def test_reconciliation_detects_a_deleted_row(built):
    conn, _ = built
    conn.execute("DELETE FROM fact_daily_sales WHERE rowid = (SELECT MIN(rowid) FROM fact_daily_sales)")
    conn.commit()
    res = etl.reconcile_range(conn, 1, "2013-01-01", "2015-07-31", N_STORES)
    assert any(not passed for _, _, _, passed in res)
