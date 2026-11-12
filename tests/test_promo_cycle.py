import sys, pathlib, sqlite3
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import check_promo_cycle as C

def test_alternating_promotions_are_detected():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE dim_date (date_key INTEGER, full_date TEXT)")
    conn.execute("CREATE TABLE fact_daily_sales (store_key INTEGER, date_key INTEGER, promo_flag INTEGER, open_flag INTEGER)")
    days = pd.date_range("2014-01-06", periods=140)
    for d in days: conn.execute("INSERT INTO dim_date VALUES (?,?)", (int(d.strftime("%Y%m%d")), d.strftime("%Y-%m-%d")))
    for s in (1, 2):
        for i, d in enumerate(days):
            promo = int((i // 7) % 2 == 0)                       # on one week, off the next
            conn.execute("INSERT INTO fact_daily_sales VALUES (?,?,?,1)", (s, int(d.strftime("%Y%m%d")), promo))
    t = C.promo_match_rates(conn)
    assert t.loc[7, "same_promo_status"] == 0.0 and t.loc[14, "same_promo_status"] == 1.0 and t.loc[21, "same_promo_status"] == 0.0
