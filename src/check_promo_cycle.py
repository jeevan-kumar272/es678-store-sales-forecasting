"""Checks one idea from the paper: does the planned promotion alternate between weeks?
If it does, the promotion flag two weeks earlier matches today's flag far more often than the flag one week earlier,
which would explain why the two-week-old value is a much better naive forecast than last week's value.
Usage: python src/check_promo_cycle.py"""
import argparse, sqlite3
import pandas as pd
from utils import load_config, ROOT

def promo_match_rates(conn, lags=(7, 14, 21, 28)):
    d = pd.read_sql("SELECT f.store_key, d.full_date AS date, f.promo_flag AS promo FROM fact_daily_sales f JOIN dim_date d ON f.date_key = d.date_key WHERE f.open_flag = 1", conn)
    d["date"] = pd.to_datetime(d["date"]); rows = {}
    for k in lags:
        past = d.assign(date=d["date"] + pd.Timedelta(days=k)).rename(columns={"promo": "promo_past"})
        m = d.merge(past, on=["store_key", "date"])
        rows[k] = {"pairs": len(m), "same_promo_status": float((m.promo == m.promo_past).mean())}
    return pd.DataFrame(rows).T.rename_axis("days_earlier")

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(); ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"])); a = ap.parse_args()
    t = promo_match_rates(sqlite3.connect(a.db))
    print("Share of open store-days whose promotion flag equals the flag N days earlier:"); print(t.round(3).to_string())
    print("\nIf 14 and 28 days earlier are much higher than 7 and 21, the promotion alternates between weeks.")

if __name__ == "__main__":
    main()
