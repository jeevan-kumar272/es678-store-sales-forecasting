"""Phase 5: build the point-in-time feature table from the warehouse.

For a target store-day T and horizon h (default 7), every history feature uses sales from days up to T minus h only.
Things known in advance (calendar, planned promotions, store attributes) may refer to T itself.
Customers is never read. History is computed with SQL window functions over a dense store-by-day grid, so a
missing day (for example the 2014 refurbishment gap) is a missing value, never a zero.

Usage:  python src/features.py [--db warehouse/rossmann.db] [--horizon 7] [--out data/processed/features_h7.pkl]
"""
import argparse, pathlib
import numpy as np, pandas as pd
from dateutil.easter import easter
from utils import load_config, ROOT

MIN_OBS = {7: 3, 28: 10, 56: 10}          # a window with fewer open days than this gives an empty feature
SLOPE_W, PROMO_W, SLOPE_MIN, PROMO_MIN, EXP_MIN = 28, 56, 10, 3, 10
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
          "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12}
ID_COLS = ["store_key", "date", "day_idx", "sales", "split"]

def make_dense(conn):
    """One row per store per calendar day. y is sales on open days, empty when closed or missing."""
    conn.execute("DROP TABLE IF EXISTS temp.dense")
    conn.execute("""
        CREATE TEMP TABLE dense AS
        SELECT s.store_key AS store_key, d.date_key AS date_key, d.full_date AS full_date,
               CAST(julianday(d.full_date) - julianday((SELECT MIN(full_date) FROM dim_date)) AS INTEGER) AS day_idx,
               CASE WHEN f.open_flag = 1 THEN f.sales END AS y,
               f.promo_flag AS promo, f.holiday_key AS hol, f.school_holiday_flag AS school
        FROM dim_store s CROSS JOIN dim_date d
        LEFT JOIN fact_daily_sales f ON f.store_key = s.store_key AND f.date_key = d.date_key""")

def history_query(h, lags, windows, warmup):
    part = "PARTITION BY store_key ORDER BY day_idx"
    past = lambda W: "ROWS BETWEEN %d PRECEDING AND %d PRECEDING" % (h + W - 1, h)
    ov = lambda fr: "OVER (%s %s)" % (part, fr)
    inner = ["store_key", "full_date AS date", "day_idx", "y AS sales", "promo", "hol AS holiday_key", "school"]
    for k in lags:
        inner.append("LAG(y, %d) OVER (%s) AS lag_%d" % (k, part, k))
    for W in windows:
        inner += ["AVG(y) %s AS m_%d" % (ov(past(W)), W), "COUNT(y) %s AS n_%d" % (ov(past(W)), W)]
    fr = ov(past(SLOPE_W))
    inner += ["COUNT(y) %s AS s_n" % fr,
              "SUM(CASE WHEN y IS NOT NULL THEN day_idx END) %s AS s_x" % fr,
              "SUM(y) %s AS s_y" % fr,
              "SUM(CASE WHEN y IS NOT NULL THEN day_idx * y END) %s AS s_xy" % fr,
              "SUM(CASE WHEN y IS NOT NULL THEN day_idx * day_idx END) %s AS s_xx" % fr]
    fr = ov(past(PROMO_W))
    inner += ["AVG(CASE WHEN promo = 1 THEN y END) %s AS pm" % fr, "COUNT(CASE WHEN promo = 1 THEN y END) %s AS pn" % fr,
              "AVG(CASE WHEN promo = 0 THEN y END) %s AS nm" % fr, "COUNT(CASE WHEN promo = 0 THEN y END) %s AS nn" % fr]
    fr = ov("ROWS BETWEEN UNBOUNDED PRECEDING AND %d PRECEDING" % h)
    inner += ["AVG(y) %s AS e_m" % fr, "COUNT(y) %s AS e_n" % fr]
    inner.append("SUM(promo) %s AS promo_prev7" % ov("ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING"))
    outer = ["store_key", "date", "day_idx", "sales", "promo", "holiday_key", "school"]
    outer += ["lag_%d" % k for k in lags]
    for W in windows:
        outer += ["CASE WHEN n_%d >= %d THEN m_%d END AS roll_mean_%d" % (W, MIN_OBS[W], W, W), "n_%d AS roll_n_%d" % (W, W)]
    outer += ["CASE WHEN s_n >= %d AND (s_n * s_xx - s_x * s_x) > 0 THEN (s_n * s_xy - s_x * s_y) * 1.0 / (s_n * s_xx - s_x * s_x) END AS slope_%d" % (SLOPE_MIN, SLOPE_W),
              "CASE WHEN pn >= %d THEN pm END AS promo_mean_%d" % (PROMO_MIN, PROMO_W),
              "CASE WHEN nn >= %d THEN nm END AS nonpromo_mean_%d" % (PROMO_MIN, PROMO_W),
              "CASE WHEN e_n >= %d THEN e_m END AS store_mean_past" % EXP_MIN,
              "COALESCE(promo_prev7, 0) AS promo_prev7"]
    return "SELECT %s FROM (SELECT %s FROM dense) WHERE sales IS NOT NULL AND day_idx >= %d ORDER BY store_key, day_idx" % (
        ", ".join(outer), ", ".join(inner), warmup + h)

def _days_since_holiday(conn, cap=30):
    hk = pd.read_sql("SELECT date_key, MAX(holiday_key) AS hk FROM fact_daily_sales GROUP BY date_key ORDER BY date_key", conn)
    out, last = {}, cap
    for k, hv in zip(hk.date_key, hk.hk):
        last = 0 if hv > 1 else min(last + 1, cap)
        out[int(k)] = last
    return out

def build_features(conn, cfg, horizon=None):
    h = int(horizon or cfg["horizon_days"])
    f = cfg["features"]
    lags = sorted({h} | {l for l in f["lags_days"] if l >= h} | {364})
    windows = list(f["rolling_windows"])
    make_dense(conn)
    df = pd.read_sql(history_query(h, lags, windows, cfg["split"]["warmup_days"]), conn)

    d = pd.to_datetime(df["date"])
    df["dow"] = d.dt.day_of_week + 1; df["dom"] = d.dt.day; df["week"] = d.dt.isocalendar().week.astype(int).values
    df["month"] = d.dt.month; df["is_month_end3"] = ((d.dt.days_in_month - d.dt.day) <= 2).astype(int)
    xmas = pd.to_datetime(d.dt.year.astype(str) + "-12-25")
    df["days_to_christmas"] = (xmas - d).dt.days.clip(-10, 45)
    easter_dates = pd.to_datetime(d.dt.year.map(lambda y: easter(int(y)).isoformat()))
    df["days_from_easter"] = (d - easter_dates).dt.days.clip(-21, 21)
    dk = df["date"].str.replace("-", "").astype(int)
    since = _days_since_holiday(conn)
    df["days_since_holiday"] = dk.map(since)
    df["holiday_code"] = df["holiday_key"] - 1                 # 0 none, 1 public, 2 Easter, 3 Christmas
    df["roll_ratio_7_56"] = df["roll_mean_7"] / df["roll_mean_56"]

    st = pd.read_sql("SELECT * FROM dim_store", conn)
    st["store_type"] = st.store_type.map({"a": 0, "b": 1, "c": 2, "d": 3})
    st["assortment"] = st.assortment.map({"a": 0, "b": 1, "c": 2})
    st["comp_dist_missing"] = st.competition_distance.isna().astype(int)
    cs = pd.to_datetime(st.competition_open_since)
    st["comp_y"] = cs.dt.year; st["comp_m"] = cs.dt.month
    st["comp_open_missing"] = cs.isna().astype(int)
    st["promo2_since_dt"] = pd.to_datetime(st.promo2_since)
    mm = np.zeros((st.store_key.max() + 1, 12), dtype=bool)
    for k, iv in zip(st.store_key, st.promo_interval):
        for m in str(iv).replace(" ", "").split(","):
            if m in MONTHS: mm[k, MONTHS[m] - 1] = True
    keep = ["store_key", "store_type", "assortment", "competition_distance", "comp_dist_missing", "comp_y", "comp_m",
            "comp_open_missing", "promo2_flag", "promo2_since_dt"]
    df = df.merge(st[keep], on="store_key", how="left")
    d = pd.to_datetime(df["date"])
    df["comp_open_months"] = (d.dt.year - df.comp_y) * 12 + (d.dt.month - df.comp_m)
    df["promo2_active"] = ((df.promo2_flag == 1) & (d >= df.promo2_since_dt) &
                           mm[df.store_key.values, d.dt.month.values - 1]).astype(int)
    df = df.drop(columns=["comp_y", "comp_m", "promo2_since_dt", "holiday_key"])

    mult = [l for l in lags if l % 7 == 0 and l != 364][:4]    # same-weekday lags available at this horizon
    df["baseline_seasonal_naive"] = df["lag_%d" % mult[0]]
    df["baseline_wd_mean4"] = df[["lag_%d" % l for l in mult]].mean(axis=1, skipna=True)

    c = cfg["split"]
    df["split"] = np.where(df["date"] <= c["dev_end"], "dev", np.where(df["date"] <= c["test_end"], "test", "after"))
    fl = df.select_dtypes("float64").columns
    df[fl] = df[fl].astype("float32")
    return df

def feature_columns(df):
    return [c for c in df.columns if c not in ID_COLS and not c.startswith("baseline_")]

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"]))
    ap.add_argument("--horizon", type=int, default=cfg["horizon_days"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import sqlite3
    conn = sqlite3.connect(a.db)
    df = build_features(conn, cfg, a.horizon)
    out = pathlib.Path(a.out or ROOT / "data" / "processed" / ("features_h%d.pkl" % a.horizon))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    fc = feature_columns(df)
    print("rows:", len(df), " features:", len(fc), " saved to", out)
    print("dates:", df["date"].min(), "to", df["date"].max())
    print(df["split"].value_counts().to_string())
    print("\nShare of empty values per feature (empty is fine, it means 'not enough history'):")
    print(df[fc].isna().mean().round(3).sort_values(ascending=False).head(12).to_string())

if __name__ == "__main__":
    main()
