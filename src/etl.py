"""Phase 3: build the Rossmann star schema in SQLite, load it day by day, reconcile every batch.
Usage:  python src/etl.py [--raw data/raw] [--db warehouse/rossmann.db] [--replay-days 30] [--fresh]
Steps:  stage raw files -> dimensions -> replay the first N days one day at a time -> bulk load the rest.
Every batch writes etl_batch_log and recon_results. A failed check stops the run."""
import argparse, hashlib, pathlib, sqlite3, datetime as dt
import numpy as np, pandas as pd
from utils import load_config, ROOT

HOLIDAYS = [(1, "0", "no holiday"), (2, "a", "public holiday"), (3, "b", "Easter holiday"), (4, "c", "Christmas")]
HOL_KEY = {c: k for k, c, _ in HOLIDAYS}

def connect(db_path):
    pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_schema(conn):
    conn.executescript((ROOT / "sql" / "schema.sql").read_text())

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def stage(conn, raw):
    """Copy the raw files into staging tables untouched, adding source_row and file checksums."""
    raw = pathlib.Path(raw)
    train = pd.read_csv(raw / "train.csv", dtype={"StateHoliday": str}, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(raw / "store.csv")
    train.insert(0, "source_row", np.arange(1, len(train) + 1))
    train["Date"] = train["Date"].dt.strftime("%Y-%m-%d")
    train.to_sql("stg_train", conn, if_exists="replace", index=False)
    store.to_sql("stg_store", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_stg_train_date ON stg_train(Date)")
    conn.execute("DELETE FROM stg_files")
    for name, n in (("train.csv", len(train)), ("store.csv", len(store))):
        conn.execute("INSERT INTO stg_files VALUES (?,?,?)", (name, sha256(raw / name), n))
    conn.commit()

def _clean(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

def load_dimensions(conn):
    lo, hi = conn.execute("SELECT MIN(Date), MAX(Date) FROM stg_train").fetchone()
    rows = []
    for d in pd.date_range(lo, hi):
        dow = d.isoweekday()
        rows.append((int(d.strftime("%Y%m%d")), d.strftime("%Y-%m-%d"), dow, int(d.isocalendar()[1]),
                     d.month, (d.month - 1) // 3 + 1, d.year, int(dow >= 6)))
    conn.executemany("INSERT OR IGNORE INTO dim_date VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.executemany("INSERT OR IGNORE INTO dim_holiday VALUES (?,?,?)", HOLIDAYS)
    st = pd.read_sql("SELECT * FROM stg_store", conn)
    out = []
    for r in st.itertuples(index=False):
        comp_since = None
        if pd.notna(r.CompetitionOpenSinceYear) and pd.notna(r.CompetitionOpenSinceMonth):
            comp_since = "%04d-%02d-01" % (int(r.CompetitionOpenSinceYear), int(r.CompetitionOpenSinceMonth))
        p2_since = None
        if pd.notna(r.Promo2SinceYear) and pd.notna(r.Promo2SinceWeek):
            try:
                p2_since = dt.date.fromisocalendar(int(r.Promo2SinceYear), int(r.Promo2SinceWeek), 1).isoformat()
            except ValueError:
                p2_since = None
        interval = r.PromoInterval if pd.notna(r.PromoInterval) else "none"
        dist = _clean(float(r.CompetitionDistance)) if pd.notna(r.CompetitionDistance) else None  # kept NULL; imputed later on train only
        out.append((int(r.Store), int(r.Store), r.StoreType, r.Assortment, dist, comp_since, int(r.Promo2), p2_since, interval))
    conn.executemany("INSERT OR IGNORE INTO dim_store VALUES (?,?,?,?,?,?,?,?,?)", out)
    conn.commit()

def reconcile_range(conn, batch_id, start, end, expected_stores):
    """Compare staged source totals with warehouse totals for [start, end]. Returns list of (check, source, warehouse, passed)."""
    a, b = int(start.replace("-", "")), int(end.replace("-", ""))
    q = lambda sql, *p: conn.execute(sql, p).fetchone()[0] or 0
    src_rows = q("SELECT COUNT(*) FROM stg_train WHERE Date BETWEEN ? AND ?", start, end)
    rej_rows = q("SELECT COUNT(*) FROM reject_rows WHERE date BETWEEN ? AND ?", start, end)
    wh_rows = q("SELECT COUNT(*) FROM fact_daily_sales WHERE date_key BETWEEN ? AND ?", a, b)
    src_sales = q("SELECT SUM(Sales) FROM stg_train WHERE Date BETWEEN ? AND ?", start, end)
    src_cust = q("SELECT SUM(Customers) FROM stg_train WHERE Date BETWEEN ? AND ?", start, end)
    rej_sales = q("SELECT SUM(s.Sales) FROM stg_train s JOIN reject_rows r ON s.source_row = r.source_row WHERE s.Date BETWEEN ? AND ?", start, end)
    rej_cust = q("SELECT SUM(s.Customers) FROM stg_train s JOIN reject_rows r ON s.source_row = r.source_row WHERE s.Date BETWEEN ? AND ?", start, end)
    wh_sales = q("SELECT SUM(sales) FROM fact_daily_sales WHERE date_key BETWEEN ? AND ?", a, b)
    wh_cust = q("SELECT SUM(customers) FROM fact_daily_sales WHERE date_key BETWEEN ? AND ?", a, b)
    dups = q("SELECT COUNT(*) - COUNT(DISTINCT Store || '-' || Date) FROM stg_train WHERE Date BETWEEN ? AND ?", start, end)
    orphans = q("""SELECT COUNT(*) FROM fact_daily_sales f
                   LEFT JOIN dim_store s ON f.store_key = s.store_key LEFT JOIN dim_date d ON f.date_key = d.date_key
                   LEFT JOIN dim_holiday h ON f.holiday_key = h.holiday_key
                   WHERE f.date_key BETWEEN ? AND ? AND (s.store_key IS NULL OR d.date_key IS NULL OR h.holiday_key IS NULL)""", a, b)
    max_stores = q("SELECT MAX(c) FROM (SELECT COUNT(*) c FROM fact_daily_sales WHERE date_key BETWEEN ? AND ? GROUP BY date_key)", a, b)
    n_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    wh_days = q("SELECT COUNT(DISTINCT date_key) FROM fact_daily_sales WHERE date_key BETWEEN ? AND ?", a, b)
    ok = lambda x, y: abs(float(x) - float(y)) < 1e-6
    res = [("rows_in_equals_loaded_plus_rejected", src_rows, wh_rows + rej_rows, ok(src_rows, wh_rows + rej_rows)),
           ("sum_sales_matches", src_sales, wh_sales + rej_sales, ok(src_sales, wh_sales + rej_sales)),
           ("sum_customers_matches", src_cust, wh_cust + rej_cust, ok(src_cust, wh_cust + rej_cust)),
           ("duplicate_store_dates_in_source", dups, 0, dups == 0),
           ("foreign_key_orphans", 0, orphans, orphans == 0),
           ("max_stores_per_day_within_limit", expected_stores, max_stores, max_stores <= expected_stores),
           ("every_calendar_day_present", n_days, wh_days, n_days == wh_days)]
    conn.execute("DELETE FROM recon_results WHERE batch_id = ?", (batch_id,))
    conn.executemany("INSERT INTO recon_results VALUES (?,?,?,?,?)", [(batch_id, n, s, w, int(p)) for n, s, w, p in res])
    conn.commit()
    return res

def load_range(conn, start, end, expected_stores):
    """Load staged rows for [start, end] into the fact table, log the batch, reconcile, stop on failure."""
    df = pd.read_sql("SELECT * FROM stg_train WHERE Date BETWEEN ? AND ?", conn, params=(start, end))
    cur = conn.execute("INSERT INTO etl_batch_log (load_date, rows_in, status) VALUES (?,?,?)",
                       ("%s..%s" % (start, end), len(df), "running"))
    batch_id = cur.lastrowid
    df["holiday_key"] = df["StateHoliday"].map(HOL_KEY)
    bad_hol = df.holiday_key.isna()
    open_zero = (df.Open == 1) & (df.Sales == 0)
    reject = bad_hol | open_zero
    rej = df[reject]
    conn.execute("DELETE FROM reject_rows WHERE date BETWEEN ? AND ?", (start, end))
    conn.executemany("INSERT OR REPLACE INTO reject_rows VALUES (?,?,?)",
                     [(int(r.source_row), r.Date, "unknown_holiday_code" if pd.isna(r.holiday_key) else "open_zero_sales") for r in rej.itertuples()])
    clean = df[~reject]
    rows = pd.DataFrame({"date_key": clean.Date.str.replace("-", "").astype(int), "store_key": clean.Store,
                         "holiday_key": clean.holiday_key.astype(int), "sales": clean.Sales, "customers": clean.Customers,
                         "open_flag": clean.Open, "promo_flag": clean.Promo, "school": clean.SchoolHoliday}).values.tolist()
    conn.executemany("INSERT OR IGNORE INTO fact_daily_sales VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    res = reconcile_range(conn, batch_id, start, end, expected_stores)
    failed = [n for n, _, _, p in res if not p]
    wh_rows = conn.execute("SELECT COUNT(*) FROM fact_daily_sales WHERE date_key BETWEEN ? AND ?",
                           (int(start.replace("-", "")), int(end.replace("-", "")))).fetchone()[0]
    conn.execute("UPDATE etl_batch_log SET rows_loaded=?, rows_rejected=?, status=? WHERE batch_id=?",
                 (wh_rows, len(rej), "FAILED" if failed else "ok", batch_id))
    conn.commit()
    if failed:
        raise RuntimeError("Reconciliation failed for %s..%s: %s" % (start, end, failed))
    return batch_id

def build(conn, raw, replay_days=30, expected_stores=1115, verbose=False):
    init_schema(conn); stage(conn, raw); load_dimensions(conn)
    lo, hi = conn.execute("SELECT MIN(Date), MAX(Date) FROM stg_train").fetchone()
    days = pd.date_range(lo, hi)
    n = min(replay_days, len(days))
    for d in days[:n]:                                   # one day at a time, to prove the daily process
        s = d.strftime("%Y-%m-%d"); load_range(conn, s, s, expected_stores)
        if verbose: print("loaded day", s)
    if n < len(days):                                    # then the rest in one bulk batch
        load_range(conn, days[n].strftime("%Y-%m-%d"), hi, expected_stores)
        if verbose: print("bulk loaded", days[n].date(), "to", hi)

def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / cfg["data"]["raw_dir"]))
    ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"]))
    ap.add_argument("--replay-days", type=int, default=30)
    ap.add_argument("--fresh", action="store_true", help="delete the database first")
    a = ap.parse_args()
    if a.fresh and pathlib.Path(a.db).exists():
        pathlib.Path(a.db).unlink()
    conn = connect(a.db)
    build(conn, a.raw, a.replay_days, cfg["data"]["expected_stores"], verbose=True)
    print("\nWarehouse summary")
    for t in ["dim_date", "dim_store", "dim_holiday", "fact_daily_sales", "reject_rows", "etl_batch_log", "recon_results"]:
        print("  %-18s %d" % (t, conn.execute("SELECT COUNT(*) FROM " + t).fetchone()[0]))
    bad = conn.execute("SELECT COUNT(*) FROM recon_results WHERE passed = 0").fetchone()[0]
    print("  failed reconciliation checks:", bad)

if __name__ == "__main__":
    main()
