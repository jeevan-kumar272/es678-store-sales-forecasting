CREATE TABLE IF NOT EXISTS dim_date (
  date_key INTEGER PRIMARY KEY, full_date TEXT UNIQUE, day_of_week INTEGER, week_of_year INTEGER,
  month INTEGER, quarter INTEGER, year INTEGER, is_weekend INTEGER);
CREATE TABLE IF NOT EXISTS dim_store (
  store_key INTEGER PRIMARY KEY, store_id INTEGER UNIQUE, store_type TEXT, assortment TEXT,
  competition_distance REAL, competition_open_since TEXT, promo2_flag INTEGER, promo2_since TEXT,
  promo_interval TEXT);
CREATE TABLE IF NOT EXISTS dim_holiday (
  holiday_key INTEGER PRIMARY KEY, code TEXT UNIQUE, description TEXT);
CREATE TABLE IF NOT EXISTS fact_daily_sales (
  date_key INTEGER REFERENCES dim_date, store_key INTEGER REFERENCES dim_store,
  holiday_key INTEGER REFERENCES dim_holiday, sales REAL, customers INTEGER,   -- customers is never a feature
  open_flag INTEGER, promo_flag INTEGER, school_holiday_flag INTEGER,
  PRIMARY KEY (date_key, store_key));
CREATE TABLE IF NOT EXISTS fact_forecast (
  run_id TEXT, model_version TEXT, store_key INTEGER REFERENCES dim_store, origin_date TEXT,
  target_date_key INTEGER REFERENCES dim_date, horizon INTEGER, y_pred REAL, p10 REAL, p90 REAL);
CREATE TABLE IF NOT EXISTS etl_batch_log (
  batch_id INTEGER PRIMARY KEY, load_date TEXT, rows_in INTEGER, rows_loaded INTEGER,
  rows_rejected INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS recon_results (
  batch_id INTEGER REFERENCES etl_batch_log, check_name TEXT, source_value REAL,
  warehouse_value REAL, passed INTEGER);
