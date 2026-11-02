"""Gate 0 and data audit for Rossmann Store Sales.
Usage:  python src/audit.py            (reads data/raw)
        python src/audit.py data/fake  (reads another folder, e.g. generated test data)
Writes artifacts/audit_report.json and prints a summary. Every number here feeds decision_log.md."""
import sys, json, pathlib
import numpy as np, pandas as pd
from utils import load_config, ROOT

def main(raw_dir=None):
    cfg = load_config()
    raw = pathlib.Path(raw_dir or ROOT / cfg["data"]["raw_dir"])
    train = pd.read_csv(raw / "train.csv", dtype={"StateHoliday": str}, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(raw / "store.csv")
    rep = {}

    # Gate 0: size and range
    rep["rows"] = int(len(train)); rep["stores"] = int(train.Store.nunique())
    rep["date_min"] = str(train.Date.min().date()); rep["date_max"] = str(train.Date.max().date())
    n_days = (train.Date.max() - train.Date.min()).days + 1
    rep["calendar_days"] = int(n_days)
    rep["gate0_rows_match"] = bool(abs(len(train) - cfg["data"]["expected_rows"]) <= 0.01 * cfg["data"]["expected_rows"])
    rep["gate0_stores_match"] = bool(rep["stores"] == cfg["data"]["expected_stores"])
    rep["duplicate_store_dates"] = int(train.duplicated(["Store", "Date"]).sum())

    # Missing store-days (rows absent from the file altogether)
    stores = np.sort(train.Store.unique()); dates = pd.date_range(train.Date.min(), train.Date.max())
    full = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"])
    miss = full.difference(train.set_index(["Store", "Date"]).index).to_frame(index=False).sort_values(["Store", "Date"])
    rep["missing_store_days"] = int(len(miss)); rep["stores_with_missing_days"] = int(miss.Store.nunique())
    if len(miss):
        miss["new_run"] = miss.groupby("Store")["Date"].diff().dt.days.ne(1)
        miss["run_id"] = miss.groupby("Store")["new_run"].cumsum()
        runs = miss.groupby(["Store", "run_id"]).agg(start=("Date", "min"), end=("Date", "max"), days=("Date", "size")).reset_index()
        long = runs[runs.days >= 30].sort_values("days", ascending=False)
        rep["stores_with_gap_30plus_days"] = int(long.Store.nunique())
        rep["longest_gap_days"] = int(runs.days.max())
        rep["long_gap_examples"] = [{"store": int(r.Store), "start": str(r.start.date()), "end": str(r.end.date()), "days": int(r.days)} for r in long.head(5).itertuples()]
        rep["missing_days_by_month"] = {str(k): int(v) for k, v in miss.Date.dt.to_period("M").value_counts().sort_index().items()}

    # Closed days and zero sales
    rep["closed_share"] = round(float((train.Open == 0).mean()), 4)
    rep["open_but_zero_sales"] = int(((train.Open == 1) & (train.Sales == 0)).sum())
    rep["closed_but_positive_sales"] = int(((train.Open == 0) & (train.Sales > 0)).sum())

    # Inconsistent codes and flags
    rep["state_holiday_values"] = {str(k): int(v) for k, v in train.StateHoliday.value_counts(dropna=False).items()}
    rep["promo_share"] = round(float(train.Promo.mean()), 4); rep["school_holiday_share"] = round(float(train.SchoolHoliday.mean()), 4)
    rep["store_file_missing"] = {k: int(v) for k, v in store.isna().sum().items() if v > 0}

    # Distribution on open days with positive sales
    o = train[(train.Open == 1) & (train.Sales > 0)]
    q = o.Sales.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(0)
    rep["sales_quantiles_open_days"] = {str(k): float(v) for k, v in q.items()}
    rep["sales_mean_open_days"] = round(float(o.Sales.mean()), 1); rep["sales_skew_raw"] = round(float(o.Sales.skew()), 2)
    rep["sales_skew_log1p"] = round(float(np.log1p(o.Sales).skew()), 2)
    rep["corr_sales_customers_open_days"] = round(float(o.Sales.corr(o.Customers)), 3)  # shows why Customers is excluded

    out = ROOT / "artifacts"; out.mkdir(exist_ok=True)
    (out / "audit_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        print(f"{k}: {v}")
    print("\nSaved artifacts/audit_report.json. Now write one decision_log.md line per issue found.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
