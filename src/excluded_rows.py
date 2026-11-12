"""Supplementary check on the frozen test predictions (reads saved files, fits nothing).
The headline table scores only rows where a seasonal naive forecast exists (the same weekday one week earlier was open).
Rows without one, for example the day after a closure, are left out of that table. This shows how the chosen model does on
them, and which dates they fall on, so the limitation can be stated with numbers.
Usage: python src/excluded_rows.py"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from utils import load_config, ROOT

def _row(d, p):
    e = (d.sales - d[p]).abs()
    return {"rows": len(d), "mae": e.mean(), "wape": e.sum() / d.sales.sum(), "mean_sales": d.sales.mean()}

def excluded_summary(test_preds, features_df, chosen):
    ctx = features_df[["store_key", "date", "holiday_code", "dow"]]
    d = test_preds.merge(ctx, on=["store_key", "date"], how="left")
    scored = d.baseline_seasonal_naive.notna()
    table = pd.DataFrame({"all_test_rows": _row(d, chosen), "scored_in_headline_table": _row(d[scored], chosen),
                          "left_out_of_headline_table": _row(d[~scored], chosen)}).T
    ex = d[~scored].copy(); ex["weekday"] = pd.to_datetime(ex.date).dt.day_name()
    by_date = ex.groupby(["date", "weekday"]).size().rename("rows_left_out").reset_index().sort_values("rows_left_out", ascending=False).head(10)
    return {"table": table, "by_date": by_date, "share_left_out": float((~scored).mean())}

def main():
    cfg = load_config(); P = ROOT / "data" / "processed"
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(P / ("features_h%d.pkl" % cfg["horizon_days"])))
    ap.add_argument("--test-preds", default=str(P / "test_preds.pkl"))
    ap.add_argument("--choice", default=str(ROOT / "artifacts" / "model_choice.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts"))
    a = ap.parse_args()
    chosen = json.load(open(a.choice))["chosen"]
    r = excluded_summary(pd.read_pickle(a.test_preds), pd.read_pickle(a.features), chosen)
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("Chosen model '%s': %.1f percent of test rows are left out of the headline table (no seasonal naive forecast)\n" % (chosen, 100 * r["share_left_out"]))
    print(r["table"].round(3).to_string()); print("\nDates with the most rows left out:"); print(r["by_date"].to_string(index=False))
    r["table"].to_csv(out / "excluded_rows_summary.csv"); r["by_date"].to_csv(out / "excluded_rows_by_date.csv", index=False)

if __name__ == "__main__":
    main()
